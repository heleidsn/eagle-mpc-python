# Geometric Controller 自动调参（Auto Tune）

本文档说明 Sim Tracking 里 **Geometric controller** 的一键自动调参功能：
针对**当前规划轨迹**自动搜索几何控制器的 4 个增益
（`kp_pos / kd_vel / kR / kOmega`），目标是最小化 base 的**平均位置误差**与
**最大位置误差**。

涉及文件：`uam_suite_gui.py`
- `GeometricAutoTuneWorker(QThread)` —— 后台调参线程（优化算法）
- `_geo_pos_error_metrics(out, alpha)` —— 误差度量
- `_on_geo_autotune_clicked / _progress / _finished` —— GUI 交互
- 复用 `scripts/s500_uam_acados_state_tracking_mpc.run_closed_loop_track_full_state_plan_acados`

---

## 1. 为什么要自动调参

Geometric controller 是「位置/速度 PD → 期望姿态 → 姿态误差 PD → CTBR 内环」的
级联结构，没有 MPC 的前瞻能力，对 figure-8 这类**曲率反转、姿态连续变化**的
轨迹比较敏感。手调 `kp_pos / kd_vel / kR / kOmega` 4 个耦合增益既费时又难权衡
「跟得紧」与「不振荡」。自动调参把这一步变成一键操作。

被调的 4 个增益（来自 `_geometric_baseline_command`）：

```
a_des    = a_ref - kp_pos·e_p - kd_vel·e_v + g·e3        # 期望比力（位置环）
rate_cmd = -kR·e_R - kOmega·ω                            # 体角速度指令（姿态环）
```

`max_tilt`（倾角限幅）作为**安全限制不参与调参**，保持界面当前值。

---

## 2. 目标函数

把闭环结果里 base 的世界系位置与参考轨迹对齐后，取逐拍位置误差范数：

```
e(t) = ‖ p_meas(t) − p_ref(t) ‖          # p_ref 由 x_plan 在 sim 时间轴上插值
J    = mean_t e(t)  +  α · max_t e(t)     # 默认 α = 1
```

- `mean` 体现整体跟踪水平，`max` 抑制最差点（如 figure-8 交叉点的冲高）。
- 若仿真发散导致 `nan/inf`，该次评估记 `J = inf`（被自动淘汰）。

实现见 `_geo_pos_error_metrics()`，返回 `(mean, max, J)`。位置在状态的 `q[:3]`，
本身就是世界系，无需旋转。

> 说明：目标默认覆盖整条轨迹（含前后 hover 段）。hover 段误差接近 0，会轻微
> 稀释 `mean`；如需只评估运动段，可在度量函数里加时间窗裁剪。

---

## 3. 优化算法：模式搜索（坐标下降 + 步长收缩）

无依赖（不用 scipy）、鲁棒、评估次数可控、可中途停止。每次「评估」=
跑一遍 headless 闭环仿真。

记增益向量 `g = [kp_pos, kd_vel, kR, kOmega]`：

1. **初始化**：以界面当前增益为起点 `g₀`（夹到边界内），评估得 `best_J`。
   初始步长 `step = max(0.25·|g|, 2·STEP_MIN)`（各维独立）。
2. **坐标探索**：依次对每个维度 `d`，尝试 `g[d] ± step[d]`（夹到边界）：
   - 一旦某方向使 `J < best_J`，**接受**该改进、更新 `best_g/best_J`，转下一维；
   - 两个方向都不改进，则该维不动。
3. **步长收缩**：若一整轮（4 维）都没有改进，`step ← step / 2`。
4. **终止**：达到评估预算 `budget`、或所有维 `step < STEP_MIN`、或用户点击 Stop。
5. 输出 `best_g`，写回界面。

### 搜索边界与步长（`GeometricAutoTuneWorker` 类常量）

| 参数      | 下界 LOWER | 上界 UPPER | 最小步长 STEP_MIN |
|-----------|-----------|-----------|------------------|
| `kp_pos`  | 0.5       | 30.0      | 0.2              |
| `kd_vel`  | 0.3       | 30.0      | 0.2              |
| `kR`      | 1.0       | 30.0      | 0.3              |
| `kOmega`  | 0.0       | 5.0       | 0.02             |

`budget` 默认 40（界面可调 8–200），通常 30–60 次足以收敛到局部最优。

> 局部最优提示：模式搜索找到的是**起点附近**的局部最优，依赖界面里的初始增益。
> 想更全局可把初值设得不同再各跑一次，取最好。

---

## 4. 调参时的仿真条件

- **强制关闭扰动与 L1**（`disturbance=None, l1=None`）——只评估控制器本身，
  避免估计/补偿环节干扰增益评估。
- `control_mode="direct"`、`baseline="geometric"`、`ctbr.enabled=True`
  （geometric 恒走 CTBR 内环）。
- 轨迹、`T_sim / sim_dt / control_dt / dt_mpc / N`、状态限幅、CTBR 内环增益、
  `max_tilt` 等都取**界面当前值**——即在你当前的内环/离散配置下调外环增益。

---

## 5. 使用步骤（GUI）

1. Sim Tracking → 跟踪方式选 **Geometric controller**；此时 geometric 参数区
   下方出现 **「Auto tune (当前轨迹)」按钮 + budget + 状态行**。
2. 先完成 **Full state** 规划（s500 的 minimum snap / figure-8 会自动用微分平坦
   升级为 full-state）。
3. 点 **Auto tune**。状态行实时刷新：
   `#评估/预算  当前J  最佳J  (mean/max mm)  当前最佳增益`。
4. 运行中按钮变 **Stop**，可随时中止（保留当前最佳）。
5. 结束后**最佳增益自动写回** 4 个 spinbox，日志也会记录。随后正常
   **Run closed-loop tracking** 验证效果。

---

## 6. 性能与可改进项

- **每次评估会重建一次 acados solver**：`run_...` 函数内部无条件构建
  （geometric 实际用不到 solver），带来固定开销；40 次评估约一两分钟。
  可改进：给 geometric 增加跳过建 solver 的快路径来提速。
- **目标只用位置误差**：如需把姿态误差纳入权衡，可扩展 `_geo_pos_error_metrics`
  增加姿态项与权重。
- **只评估运动段**：在度量函数里按时间窗裁掉前后 hover，可更突出
  figure-8 交叉点等关键误差。
- **多起点 / 其它优化器**：当前是单起点模式搜索；如需更强全局性，可加
  随机多起点或粗网格预筛。

---

## 7. 关键数据流

```
[GUI] Auto tune 点击
   └─ _on_geo_autotune_clicked()           # 校验 full-state、组装 base 参数
        └─ GeometricAutoTuneWorker(QThread)
             ├─ _evaluate(g):               # 一次评估
             │    run_closed_loop_track_full_state_plan_acados(
             │        baseline="geometric", geometric=g,
             │        disturbance=None, l1=None, ctbr.enabled=True, ...)
             │    └─ _geo_pos_error_metrics(out, α) → (mean, max, J)
             ├─ 模式搜索循环（坐标下降 + 步长减半）
             ├─ progress 信号 → _on_geo_autotune_progress()   # 状态行
             └─ finished 信号 → _on_geo_autotune_finished()   # 写回 spinbox
```
