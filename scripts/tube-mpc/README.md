# Tube MPC 与动态 Tube MPC 实验

本目录用于验证 Tube 控制的误差集合、约束收紧和动态 tube 优化，包含一个线性跟踪示例，以及一个支持 SciPy / acados 后端的二维动态 Tube MPC 原型。目前是独立实验代码，尚未接入项目原有控制流程。

## 时间记录

| 时间 | 内容 |
| --- | --- |
| 2026-04-18 11:48–11:53 | `tube_mpc.py` 的本地文件创建至最后修改时间，对应线性 Tube 跟踪示例。 |
| 2026-04-18 12:30–15:12 | `dtmpc_prototype.py` 的本地文件创建至最后修改时间，对应二维动态 Tube MPC 原型。 |
| 2026-04-18 13:53–15:12 | `dtmpc_acados.py` 的本地文件创建至最后修改时间，对应 acados 后端。 |
| 2026-04-18 15:06 | `dtmpc_demo.png` 的本地文件最后修改时间。 |
| 2026-11-18 | 编写本 README；在仓库 `.gitignore` 中加入 `scripts/tube-mpc/c_generated_dtmpc/`，忽略 acados 自动生成代码及编译产物。本次未修改上述 Python 脚本。 |

日期按 Asia/Shanghai 时区记录。代码时间取自当前文件系统的创建时间和最后修改时间，并非 Git 提交历史，也不代表期间持续开发；复制、恢复文件等操作可能影响这些时间。当前这些新增脚本尚无 Git 提交记录，因此只能将本地文件时间作为开发时间的参考。

## 文件与功能

### `tube_mpc.py`：线性 Tube 跟踪示例

采用离散双积分器，状态为位置和每步位置增量，演示：

- 通过有限项集合和 `E ≈ Σ (A + BK)^i W` 近似计算误差集合，并用轴对齐盒子表示。
- 根据误差集合计算状态约束和名义输入约束的收紧量。
- 从正弦位置参考生成满足离散动力学的名义状态与前馈输入。
- 使用 `u = v + K(x - z)` 抵抗盒子范围内的随机扰动。
- 绘制误差集合、多次随机试验的误差轨迹，以及位置、速度增量和控制输入曲线。

这里的名义轨迹由参考直接构造，没有在线求解有限时域 MPC 优化问题。收紧约束主要用于计算和对照，脚本不会自动调整参考来保证约束可行。有限项集合和也不等于经过严格验证的鲁棒正不变集。

### `dtmpc_prototype.py`：二维动态 Tube MPC 原型

状态为 `[px, py, vx, vy]`，控制输入为二维力项，动力学包含名义二次阻力。预测模型额外维护 `phi`、带宽 `alpha`、位置误差界 `omega_p` 和速度误差界 `omega_v`。

当前支持：

- 联合优化名义控制序列与带宽变化率，让 tube 随预测运动和不确定性变化。
- 用随速度平方变化的阻力不确定性界，加上有界扰动，驱动 tube 演化。
- 在避障安全半径中加入机器人半径和位置 tube 半径。
- 用惩罚项近似处理收紧后的速度和合力幅值约束。
- 默认通过 SciPy 的 L-BFGS-B 求解，也可配置其他兼容的优化方法。
- 执行滚动时域仿真，并将上一轮控制序列移位后作为下一轮初值。
- 输出轨迹、tube 状态、优化带宽和最终目标距离。

默认示例从 `[0, -0.75]` 前往 `[3, 0.75]`，场景包含两个圆形障碍。默认采样周期为 0.1 秒，预测步数为 20，闭环执行 65 步，即仿真 6.5 秒。实际是否穿过障碍间隙、到达目标，取决于优化结果与参数，不能仅由示例设置保证。

### `dtmpc_acados.py`：可选 acados 后端

将上述问题表示为 CasADi 符号模型，构建 acados 离散时间最优控制问题：

- 8 维增广状态：`[px, py, vx, vy, phi, alpha, omega_p, omega_v]`。
- 3 维控制：`[ux, uy, v_alpha]`。
- 使用 SQP 求解，支持控制边界、包含 tube 半径的非线性硬避障约束、终端代价和热启动。
- 自动生成 C 函数、求解器接口和编译产物。

两个后端并非完全相同的优化问题：SciPy 默认通过软惩罚避障，acados 默认启用硬避障约束；当前 acados 在启用硬避障时，仅在终端保留缩放后的软避障惩罚。`solve()` 返回的 `cost` 仍按 Python 中的目标函数重新计算，因此不能直接当作 acados 内部目标值。

### 结果与生成文件

- `dtmpc_demo.png`：动态 Tube MPC 示例的结果图。运行脚本会覆盖此文件，已有图片不代表当前代码已重新验证。
- `c_generated_dtmpc/`：acados 生成的 `.c`、`.h`、JSON、Makefile 及编译产物，已加入 Git 忽略规则。通常修改 Python 模型后重新生成，无需手工编辑这些文件。

## 运行方式

以下命令均在仓库根目录执行。

### 基础依赖

在当前 Python 环境中安装：

```bash
python -m pip install numpy scipy matplotlib
```

运行线性示例：

```bash
python scripts/tube-mpc/tube_mpc.py
```

该脚本打印误差集合和约束收紧结果，并通过 `plt.show()` 显示图形，需要可用的图形环境。

运行默认 SciPy 动态示例：

```bash
python scripts/tube-mpc/dtmpc_prototype.py
```

脚本保存 `scripts/tube-mpc/dtmpc_demo.png`，并打印闭环运行耗时和最终位置到目标的距离。在无图形界面的环境中可运行：

```bash
MPLBACKEND=Agg python scripts/tube-mpc/dtmpc_prototype.py
```

### 使用 acados

需要另外安装 CasADi、编译 acados，并安装其 Python 接口 `acados_template`，配置好相应运行环境和编译工具链。仅安装基础依赖不能运行该后端。

可在本目录中执行：

```bash
cd scripts/tube-mpc
python - <<'PY'
from dtmpc_prototype import DTMPCConfig, run_closed_loop_demo

cfg = DTMPCConfig(mpc_backend="acados")
out = run_closed_loop_demo(save_path="dtmpc_demo.png", cfg=cfg)
print(f"Final distance to goal: {out['dist_to_goal']:.4f} m")
PY
```

第一次求解时会生成和编译求解器。可通过 `acados_code_export_dir` 指定其他输出目录。

## 当前限制

- 动态原型采用简化的 tube 误差界演化模型，属于受 Dynamic Tube MPC 思路启发的教学实现，没有完整复现论文的理论条件或证明。
- 动态示例闭环只施加优化所得的前馈控制，明确省略了辅助误差反馈；虽然加入了扰动，但不能据此宣称具备完整的鲁棒 Tube MPC 保证。
- 速度与合力幅值的收紧主要通过惩罚项实现，可能发生违反；控制各分量的盒约束与合力幅值约束也不是同一回事。
- 求解失败时，示例仍会使用当前迭代解继续仿真；acados 默认还会将状态码 2（达到最大迭代次数）视为可接受结果。代码没有额外的可行性验收或安全回退控制。
- tube 和带宽使用显式 Euler 离散化及截断处理，改变采样周期或带宽范围时需要重新检查数值行为。
- 当前没有自动化回归测试或后端性能对比。本 README 基于代码静态阅读整理，未在此次文档整理中重新运行完整仿真。
