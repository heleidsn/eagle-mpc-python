# Tracking Error NN Model说明

## 模型文件
- 模型路径: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_model.joblib`
- 输入特征顺序: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_feature_names.csv`
- 单次切分评估: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_metrics.csv`
- GroupKFold评估: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_kfold_metrics.csv`

## 输入 (X)
输入是9维浮点向量，顺序必须严格一致：
0. `v_track_world_norm`
1. `v_plan_norm`
2. `v_diff_norm`
3. `a_plan_norm`
4. `jerk_plan_norm`
5. `snap_plan_norm`
6. `vx_world_error`
7. `vy_world_error`
8. `vz_world_error`

各特征定义：
- `v_track_world_norm`: tracking机体系速度经四元数旋转到世界系后的速度模长
- `v_plan_norm`: planning速度模长
- `v_diff_norm`: `|v_track_world_norm - v_plan_norm|`
- `a_plan_norm`: planning加速度模长
- `jerk_plan_norm`: planning jerk模长（由加速度对时间求导）
- `snap_plan_norm`: planning snap模长（由jerk对时间求导）
- `vx_world_error, vy_world_error, vz_world_error`: 世界系速度分量误差

## 输出 (y)
- 网络输出为 `log1p(position_error)` 域的预测值，脚本中会做 `expm1` 还原。
- 最终物理输出是 `position_error`，定义为：
  `sqrt((px_track-px_plan)^2 + (py_track-py_plan)^2 + (pz_track-pz_plan)^2)`，单位米。

## 推理注意事项
1. 推理时必须使用同样的输入定义和顺序。
2. 模型文件是包含 `StandardScaler + MLPRegressor` 的 Pipeline，可直接 `model.predict(X)`。
3. `predict` 后需执行：`y_pred = np.expm1(y_pred_log)`，并建议 `clip(y_pred, 0, +inf)`。

## 训练结构
- MLP: hidden layers `(128, 128)`, ReLU, Adam, alpha=1e-3
- 目标变换: `log1p(error)`
- 验证策略: 按轨迹 GroupKFold 交叉验证
