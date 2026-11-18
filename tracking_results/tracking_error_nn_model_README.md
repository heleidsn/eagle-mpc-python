# Tracking Error NN 模型说明

## 模型文件
- 模型: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_model.joblib`
- 输入特征顺序: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_feature_names.csv`
- 训练指标: `/home/helei/catkin_eagle_mpc/src/eagle-mpc-python/tracking_results/tracking_error_nn_metrics.csv`

## 输入 X
- 单帧特征维度: 25
- 窗口长度: 1 帧
- 实际输入维度: 25
- 单帧特征按以下顺序拼接：
  - 0: `ep_x`
  - 1: `ep_y`
  - 2: `ep_z`
  - 3: `ev_x`
  - 4: `ev_y`
  - 5: `ev_z`
  - 6: `a_ref_x`
  - 7: `a_ref_y`
  - 8: `a_ref_z`
  - 9: `j_ref_x`
  - 10: `j_ref_y`
  - 11: `j_ref_z`
  - 12: `yaw_ref`
  - 13: `yaw_rate_ref`
  - 14: `v_world_x`
  - 15: `v_world_y`
  - 16: `v_world_z`
  - 17: `omega_x`
  - 18: `omega_y`
  - 19: `omega_z`
  - 20: `body_z_world_x`
  - 21: `body_z_world_y`
  - 22: `body_z_world_z`
  - 23: `u_thrust`
  - 24: `thrust_margin`

## 输出 y
- 6维向量，拟合当前误差：
  - 位置误差 `e_p = p_ref - p` 的3个分量
  - 速度误差 `e_v = v_ref - v` 的3个分量

## 训练设置
- MLP 结构: `(128, 128)`
- 激活: ReLU, 优化器: Adam
- 训练轮数: 逐epoch warm-start, 共 `EPOCHS` 轮
- 训练/测试loss图: `analysis_plots/tracking_error_nn_loss.png`
