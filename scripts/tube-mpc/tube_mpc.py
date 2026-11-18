import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def box_corners(bounds):
    """
    给定盒子边界 bounds = [(xmin,xmax), (ymin,ymax), ...]
    返回所有顶点
    """
    return np.array(list(product(*bounds)), dtype=float)


def linear_map_of_box(M, bounds):
    """
    计算线性变换 M(W) 的顶点，其中 W 是一个盒子
    """
    corners = box_corners(bounds)   # shape: (num_vertices, dim)
    mapped = (M @ corners.T).T
    return mapped


def box_outer_approx(points):
    """
    对点集做轴对齐盒子外逼近
    返回 [(xmin,xmax), (ymin,ymax), ...]
    """
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return list(zip(mins, maxs))


def minkowski_sum_boxes(box1, box2):
    """
    两个轴对齐盒子的 Minkowski sum
    """
    out = []
    for (a1, b1), (a2, b2) in zip(box1, box2):
        out.append((a1 + a2, b1 + b2))
    return out


def print_box(name, box):
    print(f"{name}:")
    for i, (a, b) in enumerate(box):
        print(f"  dim {i+1}: [{a:.6f}, {b:.6f}]")
    print()


def plot_box(ax, box, color="blue", label=None, linestyle="-"):
    """
    仅支持二维盒子画图
    """
    x_min, x_max = box[0]
    y_min, y_max = box[1]

    xs = [x_min, x_max, x_max, x_min, x_min]
    ys = [y_min, y_min, y_max, y_max, y_min]
    ax.plot(xs, ys, color=color, linestyle=linestyle, label=label)


def sample_disturbance_sequence(W_box, steps):
    """
    从扰动盒子中随机采样一串扰动
    """
    dim = len(W_box)
    w_seq = []
    for _ in range(steps):
        w = np.array([np.random.uniform(a, b) for (a, b) in W_box])
        w_seq.append(w)
    return w_seq


def simulate_error_dynamics(Ak, W_box, steps=50, num_trials=100, e0=None):
    """
    仿真误差动力学 e_{k+1} = Ak e_k + w_k
    """
    if e0 is None:
        e0 = np.zeros(Ak.shape[0])

    trajs = []
    for _ in range(num_trials):
        e = e0.copy()
        traj = [e.copy()]
        w_seq = sample_disturbance_sequence(W_box, steps)
        for w in w_seq:
            e = Ak @ e + w
            traj.append(e.copy())
        trajs.append(np.array(traj))
    return trajs


def sin_position_reference(num_samples, amp, omega):
    """
    正弦位置参考 p_k = amp * sin(omega * k)，k = 0 .. num_samples-1
    """
    k = np.arange(num_samples, dtype=float)
    return amp * np.sin(omega * k)


def nominal_state_and_input_from_p_ref(p_ref, k):
    """
    与 A=[[1,1],[0,1]], B=[[0],[1]] 相容的名义状态与前馈名义输入:
    z_k = [p_k, p_{k+1}-p_k],  v_k = (p_{k+2}-p_{k+1}) - (p_{k+1}-p_k)
    使得 z_{k+1} = A z_k + B v_k 精确等于下一时刻参考。
    """
    p_k, p_k1, p_k2 = p_ref[k], p_ref[k + 1], p_ref[k + 2]
    v_disc_k = p_k1 - p_k
    v_disc_k1 = p_k2 - p_k1
    z_k = np.array([p_k, v_disc_k], dtype=float)
    v_k = v_disc_k1 - v_disc_k
    return z_k, v_k


def simulate_tube_tracking_once(A, B, K, p_ref, W_box, steps, e0=None, rng=None):
    """
    单次闭环仿真：w_k 从 W 中均匀采样。返回状态/名义参考/总输入时间序列。
    """
    if e0 is None:
        e0 = np.zeros(A.shape[0])
    if rng is None:
        rng = np.random.default_rng()

    dim = len(W_box)
    z0, _ = nominal_state_and_input_from_p_ref(p_ref, 0)
    x = z0 + e0
    x_hist = [x.copy()]
    z_hist = [z0.copy()]
    u_hist = []
    for t in range(steps):
        z_t, v_t = nominal_state_and_input_from_p_ref(p_ref, t)
        w = np.array([rng.uniform(a, b) for (a, b) in W_box])
        u = v_t + (K @ (x - z_t)).item()
        u_hist.append(u)
        x = A @ x + B.flatten() * u + w
        z_next, _ = nominal_state_and_input_from_p_ref(p_ref, t + 1)
        x_hist.append(x.copy())
        z_hist.append(z_next.copy())
    return np.array(x_hist), np.array(z_hist), np.array(u_hist)


def simulate_tube_tracking(A, B, K, p_ref, W_box, steps, num_trials=100, e0=None):
    """
    跟踪正弦（或其它）p_ref：名义 z 由 p_ref 生成，u_k = v_k + K(x_k - z_k)，
    x_{k+1} = A x_k + B u_k + w_k。返回每条 trial 的误差轨迹与无扰动名义轨迹（用于对照）。
    """
    if e0 is None:
        e0 = np.zeros(A.shape[0])

    e_trajs = []
    for _ in range(num_trials):
        w_seq = sample_disturbance_sequence(W_box, steps)
        z0, _ = nominal_state_and_input_from_p_ref(p_ref, 0)
        x = z0 + e0
        e = x - z0
        e_traj = [e.copy()]
        for t in range(steps):
            z_t, v_t = nominal_state_and_input_from_p_ref(p_ref, t)
            u = v_t + (K @ (x - z_t)).item()
            x = A @ x + B.flatten() * u + w_seq[t]
            z_next, _ = nominal_state_and_input_from_p_ref(p_ref, t + 1)
            e = x - z_next
            e_traj.append(e.copy())
        e_trajs.append(np.array(e_traj))

    # 无扰动：名义参考即闭环（误差恒为 0），用于画参考曲线
    z0, _ = nominal_state_and_input_from_p_ref(p_ref, 0)
    x = z0.copy()
    x_nom = [x.copy()]
    z_nom = [z0.copy()]
    v_nom = []
    for t in range(steps):
        z_t, v_t = nominal_state_and_input_from_p_ref(p_ref, t)
        v_nom.append(v_t)
        u = v_t + (K @ (x - z_t)).item()
        x = A @ x + B.flatten() * u
        z_next, _ = nominal_state_and_input_from_p_ref(p_ref, t + 1)
        x_nom.append(x.copy())
        z_nom.append(z_next.copy())
    return e_trajs, np.array(x_nom), np.array(z_nom), np.array(v_nom)


if __name__ == "__main__":
    # =========================
    # 1. 定义系统
    # =========================
    A = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    B = np.array([
        [0.0],
        [1.0]
    ])

    K = np.array([[-0.5, -1.0]])   # shape (1,2)

    Ak = A + B @ K

    print("A =\n", A)
    print("B =\n", B)
    print("K =\n", K)
    print("A_K = A + B K =\n", Ak)

    eigvals = np.linalg.eigvals(Ak)
    print("eig(A_K) =", eigvals)
    print("spectral radius =", max(abs(eigvals)))
    print()

    # =========================
    # 2. 扰动集合 W
    #    W = { w : |w1| <= 0.05, |w2| <= 0.02 }
    # =========================
    W_box = [(-0.05, 0.05), (-0.02, 0.02)]
    print_box("W", W_box)

    # =========================
    # 3. 近似计算 E = sum_{i=0}^N A_K^i W
    # =========================
    N = 20   # 截断项数，可调大一些
    E_box = [(0.0, 0.0), (0.0, 0.0)]   # 初始为空集的盒子近似（这里用0盒子）

    term_boxes = []

    Ak_power = np.eye(2)
    for i in range(N + 1):
        # 计算 A_K^i W 的顶点
        mapped_vertices = linear_map_of_box(Ak_power, W_box)

        # 对该集合做盒子外逼近
        term_box = box_outer_approx(mapped_vertices)
        term_boxes.append(term_box)

        # 累加 Minkowski sum 的盒子近似
        E_box = minkowski_sum_boxes(E_box, term_box)

        print_box(f"A_K^{i} W (box approx)", term_box)

        Ak_power = Ak_power @ Ak

    print_box("Approximate E", E_box)

    # =========================
    # 4. 根据 E 收紧约束
    #    假设原始状态约束:
    #       |x1| <= 5, |x2| <= 2
    #    原始输入约束:
    #       |u| <= 1
    # =========================
    X_box = [(-5.0, 5.0), (-2.0, 2.0)]

    # X ⊖ E 的盒子近似
    # 若 X=[-a,a], E=[-e,e], 则 X⊖E=[-(a-e), a-e]
    E_abs = [max(abs(a), abs(b)) for a, b in E_box]
    X_tight = [
        (X_box[0][0] + E_abs[0], X_box[0][1] - E_abs[0]),
        (X_box[1][0] + E_abs[1], X_box[1][1] - E_abs[1]),
    ]

    print_box("Original state constraint X", X_box)
    print_box("Tightened state constraint X_tight", X_tight)

    # 输入约束 tightening
    # u = v + K e
    # 对盒子 E，求 max |K e|
    e1_max = E_abs[0]
    e2_max = E_abs[1]
    Ke_max = abs(K[0, 0]) * e1_max + abs(K[0, 1]) * e2_max

    U_max = 1.0
    V_max = U_max - Ke_max

    print(f"max |K e| over E ≈ {Ke_max:.6f}")
    print(f"Original input constraint: |u| <= {U_max:.6f}")
    print(f"Tightened nominal input constraint: |v| <= {V_max:.6f}")
    print()

    # =========================
    # 5. 正弦位置参考 + Tube 跟踪仿真
    #    p_k = amp * sin(omega * k)，与离散双积分器相容的名义状态
    #    z_k = [p_k, p_{k+1}-p_k]，前馈名义输入 v_k 使 z 精确沿参考演化
    # =========================
    steps = 60
    amp = 1.5
    omega = 0.12
    # 需要 p_ref[k], p_ref[k+1], p_ref[k+2]，故长度 >= steps + 3
    p_ref = sin_position_reference(steps + 3, amp, omega)

    print(f"Sinusoidal reference: p_k = {amp} * sin({omega} * k)")
    e_trajs, x_nom, z_nom, v_nom = simulate_tube_tracking(
        A, B, K, p_ref, W_box, steps=steps, num_trials=200
    )
    v_nom_max = float(np.max(np.abs(v_nom)))
    print(f"max |nominal v_k| over horizon ≈ {v_nom_max:.6f} (compare with V_max = {V_max:.6f})")
    print()
    x_noisy, z_noisy, u_noisy = simulate_tube_tracking_once(
        A, B, K, p_ref, W_box, steps=steps, rng=np.random.default_rng(0)
    )

    # =========================
    # 6. 画图：误差相平面 + 时间域跟踪
    # =========================
    fig, ax = plt.subplots(figsize=(7, 7))

    plot_box(ax, W_box, color="green", label="W")

    colors = ["orange", "purple", "brown", "gray"]
    for i in range(min(4, len(term_boxes))):
        plot_box(
            ax,
            term_boxes[i],
            color=colors[i % len(colors)],
            label=f"A_K^{i} W",
            linestyle="--"
        )

    plot_box(ax, E_box, color="red", label="Approx E", linestyle="-")

    for traj in e_trajs:
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.15, color="C0")

    ax.set_xlabel("e1 (position error)")
    ax.set_ylabel("e2 (velocity increment error)")
    ax.set_title("Tube MPC: error set & tracking (sin reference)")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()

    k_axis = np.arange(steps + 1)
    fig2, (ax_p, ax_v, ax_u) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax_p.plot(k_axis, z_nom[:, 0], "k--", label="p_ref (nominal z1)")
    ax_p.plot(k_axis, x_noisy[:, 0], "C0-", label="x1 (closed-loop, w in W)")
    ax_p.set_ylabel("position")
    ax_p.legend(loc="upper right")
    ax_p.grid(True)
    ax_p.set_title("Sinusoidal reference tracking (discrete double integrator)")

    ax_v.plot(k_axis, z_nom[:, 1], "k--", label="v_disc ref (z2 = p_{k+1}-p_k)")
    ax_v.plot(k_axis, x_noisy[:, 1], "C1-", label="x2")
    ax_v.set_ylabel("disc. velocity")
    ax_v.legend(loc="upper right")
    ax_v.grid(True)

    t_u = np.arange(steps)
    ax_u.step(t_u, v_nom, "k--", where="post", label="nominal v_k (feedforward)")
    ax_u.step(t_u, u_noisy, "C2-", where="post", label="total u_k = v_k + K e_k")
    ax_u.set_xlabel("k")
    ax_u.set_ylabel("input")
    ax_u.legend(loc="upper right")
    ax_u.grid(True)

    plt.tight_layout()
    plt.show()