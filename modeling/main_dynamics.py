import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from modeling.accel_model import accel_model_euler_poincare
from modeling.make_trajectories import generate_base_trajectories

# def interpolate_trajectory(t: float, time_vector: np.ndarray, data: np.ndarray):
#     """
#     linear interpolation of trajectory data for the ode solver.
#     """
#     idx = np.searchsorted(time_vector, t) - 1
#     idx = np.clip(idx, 0, len(time_vector) - 2)
#     dt = time_vector[idx + 1] - time_vector[idx]
#     w = (t - time_vector[idx]) / dt
#     return (1.0 - w) * data[idx] + w * data[idx + 1]


def find_initial_equilibrium(
    model: accel_model_euler_poincare,
    a_b_0: np.ndarray,
    omega_b_0: np.ndarray,
    dot_omega_b_0: np.ndarray,
    g_b_0: np.ndarray,
):
    """
    Encontra a posição e atitude relativas iniciais exatas que equilibram as forças
    e momentos no instante t=0, eliminando o transiente inicial.
    """

    def residual(x: np.ndarray):
        r_rel = x[0:3]
        rot_vec = x[3:6]
        r_m_b = Rotation.from_rotvec(rot_vec).as_matrix()

        v_rel = np.zeros(3)
        omega_rel = np.zeros(3)

        # Queremos que as acelerações relativas sejam zero no instante inicial
        a_rel, dot_omega_rel = model.forward_dynamics(
            r_rel, v_rel, r_m_b, omega_rel, a_b_0, omega_b_0, dot_omega_b_0, g_b_0
        )
        return np.concatenate((a_rel, dot_omega_rel))*1e6

    print("Calculando condição inicial de equilíbrio...")
    res = least_squares(residual, np.zeros(6), method="lm")

    r_rel_0 = res.x[0:3]
    # O scipy retorna quatérnios no formato [x, y, z, w]
    q_rel_0 = Rotation.from_rotvec(res.x[3:6]).as_quat()

    return r_rel_0, q_rel_0


def dynamics_ode(
    t:float,
    state: np.ndarray,
    model: accel_model_euler_poincare,
    interp_a: interp1d,
    interp_omega: interp1d,
    interp_dot_omega: interp1d,
    interp_g: interp1d,
):
    """
    ode function for the relative dynamics of the seismic mass.
    state = [r_rel (3), v_rel (3), q_rel (4), omega_rel (3)]
    """
    r_rel_b = state[0:3]
    v_rel_b = state[3:6]
    q_rel_m_b = state[6:10]  # scalar last: [x, y, z, w]
    omega_rel_m = state[10:13]

    # normalize quaternion to prevent numerical drift
    q_rel_m_b = q_rel_m_b / np.linalg.norm(q_rel_m_b)
    r_m_b = Rotation.from_quat(q_rel_m_b).as_matrix()

    a_b_b = interp_a(t)
    omega_b_b = interp_omega(t)
    dot_omega_b_b = interp_dot_omega(t)
    g_b = interp_g(t)

    # forward dynamics from euler-poincare model
    a_rel_b, dot_omega_rel_m = model.forward_dynamics(
        r_rel_b, v_rel_b, r_m_b, omega_rel_m, a_b_b, omega_b_b, dot_omega_b_b, g_b
    )
    # --- INJEÇÃO DE AMORTECIMENTO ARTIFICIAL ---
    # Frequências naturais aproximadas (4 fibras por eixo)
    omega_n_trans = np.sqrt(4.0 * model.k / model.seismic_mass)
    omega_n_rot = np.sqrt(4.0 * model.k * (model.seismic_edge / 2.0) ** 2 / model.i_m)
    # Amortecimento estrutural real da fibra óptica de sílica
    damping_ratio = 0.00

    # Adicionamos a força dissipativa (c * v) diretamente nas acelerações
    a_rel_b -= 2.0 * damping_ratio * omega_n_trans * v_rel_b
    dot_omega_rel_m -= 2.0 * damping_ratio * omega_n_rot * omega_rel_m
    # -------------------------------------------
    # quaternion kinematics
    w_x, w_y, w_z = omega_rel_m
    omega_matrix = np.array(
        [
            [0, w_z, -w_y, w_x],
            [-w_z, 0, w_x, w_y],
            [w_y, -w_x, 0, w_z],
            [-w_x, -w_y, -w_z, 0],
        ],np.float64
    )
    q_dot = 0.5 * omega_matrix @ q_rel_m_b

    return np.concatenate((v_rel_b, a_rel_b, q_dot, dot_omega_rel_m))


def simulate_dynamics(t_end: float, dt: float):
    """
    Gera a trajetória perfeita sob demanda, integra a dinâmica,
    extrai os comprimentos e salva com pandas.
    """
    # 1. Definir parâmetros de tempo e gerar a trajetória
    print(f"generate trajectories with final time{t_end}s spaced by dt {dt}s")
    df_traj = generate_base_trajectories(t_end=t_end, dt=dt)

    # 2. Extrair dados para numpy arrays (necessário para o interpolador)
    time_vector = df_traj["time"].values
    a_b_data = df_traj[["a_b_x", "a_b_y", "a_b_z"]].values
    omega_b_data = df_traj[["omega_b_x", "omega_b_y", "omega_b_z"]].values
    dot_omega_b_data = df_traj[
        ["dot_omega_b_x", "dot_omega_b_y", "dot_omega_b_z"]
    ].values
    g_b_data = df_traj[["g_b_x", "g_b_y", "g_b_z"]].values

    # --- creation of cubic interpolators ---
    interp_a = interp1d(
        time_vector, a_b_data, axis=0, kind="cubic", fill_value="extrapolate"
    )
    interp_omega = interp1d(
        time_vector, omega_b_data, axis=0, kind="cubic", fill_value="extrapolate"
    )
    dot_omega_alpha = interp1d(
        time_vector, dot_omega_b_data, axis=0, kind="cubic", fill_value="extrapolate"
    )
    interp_g = interp1d(
        time_vector, g_b_data, axis=0, kind="cubic", fill_value="extrapolate"
    )
    t_span = (time_vector[0], time_vector[-1])
    model = accel_model_euler_poincare()
    # --- INICIALIZAÇÃO NO EQUILÍBRIO ---
    r_rel_0, q_rel_0 = find_initial_equilibrium(
        model, a_b_data[0], omega_b_data[0], dot_omega_b_data[0], g_b_data[0]
    )
    # Condição inicial [r_rel (3), v_rel(3), q (4), omega_rel (3)]
    state_0 = np.zeros(13)
    state_0[0:3] = r_rel_0  # Posição de equilíbrio
    state_0[6:10] = q_rel_0  # Atitude de equilíbrio

    print("Iniciando integração numérica...")
    sol: OdeResult = solve_ivp(
        dynamics_ode,
        t_span,
        state_0,
        method="BDF",
        t_eval=time_vector,
        max_step=1e-3,
        args=(model, interp_a, interp_omega, dot_omega_alpha, interp_g),
        # rtol=1e-9,
        # atol=1e-6,
    )

    num_steps = len(sol.t)
    fiber_lengths = np.zeros((num_steps, 12))

    for i in range(num_steps):
        r_rel = sol.y[0:3, i]
        q_rel = sol.y[6:10, i]
        q_rel /= np.linalg.norm(q_rel)
        r_m_b = Rotation.from_quat(q_rel).as_matrix()

        for j in range(12):
            fiber_vec_b = r_rel + r_m_b @ model.m_m[j] - model.b_b[j]
            fiber_lengths[i, j] = np.linalg.norm(fiber_vec_b)

    # 3. Exportar resultados da integração
    data_dict = {"time": sol.t}
    state_labels = [
        "r_rel_x",
        "r_rel_y",
        "r_rel_z",
        "v_rel_x",
        "v_rel_y",
        "v_rel_z",
        "q_x",
        "q_y",
        "q_z",
        "q_w",
        "omega_rel_x",
        "omega_rel_y",
        "omega_rel_z",
    ]
    for idx, label in enumerate(state_labels):
        data_dict[label] = sol.y[idx, :]

    for j in range(12):
        data_dict[f"fiber_{j+1}_length"] = fiber_lengths[:, j]

    df_sim = pd.DataFrame(data_dict)
    save_data_pah = "./modeling/data/simulation_output.csv"
    df_sim.to_csv(save_data_pah, index=False)
    print(f"Dados da simulação salvos em {save_data_pah}")


if __name__ == "__main__":
    simulate_dynamics(t_end=0.3, dt=1e-4)
