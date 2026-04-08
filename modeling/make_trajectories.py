import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def generate_base_trajectories(case:str, t_end=0.1, dt=1e-4):
    """
    Gera trajetórias sintéticas definindo posição e atitude,
    e derivando numericamente para obter velocidades e acelerações.
    """
    pad_stesps = 5
    time_vector = np.arange(
        0 - pad_stesps * dt, t_end + (pad_stesps * dt), dt, dtype=np.float64
    )
    num_steps = len(time_vector)

    # 1. Definir Posição (r_b) e Atitude (ângulos de Euler em radianos)
    f_trans = [1.0, 1.5, 1.0]
    amplitude_trans = [.5, .5, -.5]
    r_b_i = np.zeros((num_steps, 3))
    for i in range(3):
        r_b_i[:, i] = amplitude_trans[i] * np.sin(2.0 * np.pi * f_trans[i] * time_vector)

    f_rot = [1.0, 2, 2.0]
    theta = [np.deg2rad(2.0), np.deg2rad(1.0), -np.deg2rad(1.0)]
    print(theta)
    angles = np.zeros((num_steps, 3))
    for i in range(3):
        angles[:, i] = theta[i] * np.sin(2.0 * np.pi * f_rot[i] * time_vector)
    # angles[:, 1] = theta_y * np.sin(2.0 * np.pi * f_rot_y * time_vector)
    # angles[:, 2] = theta_z * np.sin(2.0 * np.pi * f_rot_z * time_vector)

    # 2. Derivação Numérica da Translação
    # Usamos np.gradient para manter o tamanho N do vetor (diferenças centrais)
    v_b_i = np.gradient(r_b_i, dt, axis=0)
    a_b_i = np.gradient(v_b_i, dt, axis=0)

    # 3. Calcular Matrizes de Rotação e Projetar g_b e calcular os angulos de rotação no corpo
    rots = Rotation.from_euler("xyz", angles)
    omega_b = np.zeros_like(angles)
    for i in range(1, num_steps):
        relative_rot = rots[i-1].inv() * rots[i]
        omega_b[i] = 2.0*relative_rot.as_rotvec()/dt
    # 4. calcula a aceleração angular, gravidade
    dot_omega_b = np.gradient(omega_b, dt, axis=0)
    g_i = np.array([0.0, 0.0, -9.81])
    # Rotaciona do Inercial para a Base
    g_b = rots.inv().apply(g_i)
    a_b = rots.inv().apply(a_b_i)

    # we filter out the initial transient
    valid_idx = (time_vector >= -1e-12) & (time_vector <= t_end + 1e-12)
    time_vector = time_vector[valid_idx]

    # 5. Organizar e salvar com Pandas
    df = pd.DataFrame(
        {
            "time": time_vector,
            "r_b_x": r_b_i[valid_idx, 0],
            "r_b_y": r_b_i[valid_idx, 1],
            "r_b_z": r_b_i[valid_idx, 2],
            "v_b_x": v_b_i[valid_idx, 0],
            "v_b_y": v_b_i[valid_idx, 1],
            "v_b_z": v_b_i[valid_idx, 2],
            "a_b_x_i": a_b_i[valid_idx, 0],
            "a_b_y_i": a_b_i[valid_idx, 1],
            "a_b_z_i": a_b_i[valid_idx, 2],
            "a_b_x": a_b[valid_idx, 0],
            "a_b_y": a_b[valid_idx, 1],
            "a_b_z": a_b[valid_idx, 2],
            "euler_phi": angles[valid_idx, 0],
            "euler_theta": angles[valid_idx, 1],
            "euler_psi": angles[valid_idx, 2],
            "omega_b_x": omega_b[valid_idx, 0],
            "omega_b_y": omega_b[valid_idx, 1],
            "omega_b_z": omega_b[valid_idx, 2],
            "dot_omega_b_x": dot_omega_b[valid_idx, 0],
            "dot_omega_b_y": dot_omega_b[valid_idx, 1],
            "dot_omega_b_z": dot_omega_b[valid_idx, 2],
            "g_b_x": g_b[valid_idx, 0],
            "g_b_y": g_b[valid_idx, 1],
            "g_b_z": g_b[valid_idx, 2],
        }
    )

    df.to_hdf("modeling/data/modeling.h5", key=f"{case}/trajectories", mode="a")
    print(f"Trajetória perfeita salva em modeling.h5 (key: {case}/trajectories)")

    return df


if __name__ == "__main__":
    generate_base_trajectories(case='teste',t_end=0.5, dt=1e-4)
