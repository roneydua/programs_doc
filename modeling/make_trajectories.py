import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def generate_base_trajectories(t_end=0.1, dt=1e-4):
    """
    Gera trajetórias sintéticas definindo posição e atitude,
    e derivando numericamente para obter velocidades e acelerações.
    """
    time_vector = np.arange(0, t_end, dt, dtype=np.float64)
    num_steps = len(time_vector)

    # 1. Definir Posição (r_b) e Atitude (ângulos de Euler em radianos)
    f_trans = 5.0
    amplitude_trans = .1
    r_b = np.zeros((num_steps, 3))
    r_b[:, 0] = amplitude_trans * np.sin(
        2.0 * np.pi * f_trans * time_vector
    )  # Oscilação em X

    f_rot = 2.0
    theta_amp = np.deg2rad(10.0)
    angles = np.zeros((num_steps, 3))
    angles[:, 1] = theta_amp * np.sin(
        2.0 * np.pi * f_rot * time_vector
    )  # Oscilação Pitch (Eixo Y)

    # 2. Derivação Numérica da Translação
    # Usamos np.gradient para manter o tamanho N do vetor (diferenças centrais)
    v_b_i = np.gradient(r_b, dt, axis=0)
    a_b_i = np.gradient(v_b_i, dt, axis=0)

    # 3. Calcular Matrizes de Rotação e Projetar g_b
    rots = Rotation.from_euler("xyz", angles)
    g_i = np.array([0.0, 0.0, -9.81])
    # Rotaciona do Inercial para a Base
    g_b = rots.inv().apply(g_i)
    a_b = rots.inv().apply(a_b_i)

    # 4. Derivação Numérica da Rotação
    euler_rates = np.gradient(angles, dt, axis=0)
    # Para rotações focadas em 1 eixo ou pequenos ângulos, omega_b ~= euler_rates
    omega_b = euler_rates
    dot_omega_b = np.gradient(omega_b, dt, axis=0)

    # 5. Organizar e salvar com Pandas
    df = pd.DataFrame(
        {
            "time": time_vector,
            "r_b_x": r_b[:, 0],
            "r_b_y": r_b[:, 1],
            "r_b_z": r_b[:, 2],
            "v_b_x": v_b_i[:, 0],
            "v_b_y": v_b_i[:, 1],
            "v_b_z": v_b_i[:, 2],
            "a_b_x": a_b[:, 0],
            "a_b_y": a_b[:, 1],
            "a_b_z": a_b[:, 2],
            "omega_b_x": omega_b[:, 0],
            "omega_b_y": omega_b[:, 1],
            "omega_b_z": omega_b[:, 2],
            "dot_omega_b_x": dot_omega_b[:, 0],
            "dot_omega_b_y": dot_omega_b[:, 1],
            "dot_omega_b_z": dot_omega_b[:, 2],
            "g_b_x": g_b[:, 0],
            "g_b_y": g_b[:, 1],
            "g_b_z": g_b[:, 2],
        }
    )

    df.to_csv("trajectories.csv", index=False)
    print("Trajetória perfeita salva em trajectories.csv")

    return df


if __name__ == "__main__":
    generate_base_trajectories(t_end=0.1, dt=1e-4)
