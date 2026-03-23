import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def plot_estimation_results():
    """
    loads simulation and estimation data, computes theoretical specific force,
    and plots comparisons between true and estimated states.
    """
    # load datasets
    df_sim = pd.read_csv("simulation_output.csv")
    df_inv = pd.read_csv("inverse_output.csv")
    traj_data = pd.read_csv("trajectories.csv")

    time = df_sim["time"].values

    # extract theoretical base kinematics
    a_b_true = traj_data[["a_b_x","a_b_y","a_b_z"]].values
    g_b_true = traj_data[["g_b_x", "g_b_y", "g_b_z"]].values
    dot_omega_b_true = traj_data[["dot_omega_b_x", "dot_omega_b_y", "dot_omega_b_z"]].values

    # theoretical specific force: f_b = a_b - g_b
    f_b_true = a_b_true - g_b_true

    # extract simulated relative position
    r_rel_true = df_sim[["r_rel_x", "r_rel_y", "r_rel_z"]].values
    # extract truequaternions and convert to Euler angles
    q_true = df_sim[["q_x", "q_y", "q_z", "q_w"]].values
    euler_true = Rotation.from_quat(q_true).as_euler("xyz", degrees=False)
    # extract estimated states
    f_b_est = df_inv[["f_b_est_x", "f_b_est_y", "f_b_est_z"]].values
    dot_omega_b_est = df_inv[["dot_omega_b_est_x", "dot_omega_b_est_y", "dot_omega_b_est_z"]].values
    r_rel_est = df_inv[["r_rel_est_x", "r_rel_est_y", "r_rel_est_z"]].values
    euler_est = df_inv[["euler_est_x", "euler_est_y", "euler_est_z"]].values
    axis_labels = ["x", "y", "z"]

    # 1. specific force plot
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle("Specific Force Comparison", fontsize=14)
    for i in range(3):
        axs1[i].plot(time, f_b_true[:, i], "k-", linewidth=2, label="true")
        axs1[i].plot(time, f_b_est[:, i], "r--", linewidth=1.5, label="estimated")
        axs1[i].set_ylabel(f"f_{axis_labels[i]} [m/s^2]")
        axs1[i].grid(True)
        axs1[i].legend(loc="upper right")
    axs1[-1].set_xlabel("time [s]")
    plt.tight_layout()

    # 2. angular acceleration plot
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("Angular Acceleration Comparison", fontsize=14)
    for i in range(3):
        axs2[i].plot(time, dot_omega_b_true[:, i], "k-", linewidth=2, label="true")
        axs2[i].plot(time, dot_omega_b_est[:, i], "b--", linewidth=1.5, label="estimated")
        axs2[i].set_ylabel(f"dot_omega_{axis_labels[i]} [rad/s^2]")
        axs2[i].grid(True)
        axs2[i].legend(loc="upper right")
    axs2[-1].set_xlabel("time [s]")
    plt.tight_layout()

    # 3. relative position plot
    fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig3.suptitle("Relative Position Comparison", fontsize=14)
    for i in range(3):
        axs3[i].plot(time, r_rel_true[:, i] * 1e6, "k-", linewidth=2, label="true")
        axs3[i].plot(
            time, r_rel_est[:, i] * 1e6, "g--", linewidth=1.5, label="estimated"
        )
        axs3[i].set_ylabel(f"r_{axis_labels[i]} [um]")
        axs3[i].grid(True)
        axs3[i].legend(loc="upper right")
    axs3[-1].set_xlabel("time [s]")
    plt.tight_layout()


    # relative orientation plot ---
    fig4, axs4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig4.suptitle("Relative Orientation Comparison", fontsize=14)
    for i in range(3):
        # Multiplicando por 1e6 para plotar em microrradianos (urad)
        axs4[i].plot(time, euler_true[:, i] * 1e6, "k-", linewidth=2, label="true")
        axs4[i].plot(time, euler_est[:, i] * 1e6, "m--", linewidth=1.5, label="estimated")
        axs4[i].set_ylabel(f"\u03b8_{axis_labels[i]} [\u03bcrad]")
        axs4[i].grid(True)
        axs4[i].legend(loc="upper right")
    axs4[-1].set_xlabel("time [s]")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_estimation_results()
