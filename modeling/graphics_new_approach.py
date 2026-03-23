import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

plt.style.use("common_functions/roney3.mplstyle")
mycolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

dpi = 144


def plot_estimation_results():
    """
    loads simulation and estimation data, computes theoretical specific force,
    and plots comparisons with the absolute error on a secondary right y-axis.
    """
    df_sim = pd.read_csv("simulation_output.csv")
    df_inv = pd.read_csv("inverse_output.csv")
    traj_data = pd.read_csv("trajectories.csv")

    time = df_sim["time"].values

    # extract true kinematics
    a_b_true = traj_data[["a_b_x", "a_b_y", "a_b_z"]].values
    g_b_true = traj_data[["g_b_x", "g_b_y", "g_b_z"]].values
    dot_omega_b_true = traj_data[
        ["dot_omega_b_x", "dot_omega_b_y", "dot_omega_b_z"]
    ].values
    f_b_true = a_b_true - g_b_true

    r_rel_true = df_sim[["r_rel_x", "r_rel_y", "r_rel_z"]].values

    q_true = df_sim[["q_x", "q_y", "q_z", "q_w"]].values
    euler_true = Rotation.from_quat(q_true).as_euler("xyz", degrees=False)
    # meu_dict[f"x{i}"] for i in range(1, 13) if f"x{i}" in meu_dict
    fiber_lengths_true = df_sim[[f"fiber_{i}_length" for i in range(1, 13)]].values

    # extract estimated states
    f_b_est = df_inv[["f_b_est_x", "f_b_est_y", "f_b_est_z"]].values
    dot_omega_b_est = df_inv[
        ["dot_omega_b_est_x", "dot_omega_b_est_y", "dot_omega_b_est_z"]
    ].values
    r_rel_est = df_inv[["r_rel_est_x", "r_rel_est_y", "r_rel_est_z"]].values
    euler_est = df_inv[["euler_est_x", "euler_est_y", "euler_est_z"]].values
    fiber_lengths_est = df_inv[[f"fiber_{i}_length_est" for i in range(1, 13)]].values
    axis_labels = ["x", "y", "z"]

    def plot_with_error(
        fig,
        axs,
        true_data,
        est_data,
        ylabel_err,
        multiplier=1.0,
        full_scale=1.0,
        sup_ylabel=None,
    ):
        """
        helper function to plot true vs estimated signals with absolute error on the right axis.
        """
        for i in range(3):
            true_signal = true_data[:, i] * multiplier
            est_signal = est_data[:, i] * multiplier
            abs_error = np.abs(est_signal - true_signal)
            # main axis (left)
            axs[i].plot(time, true_signal, mycolors[0], label="true")
            axs[i].plot(time, est_signal, mycolors[1], label="estimated")
            axs[i].set_ylabel(f"${axis_labels[i]}$")
            axs[i].grid(True)
            # error axis (right)
            ax_err = axs[i].twinx()
            ax_err.fill_between(
                time, 0, abs_error, color=mycolors[2], alpha=0.2, label="abs error"
            )
            ax_err.plot(time, abs_error, mycolors[2], alpha=0.5)
            ax_err.set_ylabel(f"{ylabel_err}", color="red")
            ax_err.tick_params(axis="y", colors="red")
            # set error axis lower limit to 0 for better visualization
            ax_err.set_ylim(bottom=0)
            # combine legends
            lines_1, labels_1 = axs[i].get_legend_handles_labels()
            lines_2, labels_2 = ax_err.get_legend_handles_labels()
            axs[i].legend(
                lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8
            )
        fig.supylabel(sup_ylabel)
        axs[-1].set_xlabel("time [\\si{\\second}]")

    # 1. specific force plot
    fig1, axs1 = plt.subplots(3, 1, dpi=dpi, sharex=True)
    plot_with_error(
        fig1,
        axs1,
        f_b_true,
        f_b_est,
        r"$\left[\si{\meter\per\second\squared}\right]$",
        sup_ylabel=r"Força específica, ${}^{\mathcal{B}}\mathbf{f} \left[\si{\meter\per\second\squared}\right]$",
    )

    # 2. angular acceleration plot
    fig2, axs2 = plt.subplots(3, 1, dpi=dpi, sharex=True)
    plot_with_error(
        fig2,
        axs2,
        dot_omega_b_true,
        dot_omega_b_est,
        r"[\si{\degree\per\second\squared}]",
        multiplier=180.0 / np.pi,
        sup_ylabel=r"Aceleração angular, $ {}^{\mathcal{B}}\dot{\boldsymbol{\omega}} \left[\si{\degree\per\second\squared}\right]$",
    )

    # 3. relative position plot
    fig3, axs3 = plt.subplots(3, 1, dpi=dpi, sharex=True)
    plot_with_error(
        fig3,
        axs3,
        r_rel_true,
        r_rel_est,
        ylabel_err=r"[$\si{\micro\meter}$]",
        multiplier=1e6,
        sup_ylabel=r"Posição relativa, $ {}^{\mathcal{B}}\mathbf{r}_\mathrm{rel} \left[\si{\micro\meter}\right]$",
    )

    # 4. relative orientation plot
    fig4, axs4 = plt.subplots(3, 1, dpi=dpi, sharex=True)
    plot_with_error(
        fig4,
        axs4,
        euler_true,
        euler_est,
        r"$\left[\si{\micro\degree}\right]$",
        multiplier=1e6 * 180 / np.pi,
        sup_ylabel=r"Orientação relativa (ângulos de Euler) $\left[\si{\micro\degree}\right]$",
    )

    fig5, axs5 = plt.subplots(4, 3, dpi=dpi, sharex=True)
    for i in range(4):
        for j in range(3):
            axs5[i, j].plot(
                time,
                (3e-3 - fiber_lengths_true[:, 3 * i + j]) / 3e-3,
                mycolors[0],linewidth=1.5,
                label="Verdadeiro",
            )
            axs5[i, j].plot(
                time,
                (3e-3 - fiber_lengths_est[:, 3 * i + j]) / 3e-3,
                mycolors[1],
                label="Estimado",
            )
            axs5[i, j].set_ylabel(r"$\mathrm{FS}_{" + str(3*i+j) + "}$")

    fig5.supylabel(r"$\mu\varepsilon'$")
    # axs5[].legend()
    fig5.supxlabel("time [\\si{\\second}]")
    plt.show()


if __name__ == "__main__":
    plot_estimation_results()
