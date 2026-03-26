import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


TESE_IMAGE_FOLDER = "./../tese/images/not_used_on_thesis/"
plt.style.use("common_functions/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4

class inverse_problem_visualizer:
    """
    class to load multiple inverse problem estimations and plot them
    against the ground truth simulation in a structured grid format.
    """

    def __init__(
        self,
        sim_file="./modeling/data/simulation_output.csv",
        traj_file="./modeling/data/trajectories.csv",
    ):
        self.dpi = 144
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.axis_labels = ["x", "y", "z"]

        # dictionary to store multiple estimation datasets
        self.estimations = {}

        # load ground truth data
        self._load_ground_truth(sim_file, traj_file)

    def _load_ground_truth(self, sim_file, traj_file):
        """
        loads the exact states from the forward simulation and reference trajectories.
        """
        df_sim = pd.read_csv(sim_file)
        traj_data = pd.read_csv(traj_file)

        self.time = df_sim["time"].values
        # absolute kinematics
        a_b_true = traj_data[["a_b_x", "a_b_y", "a_b_z"]].values
        g_b_true = traj_data[["g_b_x", "g_b_y", "g_b_z"]].values
        self.f_b_true = a_b_true - g_b_true
        self.dot_omega_b_true = traj_data[
            ["dot_omega_b_x", "dot_omega_b_y", "dot_omega_b_z"]
        ].values

        # relative kinematics
        self.r_rel_true = df_sim[["r_rel_x", "r_rel_y", "r_rel_z"]].values
        q_true = df_sim[["q_x", "q_y", "q_z", "q_w"]].values
        self.euler_true = Rotation.from_quat(q_true).as_euler("xyz", degrees=False)

        # fiber lengths
        self.fiber_lengths_true = df_sim[
            [f"fiber_{i}_length" for i in range(1, 13)]
        ].values

    def add_estimation(self, label: str, filepath: str):
        """
        loads an estimation csv and stores it in the dictionary under the given label.
        """
        try:
            df_inv = pd.read_csv(filepath)

            est_data = {
                "f_b": df_inv[["f_b_est_x", "f_b_est_y", "f_b_est_z"]].values,
                "r_rel": df_inv[["r_rel_est_x", "r_rel_est_y", "r_rel_est_z"]].values,
                "fiber_lengths": df_inv[
                    [f"fiber_{i}_length_est" for i in range(1, 13)]
                ].values,
                
            }
            if "dot_omega_b_est_x" in df_inv.columns:
                est_data["dot_omega_b"] = (
                    df_inv[
                        ["dot_omega_b_est_x", "dot_omega_b_est_y", "dot_omega_b_est_z"]
                    ].values
                )
                est_data["euler"]= df_inv[["euler_est_x", "euler_est_y", "euler_est_z"]].values
            # check if temperature was estimated (7-DOF scenario)
            if "dT_est" in df_inv.columns:
                est_data["dT_est"] = df_inv["dT_est"].values
            if "dT_true" in df_inv.columns:
                est_data["dT_true"] = df_inv["dT_true"].values

            self.estimations[label] = est_data
            print(f"loaded estimation: {label} from {filepath}")

        except FileNotFoundError:
            print(f"warning: file {filepath} not found. skipping {label}.")

    def _plot_3x2_grid(
        self, true_data:np.ndarray, dict_key, multiplier:float, unit_state, unit_err, sup_ylabel, figure_save_name:str
    ):
        """
        internal helper to generate a 3x2 grid plot for 3d vectors.
        left column: states comparison. right column: errors (est - true).
        """
        fig, axs = plt.subplots(3, 2, dpi=self.dpi,figsize=(FIG_L,FIG_A*2), sharex=True)

        for i in range(3):
            # plot ground truth on the left
            base_signal = true_data[:, i] * multiplier
            axs[i, 0].plot(
                self.time,
                base_signal,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Referência",
            )

            # plot estimations and errors
            for idx, (label, est_data) in enumerate(self.estimations.items()):

                if dict_key in est_data.keys():
                    color = self.colors[idx % len(self.colors)]
                    signal_est = est_data[dict_key][:, i] * multiplier
                    error = np.abs(signal_est - base_signal)

                    # left: state
                    axs[i, 0].plot(self.time, signal_est, color=color, label=label)
                    # right: error
                    axs[i, 1].plot(self.time, error, color=color, label=label)

            # formatting
            axs[i, 0].set_ylabel(f"${self.axis_labels[i]}$ {unit_state}")
            # axs[i, 1].set_ylabel(f"Erro {self.axis_labels[i]} {unit_err}")
            axs[i, 0].grid(True)
            axs[i, 1].grid(True)
            axs[i,1].set_yscale("log")

        # legends and labels
        # axs[0, 0].legend(loc="upper right", fontsize=8)
        axs[-1, 0].set_xlabel(r"Tempo $[\si{\second}]$")
        axs[-1, 1].set_xlabel(r"Tempo $[\si{\second}]$")
        axs[0, 0].set_title("Estado estimado")
        axs[0, 1].set_title("Erro de estimação")
        handles, labels = axs[0,0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            fontsize="small",
            frameon=True,
        )
        fig.supylabel(sup_ylabel)
        plt.savefig(
            TESE_IMAGE_FOLDER + figure_save_name + ".pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close("all")

    def plot_specific_force(self):
        self._plot_3x2_grid(
            true_data=self.f_b_true,
            dict_key="f_b",
            multiplier=1.0,
            unit_state=r"$[\si{\meter\per\second\squared}]$",
            unit_err=r"$[\si{\meter\per\second\squared}]$",
            sup_ylabel=r"Força específica, ${}^{\mathcal{B}}\mathbf{f}$",
            figure_save_name="specific_force"
        )

    def plot_angular_acceleration(self):
        self._plot_3x2_grid(
            true_data=self.dot_omega_b_true,
            dict_key="dot_omega_b",
            multiplier= 180.0 / np.pi,
            unit_state=r"$[\si{\degree\per\second\squared}]$",
            unit_err=r"$[\si{\degree\per\second\squared}]$",
            sup_ylabel=r"Aceleração angular, ${}^{\mathcal{B}}\dot{\boldsymbol{\omega}}$",
            figure_save_name="angular_acceleration"
        )

    def plot_relative_position(self):
        self._plot_3x2_grid(
            true_data=self.r_rel_true,
            dict_key="r_rel",
            multiplier=1e6,
            unit_state=r"$[\si{\micro\meter}]$",
            unit_err=r"$[\si{\micro\meter}]$",
            sup_ylabel=r"Posição relativa, ${}^{\mathcal{B}}\mathbf{r}_\mathrm{rel}$",
            figure_save_name="relative_position"
        )

    def plot_relative_orientation(self):
        self._plot_3x2_grid(
            true_data=self.euler_true,
            dict_key="euler",
            multiplier= 1e6*180.0 / np.pi,
            unit_state=r"$[\si{\micro\degree}]$",
            unit_err=r"$[\si{\micro\degree}]$",
            sup_ylabel=r"Orientação relativa (Euler)",
            figure_save_name="relative_orientation"
        )

    def plot_fiber_errors(self, multipler:float, unit:str):
        """
        plots a 4x3 grid showing purely the estimation error for each fiber length.
        """
        fig, axs = plt.subplots(4, 3, sharex=True,sharey=True,figsize=(FIG_L,FIG_A*3), dpi=self.dpi)

        for i in range(4):
            for j in range(3):
                fiber_idx = 3 * i + j
                true_len = self.fiber_lengths_true[:, fiber_idx]

                for idx, (label, est_data) in enumerate(self.estimations.items()):
                    color = self.colors[idx % len(self.colors)]
                    est_len = est_data["fiber_lengths"][:, fiber_idx]
                    # error in micrometers
                    error_um = np.abs(est_len - true_len) * multipler
                    axs[i, j].plot(self.time, error_um, color=color, label=label)

                axs[i, j].set_ylabel(
                    r"$\ell_{" + str(fiber_idx + 1) + r"}$"
                )
                axs[i, j].grid(True)
                axs[i,j].set_yscale("log")
        handles, labels = axs[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            fontsize="small",
            frameon=True,
        )
        axs[0, 2].legend(loc="upper right", fontsize=8)
        fig.supxlabel(r"Tempo $[\si{\second}]$")
        fig.supylabel("Erro de estimação dos comprimentos das fibras "+unit)
        # fig.suptitle("Erro de Estimação dos Comprimentos Ópticos")
        plt.savefig(TESE_IMAGE_FOLDER + "comprimentos_fibras.pdf",bbox_inches="tight", format="pdf")
        plt.close("ALL")

    def plot_temperature(self):
        """
        plots the estimated temperature gradient if the 7-dof model was loaded.
        """
        has_temp = any("dT_est" in est for est in self.estimations.values())
        if not has_temp:
            return

        fig, ax = plt.subplots(1, 1, dpi=self.dpi)
        # assume ground truth temperature variation is 0 (or load it if you simulate a varying profile)
        # ax.plot(
        #     self.time,
        #     self.estimations["["dT_true"],
        #     color="black",
        #     linestyle="--",
        #     linewidth=1.5,
        #     label="Referência",
        # )
        _ax_error = ax.twinx()
        for idx, (label, est_data) in enumerate(self.estimations.items()):
            print(label)
            if "dT_est" in est_data:
                print(label)
                color = self.colors[idx % len(self.colors)]
                # _ax_error.plot(
                #     self.time,
                #     np.abs(est_data["dT_est"] - est_data["dT_true"]),
                #     color=self.colors[0],
                #     label="Erro",
                # )
                _ax_error.fill_between(
                    self.time,
                    np.zeros(len(self.time)),
                    np.abs(est_data["dT_est"] - est_data["dT_true"]),
                    color=self.colors[0],
                    alpha=0.2,
                )
                ax.plot(self.time, est_data["dT_true"], color='black',lw=2,label="Referência")
                ax.plot(self.time, est_data["dT_est"], color=self.colors[1], label="Estimado")
        _ax_error.set_ylabel(r"Erro de $\Delta T\; [\si{\degreeCelsius}]$")
        ax.set_ylabel(r"$\Delta T\; [\si{\degreeCelsius}]$")
        ax.set_xlabel(r"Tempo $[\si{\second}]$")
        ax.set_title("Estimação da variação de temperatura")
        ax.grid(True)
        ax.legend()
        plt.savefig(TESE_IMAGE_FOLDER + "temperature_estimation.pdf", format="pdf")
        plt.close("all")

    def show_all(self):
        self.plot_specific_force()
        self.plot_angular_acceleration()
        self.plot_relative_position()
        self.plot_relative_orientation()
        self.plot_fiber_errors(multipler=1e9, unit="$[\\si{\\nano\\meter}]$")
        self.plot_temperature()
        plt.show()

def plot_graphics():
    visualizer = inverse_problem_visualizer()

    # load multiple datasets to compare
    # visualizer.add_estimation(
    #     label="3-DOF",
    #     filepath="./modeling/data/inverse_output_closed_form_translacional.csv",
    # )
    visualizer.add_estimation(
        label="6-DOF",
        filepath="./modeling/data/inverse_output_closed_form_translacional_angular.csv",
    )
    visualizer.add_estimation(
        label="7-DOF",
        filepath="./modeling/data/inverse_output_closed_form_translacional_angular_thermal.csv",
    )

    visualizer.add_estimation(
        label="Push-pull cruzado",
        filepath="./modeling/data/inverse_output_optical_push_pull_cruzed.csv",
    )
    # visualizer.add_estimation(
    #     label="Push-pull alinhado",
    #     filepath="./modeling/data/inverse_output_optical_push_pull_aligned.csv",
    # )
    # visualizer.add_estimation(label="Levenberg-Marquardt", filepath="inverse_output_lm.csv")

    # generate and show all plots
    visualizer.show_all()


if __name__ == "__main__":
    plot_graphics()
