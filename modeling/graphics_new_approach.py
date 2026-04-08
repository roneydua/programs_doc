import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


TESE_IMAGE_FOLDER = "./../tese/images/used_on_thesis/"
plt.style.use("common_functions/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class inverse_problem_visualizer:
    """
    class to load multiple inverse problem estimations and plot them
    against the ground truth simulation in a structured grid format.
    """

    def __init__(
        self,
        case: str,
        data_main_file="./modeling/data/modeling.h5",
    ):
        self.dpi = 144
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.axis_labels = [r"\mathbf{x}", r"\mathbf{y}", r"\mathbf{z}"]
        self.case = case
        # dictionary to store multiple estimation datasets
        self.estimations = {}
        self.data_main_file = data_main_file
        # load ground truth data
        self._load_ground_truth()

    def _load_ground_truth(self):
        """
        loads the exact states from the forward simulation and reference trajectories.
        """
        df_sim = pd.read_hdf(self.data_main_file, key=f"{self.case}/simulation")
        traj_data = pd.read_hdf(self.data_main_file, key=f"{self.case}/trajectories")

        self.time = df_sim["time"].values
        # absolute kinematics
        self.a_b_true = traj_data[["a_b_x", "a_b_y", "a_b_z"]].values
        g_b_true = traj_data[["g_b_x", "g_b_y", "g_b_z"]].values
        self.r_b_true = traj_data[["r_b_x", "r_b_y", "r_b_z"]].values
        self.v_b_true = traj_data[["v_b_x", "v_b_y", "v_b_z"]].values
        self.a_b_i = traj_data[["a_b_x_i", "a_b_y_i", "a_b_z_i"]].values
        self.f_b_true = self.a_b_true - g_b_true
        self.dot_omega_b_true = traj_data[
            ["dot_omega_b_x", "dot_omega_b_y", "dot_omega_b_z"]
        ].values
        self.omega_b_true = traj_data[["omega_b_x", "omega_b_y", "omega_b_z"]].values
        self.euler_traj_true = traj_data[
            ["euler_phi", "euler_theta", "euler_psi"]
        ].values

        # relative kinematics
        self.r_rel_true = df_sim[["r_rel_x", "r_rel_y", "r_rel_z"]].values
        q_true = df_sim[["q_x", "q_y", "q_z", "q_w"]].values
        self.euler_true = Rotation.from_quat(q_true).as_euler("xyz", degrees=False)

        # fiber lengths
        self.fiber_lengths_true = df_sim[
            [f"fiber_{i}_length" for i in range(1, 13)]
        ].values

    def add_estimation(self, label: str, key: str):
        """
        loads an estimation h5 and stores it in the dictionary under the given label.
        """
        try:
            df_inv = pd.read_hdf(
                path_or_buf=self.data_main_file, key=f"{self.case}/{key}"
            )

            est_data = {
                "f_b": df_inv[["f_b_est_x", "f_b_est_y", "f_b_est_z"]].values,
                "r_rel": df_inv[["r_rel_est_x", "r_rel_est_y", "r_rel_est_z"]].values,
                "fiber_lengths": df_inv[
                    [f"fiber_{i}_length_est" for i in range(1, 13)]
                ].values,
            }
            if "dot_omega_b_est_x" in df_inv.columns:
                est_data["dot_omega_b"] = df_inv[
                    ["dot_omega_b_est_x", "dot_omega_b_est_y", "dot_omega_b_est_z"]
                ].values
                est_data["euler"] = df_inv[
                    ["euler_est_x", "euler_est_y", "euler_est_z"]
                ].values
            # check if temperature was estimated (7-DOF scenario)
            if "dT_est" in df_inv.columns:
                est_data["dT_est"] = df_inv["dT_est"].values
            if "dT_true" in df_inv.columns:
                est_data["dT_true"] = df_inv["dT_true"].values

            self.estimations[label] = est_data
            print(f"loaded estimation: {label} from {label}")

        except FileNotFoundError:
            print(f"warning: file {label} not found. skipping {label}.")

    def _plot_trajectory_translational(self):
        fig, ax = plt.subplots(
            nrows=3,
            ncols=1,
            num=1,
            sharex=True,
            figsize=(FIG_L, 1.5 * FIG_A),
            squeeze=False,
        )
        for i in range(3):
            ax[0, 0].plot(
                self.time, self.r_b_true[:, i], label=f"${self.axis_labels[i]}$"
            )
            ax[1, 0].plot(self.time, self.v_b_true[:, i])
            ax[2, 0].plot(self.time, self.a_b_i[:, i])
            # ax[3,0].plot(self.time, self.a_b_true[:, i])

        # ax[0,0].set_ylabel(f"${self.axis_labels[i]}$")
        ax[0, 0].set_ylabel(
            r"${}^\mathcal{I}\boldsymbol{r}_{\mathrm{b}} [\si{\meter}]$"
        )
        ax[1, 0].set_ylabel(
            r"${}^\mathcal{I}\dot{\boldsymbol{r}}_{\mathrm{b}} [\si{\meter\per\second}]$"
        )
        ax[2, 0].set_ylabel(
            r"${}^\mathcal{I}\ddot{\boldsymbol{r}}_{\mathrm{b}} [\si{\meter\per\second\squared}]$"
        )
        # ax[3, 0].set_ylabel(
        # r"${}^\mathcal{B}\ddot{\boldsymbol{r}}_{\mathrm{b}} [\si{\meter\per\second\squared}]$"
        # )
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            # mode="expand",
            # frameon=True,
        )
        # ax[0, 0].legend(ncols=3)
        fig.supxlabel(r"Tempo [\si{\second}]")
        plt.savefig(
            TESE_IMAGE_FOLDER + "estados_referencia_translacional.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig=1)

    def _plot_trajectory_rotational(self):
        fig, ax = plt.subplots(
            nrows=3,
            ncols=1,
            num=1,
            sharex=True,
            figsize=(FIG_L, 1.5 * FIG_A),
            squeeze=False,
        )
        _euler_label = [r"$\phi$", r"$\theta$", r"$\psi$"]
        for i in range(3):
            ax[0, 0].plot(
                self.time, np.rad2deg(self.euler_traj_true[:, i]), label=_euler_label[i]
            )
            ax[1, 0].plot(
                self.time,
                np.rad2deg(self.omega_b_true[:, i]),
                label=f"${self.axis_labels[i]}$",
            )
            ax[2, 0].plot(
                self.time,
                np.rad2deg(self.dot_omega_b_true[:, i]),
                label=f"${self.axis_labels[i]}$",
            )

        # ax[0,0].set_ylabel(f"${self.axis_labels[i]}$")
        ax[0, 0].set_ylabel(r"Ângulos de Euler [\si{\degree}]")
        ax[1, 0].set_ylabel(
            r"${}^\mathcal{B}{\boldsymbol{\omega}} [\si{\degree\per\second}]$"
        )
        ax[2, 0].set_ylabel(
            r"${}^\mathcal{B}{\dot{\boldsymbol{\omega}}} [\si{\degree\per\second}]$"
        )
        # handles, labels_legend = ax[1, 0].get_legend_handles_labels()
        # fig.legend(
        #     handles,
        #     labels_legend,
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 1.02),
        #     ncol=5,
        #     # mode="expand",
        #     fontsize="small",
        #     # frameon=True,
        # )
        ax[0, 0].legend(ncols=3)
        ax[1, 0].legend(ncols=3)
        ax[2, 0].legend(ncols=3)

        fig.supxlabel(r"Tempo [\si{\second}]")
        plt.savefig(
            TESE_IMAGE_FOLDER + "estados_referencia_rotacional.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig=1)

    def _plot_3x2_grid(
        self,
        true_data: np.ndarray,
        dict_key: str,
        multiplier: float,
        unit_state: str,
        sup_ylabel: str,
        figure_save_name: str,
        plot_norm: bool = False,
        norm_name: str = "",
        plot_error: bool = True,
    ):
        """
        internal helper to generate a grid plot for 3d vectors.
        left column: states comparison. right column: errors (est - true).
        """
        rows = 4 if plot_norm else 3
        ncols = 2 if plot_error else 1

        fig, axs = plt.subplots(
            rows,
            ncols,
            dpi=self.dpi,
            figsize=(FIG_L, FIG_A * (rows * 0.5)),
            sharex=True,
            squeeze=False,
        )

        if plot_norm:
            label_norm = rf"\|{norm_name}\|" if norm_name else r"\|\cdot\|"
            labels = self.axis_labels + [label_norm]
        else:
            labels = self.axis_labels

        for i in range(rows):
            # plot ground truth on the left
            if i < 3:
                base_signal = true_data[:, i] * multiplier
            else:
                base_signal = np.linalg.norm(true_data, axis=1) * multiplier

            axs[i, 0].plot(
                self.time,
                base_signal,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Referência",
            )
            axs[i, 0].set_ylabel(f"${labels[i]}$ {unit_state}")
            axs[i, 0].grid(True)
            if plot_error:
                # plot estimations and errors
                for idx, (label, est_data) in enumerate(self.estimations.items()):

                    if dict_key in est_data.keys():
                        color = self.colors[idx % len(self.colors)]
                        if i < 3:
                            signal_est = est_data[dict_key][:, i] * multiplier
                        else:
                            signal_est = (
                                np.linalg.norm(est_data[dict_key], axis=1) * multiplier
                            )
                        error = np.abs(signal_est - base_signal)
                        # left: state
                        axs[i, 0].plot(self.time, signal_est, color=color, label=label)
                        # right: error
                        axs[i, 1].plot(self.time, error, color=color, label=label)
                        if max(np.abs(signal_est)) > 10 * max(np.abs(base_signal)):
                            if i < 3:
                                axs[i, 0].set_ylim(
                                    [
                                        -1.2 * max(np.abs(base_signal)),
                                        1.2 * max(np.abs(base_signal)),
                                    ]
                                )
                            else:
                                axs[i, 0].set_ylim([0, 1.2 * max(np.abs(base_signal))])

                axs[i, 1].grid(True)
                axs[i, 1].set_yscale("log")
                axs[0, 1].set_title("Erro de estimação")

        # legends and labels
        # axs[0, 0].legend(loc="upper right", fontsize=8)
        # axs[-1, 0].set_xlabel(r"Tempo $[\si{\second}]$")
        # axs[-1, 1].set_xlabel(r"Tempo $[\si{\second}]$")
        axs[0, 0].set_title("Estado estimado")
        handles, labels_legend = axs[0, 0].get_legend_handles_labels()
        fig.supxlabel(r"Tempo $[\si{\second}]$")
        fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=True,
        )
        fig.supylabel(sup_ylabel)
        plt.savefig(
            f"{TESE_IMAGE_FOLDER}{figure_save_name}_{self.case}.pdf",
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
            # unit_err=r"$[\si{\meter\per\second\squared}]$",
            sup_ylabel=r"Força específica, ${}^{\mathcal{B}}\mathbf{f}$",
            figure_save_name="specific_force",
            plot_norm=True,
            norm_name=r"{}^{\mathcal{B}}\mathbf{f}",
        )

    def plot_angular_acceleration(self):
        self._plot_3x2_grid(
            true_data=self.dot_omega_b_true,
            dict_key="dot_omega_b",
            multiplier=180.0 / np.pi,
            unit_state=r"$[\si{\degree\per\second\squared}]$",
            # unit_err=r"$[\si{\degree\per\second\squared}]$",
            sup_ylabel=r"Aceleração angular, ${}^{\mathcal{B}}\dot{\boldsymbol{\omega}}$",
            figure_save_name="angular_acceleration",
            plot_norm=True,
            norm_name=r"{}^{\mathcal{B}}\dot{\boldsymbol{\omega}}",
        )

    def plot_relative_position(self):
        self._plot_3x2_grid(
            true_data=self.r_rel_true,
            dict_key="r_rel",
            multiplier=1e6,
            unit_state=r"$[\si{\micro\meter}]$",
            # unit_err=r"$[\si{\micro\meter}]$",
            sup_ylabel=r"Posição relativa, ${}^{\mathcal{B}}\mathbf{r}_\mathrm{rel}$",
            figure_save_name="relative_position",
        )

    def plot_relative_orientation(self):
        self._plot_3x2_grid(
            true_data=self.euler_true,
            dict_key="euler",
            multiplier=1e6 * 180.0 / np.pi,
            unit_state=r"$[\si{\micro\degree}]$",
            sup_ylabel=r"Orientação relativa (Euler)",
            figure_save_name="relative_orientation",
        )
        # self._plot_trajectory_rotational()
        # self._plot_trajectory_translational()
        # def plot_position_on_inertial(self):
        #     self._plot_3x2_grid(
        #         true_data=self.r_b_true,
        #         dict_key=None,
        #         multiplier=1,
        #         unit_state=r"$[\si{\meter}]$",
        #         plot_error=False,
        #         sup_ylabel=r"Posição no corpo, ${}^{\mathcal{I}}\mathbf{r}$",
        #         figure_save_name="position_body_on_inertial",
        #     )

        # def plot_velocity_on_inertial(self):
        #     self._plot_3x2_grid(
        #         true_data=self.v_b_true,
        #         dict_key=None,
        #         multiplier=1,
        #         unit_state=r"$[\si{\meter\per\second}]$",
        #         plot_error=False,
        #         sup_ylabel=r"Velocidade do corpo, ${}^{\mathcal{I}}\dot{\mathbf{r}}$",
        #         figure_save_name="velocity_body_on_inertial",
        #     )

        # def plot_position_on_inertial(self):
        #     self._plot_3x2_grid(
        #         true_data=self.r_b_true,
        #         dict_key=None,
        #         multiplier=1,
        #         unit_state=r"$[\si{\meter}]$",
        #         plot_error=False,
        #         sup_ylabel=r"Posição no corpo, ${}^{\mathcal{I}}\mathbf{r}$",
        #         figure_save_name="position_body_on_inertial",
        #     )

    def plot_fiber_errors(self, multipler: float, unit: str):
        """
        plots a 4x3 grid showing purely the estimation error for each fiber length.
        """
        fig, axs = plt.subplots(
            4, 3, sharex=True, sharey=True, figsize=(FIG_L, FIG_A * 2), dpi=self.dpi
        )

        for i in range(4):
            for j in range(3):
                fiber_idx = 3 * i + j
                true_len = self.fiber_lengths_true[:, fiber_idx]

                for idx, (label, est_data) in enumerate(self.estimations.items()):
                    color = self.colors[idx % len(self.colors)]
                    est_len = est_data["fiber_lengths"][:, fiber_idx]
                    # error in micrometers
                    error_um = (np.abs(est_len - true_len)) * multipler
                    axs[i, j].plot(self.time, error_um, color=color, label=label)

                axs[i, j].set_ylabel(r"$\ell_{" + str(fiber_idx + 1) + r"}$")
                axs[i, j].grid(True)
                axs[i, j].set_yscale("log")
        handles, labels = axs[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=True,
        )
        # axs[0, 2].legend(loc="upper right", fontsize=8)
        fig.supxlabel(r"Tempo $[\si{\second}]$")
        fig.supylabel("Erro de estimação dos comprimentos das fibras " + unit)
        # fig.suptitle("Erro de Estimação dos Comprimentos Ópticos")
        plt.savefig(
            TESE_IMAGE_FOLDER + f"erro_estimacao_comprimentos_fibras_{self.case}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close("ALL")

    def plot_fiber_estimated_deformations(self, multipler: float, unit: str):
        """
        plots a 4x3 grid showing purely the estimation error for each fiber length.
        """
        fig, axs = plt.subplots(
            nrows=4, ncols=3,sharex=True,figsize=(FIG_L, FIG_A * 2), dpi=self.dpi
        )
        l0 = 0.00300000000000000
        for i in range(4):
            for j in range(3):
                fiber_idx = 3 * i + j
                true_len_strain = (self.fiber_lengths_true[:, fiber_idx]) / l0 - 1.0
                axs[i, j].plot(
                    self.time,
                    true_len_strain * multipler,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Referência",
                )
                for idx, (label, est_data) in enumerate(self.estimations.items()):
                    print(label)
                    color = self.colors[idx % len(self.colors)]
                    est_len_strain = (
                        est_data["fiber_lengths"][:, fiber_idx]
                    ) / l0 - 1.0
                    axs[i, j].plot(
                        self.time,
                        est_len_strain * multipler,
                        color=color,
                        label=label,
                    )

                axs[i, j].set_ylabel(r"$\varepsilon_{" + str(fiber_idx + 1) + r"}$")
        handles, labels = axs[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=True,
        )
        # axs[0,0].set_ylim([-120,120])
        # axs[0, 2].legend(loc="upper right", fontsize=8)
        fig.supxlabel(r"Tempo $[\si{\second}]$")
        fig.supylabel("Deformações das fibras " + unit)
        # fig.suptitle("Erro de Estimação dos Comprimentos Ópticos")
        plt.savefig(
            TESE_IMAGE_FOLDER + f"deformacoes_estimadas_fibras_{self.case}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close("ALL")

    def plot_temperature(self):
        """
        plots the estimated temperature gradient if the 7-dof model was loaded.
        """
        has_temp = any("dT_est" in est for est in self.estimations.values())
        if not has_temp:
            return

        fig, ax = plt.subplots(1, 1, dpi=self.dpi, figsize=(FIG_L, FIG_A))
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
                ax.plot(
                    self.time,
                    est_data["dT_true"],
                    color="black",
                    lw=2,
                    label="Referência",
                )
                ax.plot(
                    self.time,
                    est_data["dT_est"],
                    color=self.colors[1],
                    label="Estimado",
                )
        _ax_error.set_ylabel(r"Erro de $\Delta T\; [\si{\degreeCelsius}]$")
        ax.set_ylabel(r"$\Delta T\; [\si{\degreeCelsius}]$")
        ax.set_xlabel(r"Tempo $[\si{\second}]$")
        # ax.set_title("Estimação da variação de temperatura")
        ax.grid(True)
        ax.legend()
        plt.savefig(
            TESE_IMAGE_FOLDER + f"temperature_estimation_{self.case}.pdf", format="pdf"
        )
        plt.close("all")

    def show_all(self):
        self.plot_specific_force()
        self.plot_angular_acceleration()
        self.plot_relative_position()
        self.plot_relative_orientation()
        self.plot_fiber_errors(multipler=1e9, unit="$[\\si{\\nano\\meter}]$")
        self.plot_fiber_estimated_deformations(
            multipler=1e6, unit=r"[\si{\micro\strain}]"
        )
        self.plot_temperature()
        self._plot_trajectory_translational()
        self._plot_trajectory_rotational()
        plt.show()


def plot_graphics(case: str):
    visualizer = inverse_problem_visualizer(case)

    # load multiple datasets to compare
    # visualizer.add_estimation(
    #     label="3-DOF",
    #     filepath="./modeling/data/inverse_output_closed_form_translacional.h5",
    # )
    visualizer.add_estimation(
        key="inverse_inverse_output_closed_form_translacional_angular", label="6-DOF"
    )
    visualizer.add_estimation(
        label="7-DOF",
        key="inverse_inverse_output_closed_form_translacional_angular_thermal",
    )

    visualizer.add_estimation(
        label="\\emph{Push-pull} cruzado",
        key="inverse_output_optical_push_pull_cruzed",
    )
    visualizer.add_estimation(
        label="\\emph{Push-pull} alinhado",
        key="inverse_output_optical_push_pull_aligned",
    )
    # visualizer.add_estimation(label="Levenberg-Marquardt", key="inverse_output_lm.h5")

    # generate and show all plots
    visualizer.show_all()


if __name__ == "__main__":
    plot_graphics("sinusoidal_with_temp_perturbation")


#

# # Nova Paleta 'Synthwave Legível' para Fundo Branco
# revised_white_bg_palette = [
#     "#10002B",  # Deep Purple Base (quase preto)
#     "#FF007F",  # Fúcsia Elétrico
#     "#FFD700",  # Ouro Elétrico Rico
#     "#00E6F0",  # Ciano Elétrico Saturado
#     "#7D00FF",  # Ultravioleta Elétrico (subst. Lavanda)
#     "#00C853",  # Esmeralda Profundo (subst. Verde Claro)
#     "#FF6E40",  # Coral Elétrico (uma nova cor para mais distinção)
#     "#26A69A",  # Turquesa Profundo
# ]

# pastel_synth_palette = [
#     '#BDE0FE', # Fundo Base (Azul-pó Pastel)
#     '#FFDFB4', # Fundo Topo (Pêssego Pastel)
#     '#80EEFF', # Curva Principal (Ciano Pastel Elétrico com Brilho)
#     '#E0D7FF', # Área de Erro (Lavanda Pastel Transparente)
#     '#FFD59F', # Tendência slope=0 (Pêssego Neon Suave com Brilho)
#     '#9DFFB0', # Tendência slope=1/2 (Menta Neon Suave com Brilho)
#     '#FFB3E6', # Tendência slope=-1/2 (Rosa Neon Suave com Brilho)
#     '#FF99CC'  # Outra Tendência (Magenta Neon Suave com Brilho)
# ]

# synthwave_light_palette = [
#     '#3c1053',  # dark purple
#     '#cc0066',  # deep magenta
#     '#008080',  # teal / dark cyan
#     '#e65c00',  # sunset orange
#     '#6600cc'   # electric purple
# ]

# t = np.linspace(0, 1, 1000)
# for i,color in enumerate(my_colors[:8]):

# # Função para visualizar a nova paleta
# def display_revised_palette(color_list):
#     if plt.fignum_exists(1):
#         plt.close(1)
#     fig, ax = plt.subplots(num=1,figsize=(10, 2))
#     for index, hex_color in enumerate(color_list):
#         # ax.add_patch(plt.Rectangle((index, 0), 1, 1, color=hex_color))
#         y = np.sin(2*np.pi*t-.1*index)
#         ax.plot(t, y, color=hex_color,lw=.1)

#     plt.show()

# # configure grid and background for academic layout
# plt.grid(color='#e0e0e0', linestyle='--', linewidth=0.5)
# plt.gca().set_facecolor('#f0ffff')
# # # Executa a visualização
# display_revised_palette(revised_white_bg_palette)
# display_revised_palette(pastel_synth_palette)
# display_revised_palette(my_colors)
# display_revised_palette(synthwave_light_palette)
