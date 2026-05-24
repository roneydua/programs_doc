import numpy as np
import matplotlib.pyplot as plt
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

FIG_L = 6.29
# change for 16:10
FIG_A = FIG_L / 1.6
def plot_geometric_fbg_model():
    # parameters definition
    r_max = 1
    delta_lambda_r = 4.0
    delta_w_0 = 3
    kinematic_shift = 0.25  # shift magnitude for push-pull simulation
    angular = r_max / delta_lambda_r

    # fbg 1 coordinates (reference state - static)
    fbg1_x = np.array([-6, -delta_lambda_r + delta_w_0 / 2, delta_w_0 / 2])
    fbg1_y = np.array([r_max, r_max, 0])

    # fbg 2 coordinates (reference state - static)
    start_fbg2 = -delta_w_0 / 2
    end_ramp_fbg2 = start_fbg2 + delta_lambda_r
    fbg2_x = np.array([start_fbg2, end_ramp_fbg2, 6])
    fbg2_y = np.array([0, r_max, r_max])

    # fbg 1 and 2 dynamic shift coordinates (push-pull state)
    fbg1_x_shifted = fbg1_x - kinematic_shift
    fbg2_x_shifted = fbg2_x + kinematic_shift

    # figure setup
    # if plt.fignum_exists(1):
    # fig.clear()
    fig, ax = plt.subplots(num=1, figsize=(FIG_L, FIG_A),dpi=288)

    # plot static fbgs
    ax.plot(fbg1_x, fbg1_y, color=my_colors[0], linewidth=1, label="FBG 1 (Repouso)")

    ax.plot(fbg2_x, fbg2_y, color=my_colors[2], linewidth=1, label="FBG 2 (Repouso)")

    # plot shifted fbgs
    ax.plot(fbg1_x_shifted, fbg1_y, color=my_colors[0], linestyle="--", alpha=0.6, label="FBG 1 (Transladada)")
    ax.plot(
        fbg2_x_shifted,
        fbg2_y,
        color=my_colors[2],
        linestyle="--",
        alpha=0.6,
        label="FBG 1 (Transladada)",
    )

    # shaded intersection area (static state)
    intersection_x = np.array([-delta_w_0 / 2, delta_w_0 / 2])
    # calculate intersection peak height analytically
    intersection_peak_x = 0
    intersection_peak_y = (r_max / delta_lambda_r) * (delta_w_0 / 2)
    intersection_peak_y_lower = (r_max / delta_lambda_r) * (delta_w_0 / 2 - kinematic_shift)
    def compute_fbg_2(_shift):
        return (r_max / delta_lambda_r) * (delta_w_0 / 2 - _shift)

    def compute_fbg_1(_shift):
        return (-r_max / delta_lambda_r) * (-delta_w_0 / 2 - _shift)

    # plot laser intersection points
    w_laser = 0.5
    ax.plot(w_laser, compute_fbg_2(w_laser), '+',ms=6, color=my_colors[4])
    ax.plot(w_laser, compute_fbg_1(w_laser), '+',ms=6, color=my_colors[4])
    ax.vlines(
        x=w_laser,
        ymin=0,
        ymax=r_max,
        lw=2,
        colors=my_colors[4],
        alpha=0.5,
    )
    ax.plot(w_laser, compute_fbg_2(w_laser+kinematic_shift), "o", ms=6, color=my_colors[4])
    ax.plot(w_laser, compute_fbg_1(w_laser-kinematic_shift), "o", ms=6, color=my_colors[4])

    ax.fill_between(
        x=[-delta_w_0 / 2+kinematic_shift, 0, delta_w_0 / 2-kinematic_shift],
        y1=[0, intersection_peak_y_lower, 0],
        # y2=[0, intersection_peak_y_lower, 0],
        color=my_colors[5],
        alpha=0.3,
        label="Área de Intersecção ($A_1$)",
    )

    ax.fill_between(
        x=[
            -delta_w_0 / 2.0,
            -delta_w_0 / 2.0 + kinematic_shift,
            0,
            delta_w_0 / 2.0 - kinematic_shift,
            delta_w_0 / 2.0,
        ],
        # x=[-2,-1,0,1,2],
        y2=[
            0,
            r_max / delta_lambda_r * kinematic_shift,
            intersection_peak_y,
            r_max / delta_lambda_r * kinematic_shift,
            0,
        ],
        y1=[0, 0, intersection_peak_y_lower, 0, 0],
        color=my_colors[3],
        alpha=0.3,
        label="Área de Intersecção ($A_2$)",
    )
    # Coordenada x do limite inferior (início da rampa de subida da FBG 2 transladada)
    lam_i_2_x = start_fbg2 + kinematic_shift

    # Coordenada x do limite superior (fim da rampa de descida da FBG 1 transladada)
    lam_f_1_x = delta_w_0/2 - kinematic_shift

    # Plota os marcadores (pontos) sobre o eixo x (y=0)
    ax.plot(lam_i_2_x, 0, marker="o", color=my_colors[2], markersize=6)
    ax.plot(lam_f_1_x, 0, marker="o", color=my_colors[0], markersize=6)

    # Insere o texto das variáveis com um pequeno recuo vertical (y = -0.05)
    ax.text(
        lam_i_2_x, -0.05, r"$\lambda_{i_2}$", color=my_colors[2], ha="center", fontsize=12
    )
    ax.text(
        lam_f_1_x, -0.05, r"$\lambda_{f_1}$", color=my_colors[0], ha="center", fontsize=12
    )
    # annotations setup
    # 1. R_max annotation
    ax.annotate(
        r"$R_{max}$",
        xy=(-4.5, r_max),
        xytext=(-5.5, r_max),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        va="center",
    )

    # 2. Delta_lambda_r annotation (width of the ramp)
    ax.annotate(
        "",
        xy=(delta_w_0 / 2, r_max + 0.01),
        xytext=(-delta_lambda_r + delta_w_0 / 2, r_max + 0.01),
        arrowprops=dict(
            arrowstyle="<->",
            lw=1.5,
            color=my_colors[1]),
    )
    ax.text(
        (-delta_lambda_r + delta_w_0) / 2,
        r_max + 0.02,
        r"$\Delta\lambda_r$",
        color=my_colors[1],
        ha="center",
    )

    # 3. Delta_w_0 annotation (initial overlap)
    ax.annotate(
        "",
        xy=(start_fbg2, -0.09),
        xytext=(delta_w_0 / 2, -0.09),
        arrowprops=dict(arrowstyle="<->", lw=1.5,color=my_colors[1]),
    )
    ax.text(
        0,
        -0.07,
        r"$\Delta w_0$",
        color=my_colors[1],
        ha="center",
    )

    # 4. Kinematic shift annotations (Push-Pull)
    ax.annotate(
        r"",
        color=my_colors[0],
        xy=(-delta_w_0 / 2 + kinematic_shift, r_max / 2),
        xytext=(-delta_w_0 / 2 + kinematic_shift - 1, r_max / 2),
        arrowprops=dict(arrowstyle="<-", lw=1.5, color=my_colors[0]),
    )
    ax.text(
        0.5 * (-delta_w_0 + 2 * kinematic_shift - 1),
        0.45,
        r"$+k_\varepsilon\varepsilon-k_{T}\Delta T$",
        color=my_colors[0],
        ha="center",
    )
    ax.annotate(
        r"",
        color=my_colors[2],
        xy=(delta_w_0 / 2 - kinematic_shift, r_max / 2),
        xytext=(delta_w_0 / 2 - kinematic_shift+1, r_max / 2),
        arrowprops=dict(arrowstyle="<-", lw=1.5, color=my_colors[2]),
    )
    ax.text(
        0.5 * (delta_w_0 - 2 * kinematic_shift + 1),
        0.45,
        r"$+k_\varepsilon\varepsilon+k_{T}\Delta T$",
        color=my_colors[2],
        ha="center",
    )
    ax.annotate(
        r"",
        color=my_colors[4],
        xy=(w_laser, 0.8),
        xytext=(w_laser + 0.85, 0.8),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=my_colors[4]),
    )
    ax.text(
        x=w_laser+0.4,
        y=0.78,
        s=r"Laser",
        color=my_colors[4],
        ha="center",
    )

    # # axes formatting
    ax.set_ylim(-0.1, r_max + 0.05)
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(
        "Comprimento de onda " r"$\lambda$",
    )
    ax.set_ylabel(
        "Refletividade " r"$r(\lambda)$",
    )

    # hide top and right spines for a clean academic look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")

    # remove x ticks since the plot is parametric
    ax.set_xticks([])
    ax.set_yticks([0, r_max])
    ax.set_yticklabels(
        ["", r"$r_{max}$"],
    )

    ax.legend(
        ncols=1,
        # loc="lower right",
        # frameon=False,
    )

    # plt.show()
    plt.savefig("./../tese/images/used_on_thesis/schematic_fbg_push_pull_analysis.pdf", format="pdf")
    # plt.close(fig='all')

if __name__ == "__main__":
    plot_geometric_fbg_model()
