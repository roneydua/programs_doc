import numpy as np
import matplotlib.pyplot as plt
import locale
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = FIG_L * 10/16
TESE_FOLDER = "./../tese/images/used_on_thesis/"


def plot_otimizacao_parametrica_frequencia():

    # Parâmetros
    g = 9.81
    lambda_b = 1550e3  # em pm
    p_e = 0.21
    rho_al = 2698.9
    E_silica = 70e9

    # Vetores
    l0 = np.linspace(0.001, 0.005, 100)  # 1mm a 3mm
    freqs = [1000, 1561, 2000]
    d_125 = 125e-6
    d_80 = 80e-6
    A_125 = np.pi * (d_125**2) / 4
    A_80 = np.pi * (d_80**2) / 4


    def calc_ssp(l0:float,wn:float):
        return 2 * lambda_b * (1.0 - p_e) * (g / (l0 * wn**2))

    fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,figsize=(FIG_L, 1.25*FIG_A))

    for fn in freqs:
        wn = 2 * np.pi * fn
        S_pp = calc_ssp(l0,wn)
        ax1.plot(l0 * 1000, S_pp, label=f'$f_n$ = {fn} Hz')

    ax1.plot(
        3,
        calc_ssp(3e-3, 1561 * 2 * np.pi),
        "o",
        color=my_colors[1],
        markerfacecolor=my_colors[4],
        markersize=5,
    )
    ax1.plot(
        2,
        calc_ssp(2e-3, 1561 * 2 * np.pi),
        "+",
        color=my_colors[4],
        markerfacecolor=my_colors[4],
        markersize=8,
    )

    def calc_L125(l0,wn):
        return ((4 * E_silica * A_125) / (rho_al * l0 * wn**2)) ** (1 / 3)
    def calc_L80(l0,wn):
        return ((4 * E_silica * A_80) / (rho_al * l0 * wn**2)) ** (1 / 3)

    for i, fn in enumerate(freqs):
        wn = 2 * np.pi * fn
        # L para 125 um
        L_80 = calc_L80(l0,wn)
        ax2.plot(l0 * 1000, L_80 * 1000, color=my_colors[i], linestyle='--',lw=1.5, label=f'{fn} Hz (80 $\\mu$m)')
        L_125 = calc_L125(l0,wn)
        ax2.plot(l0 * 1000, L_125 * 1000, color=my_colors[i], linestyle='-', label=f'{fn} Hz (125 $\\mu$m)')
        # L para 80 um
    ax2.plot(
        3,
        1000 * calc_L125(3e-3, 1561 * 2 * np.pi),
        "o",
        color=my_colors[1],
        markerfacecolor=my_colors[4],
        markersize=5,
    )
    ax2.plot(
        2,
        1000 * calc_L125(2e-3, 1561 * 2 * np.pi),
        "+",
        color=my_colors[4],
        markerfacecolor=my_colors[4],
        markersize=8,
    )
    fig.supxlabel(r'Comprimento inicial $\ell_0$ [$\si{\milli\meter}]$')
    ax1.set_ylabel(r"Sensibilidade $[\si{\pico\meter\per\g_{E}}]$")
    ax1.legend()
    ax2.set_ylabel(r'Aresta da massa $L$ $[\si{\milli\meter}]$')
    ax2.legend(ncols=3)
    plt.savefig(TESE_FOLDER+'otimizacao_parametrica_frequencia.pdf', format="pdf")
    # plt.show()

if __name__ == "__main__":
    plot_otimizacao_parametrica_frequencia()