# %%
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pure_reflection_and_pure_transmission_analysis.ipynb
@Time    :   2023/05/08 15:45:23
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import locale

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import fixed, interact, interactive
from modeling.math_model_accel import AccelModelInertialFrame
from matplotlib import ticker
from IPython.display import display, Markdown

from bragg.bragg import Bragg
from common_functions.generic_functions import *
from bragg.pure_reflection_and_pure_transmission_analysis_Functions import *

plt.style.use("./common_functions/roney3.mplstyle")
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = FIG_L / 1.6
plt.rcParams["figure.dpi"] = 288
plt.rcParams["figure.figsize"] = (FIG_L, FIG_A)
# Create a FBG on lambda with 1550nm

accel = AccelModelInertialFrame()
max_deformation = accel.seismic_mass * 9.89 * 0.25 / accel.k / accel.fiber_length
deformation_vector = np.linspace(-max_deformation, max_deformation, 5, endpoint=True)
gravity_span_vector = (
    4.0 * accel.k * accel.fiber_length * deformation_vector / accel.seismic_mass
)
update_pot_vs_gravity = True

TESE_FOLDER = "./../tese/images/used_on_thesis/"


def interrogation_laser_one_fbg():
    _bragg = Bragg(
        fbg_size=2e-3,
        delta_n=4e-4,
        delta_span_wavelength=5000,
        diff_of_peak=1.0,
        wavelength_peak=1550,
    )
    _laser = calc_laser(
        w=_bragg.wavelength_span, center_w=1550.25 * 1e-9, std=0.0005 * 1e-9
    )
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A * 0.75))

    ax.plot(_bragg.wavelength_span_nm, _bragg.r0, label=r"$\varepsilon=0$")
    ax.plot(
        _bragg.wavelength_span_nm + 0.05,
        _bragg.r0,
        color=my_colors[2],
        label=r"$\varepsilon>0$",
    )
    ax.plot(
        _bragg.wavelength_span_nm - 0.05,
        _bragg.r0,
        color=my_colors[3],
        label=r"$\varepsilon<0$",
    )
    # total_power = np.trapezoid(x=_bragg.wavelength_span, y=_laser)
    # ax[1].xaxis.set_major_locator(ticker.MultipleLocator(.5))
    ax2 = ax.twinx()
    ax2.yaxis.label.set_color(my_colors[1])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(my_colors[1])
    ax2.tick_params(axis="y", colors=my_colors[1])
    ax2.plot(_bragg.wavelength_span_nm, _laser / _laser.max(), color=my_colors[1])
    ax2.set_ylabel(r"Potência refletida $[\unit{\mW}]$")
    ax.set_ylabel(r"Refletividade")
    fig.supxlabel("$\\lambda$ [\\si{\\nm}]")
    ax.legend()
    plt.savefig(TESE_FOLDER + "/interrogation_laser_one_fbg.pdf", format="pdf")
    plt.close(fig=1)


def interrogation_two_fbgs_reflection_of_reflection():
    _bragg_l = Bragg(
        fbg_size=2e-3,
        delta_n=4e-4,
        delta_span_wavelength=5000,
        diff_of_peak=1.0,
        wavelength_peak=1550,
    )
    _bragg_r = Bragg(
        fbg_size=2e-3,
        delta_n=4e-4,
        delta_span_wavelength=5000,
        wavelength_span=_bragg_l.wavelength_span,
        diff_of_peak=1.0,
        wavelength_peak=1550.5,
    )
    a = -(0.173 - 0.4093) / 0.1
    b = 0.2963 - a * 1550.25
    r1 = lambda l: a * l + b
    r2 = lambda l: -a * l + 2 * (a * 1550.25) + b
    r1(1550.25)
    r2(1550.25)
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A * 0.75))
    # ax.set_ylim(0, 0.6)
    ax.set_xlim(1549.5, 1551.0)
    vector_wavelength = np.linspace(1550.15,1550.35,3)

    def plot_with_approximation(dl,_alpha):
        ax.plot(
            _bragg_l.wavelength_span_nm - dl,
            _bragg_l.r0,
            '-',
            label=r"$\text{FBG}_{1}$",
            color=my_colors[0],
            alpha=_alpha
        )
        ax.plot(
            _bragg_r.wavelength_span_nm + dl,
            _bragg_r.r0,
            '-',
            label=r"$\text{FBG}_{2}$",
            color=my_colors[1],
            alpha=_alpha,
        )
        ax.plot(
            _bragg_r.wavelength_span_nm,
            _bragg_r.calc_bragg(dl / (1550.0 * 0.8))
            * _bragg_l.calc_bragg(-dl / (1550.5 * 0.8)),
            '-',
            label=r"$\mathbf{r}_{1}\mathbf{r}_{2}$",
            alpha=_alpha,
            color=my_colors[3],
        )
        ax.plot(
            vector_wavelength,
            r1(vector_wavelength + dl) * r2(vector_wavelength - dl),
            "--",
            color=my_colors[3],
            alpha=_alpha,
        )
    alpha = np.linspace(1, .1, 10)
    deformation = np.linspace(-.02, .02,alpha.size)
    for i in range(alpha.size):
        plot_with_approximation(deformation[i], _alpha=alpha[i])
        if i == 0:
            ax.legend()

    ax.set_ylabel(r"Refletividade")
    fig.supxlabel("$\\lambda$ [\\si{\\nm}]")

    plt.savefig(
        TESE_FOLDER + "/interrogation_two_fbgs_reflection_of_reflection.pdf",
        format="pdf",
    )
    plt.close(fig=1)


def interrogation_two_fbgs_reflection_of_reflection_equation_analysis():
    p11 = 0.113
    p12 = 0.252
    v = 0.16
    n_eff = 1.482
    pe = n_eff**2 * (p12 - v * (p11 + p12)) / 2.0
    alpha = 0.55e-6
    zeta = 8.6e-6
    l1 = 1550.0e-9
    l2 = 1550.5e-9
    d1 = -(1-pe)*l1
    d2 = -(1 - pe) * l2
    e1 = -(alpha+zeta)*l1
    e2 = -(alpha + zeta) * l2
    d1
    d2
    e1
    e2
    d1
# t = np.arange(-4, 0, 0.01)
# dt = 1

# a = 3.0
# b = 6.0

# f1 = lambda _t: a * _t + b
# f2 = lambda _t: -a * _t - b
# integral = lambda _t: -1 / (3 * a) * (a * _t + b) ** 3
# fig.clear()
# fig, ax = plt.subplots(1, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))

# ax.plot(t, f1(t + dt))
# ax.plot(t, f2(t - dt))
# ax.plot(t, f1(t + dt) * f2(t - dt))

# ax.plot(-b / a, f1(-b / a + dt), "+", ms=10)
# integral(-1) - integral(-3)
