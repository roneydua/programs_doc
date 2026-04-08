#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   push_pull_interrogation_analysis.py
@Time    :   2024/01/25 14:42:05
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

import numpy as np
import sympy as sp
from sympy.integrals.risch import risch_integrate
import locale
import matplotlib.pyplot as plt
import h5py

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
from modeling.math_model_accel import AccelModelInertialFrame
from common_functions.generic_functions import calc_laser

FIG_L = 6.29
FIG_A = (90.0) / 25.4
from scipy.ndimage import shift

TESE_FOLDER = "./../tese/images/used_on_thesis/"


class fbg_simulation(object):
    def __init__(self, fbg_name_d, fbg_name_e, date=""):
        f = h5py.File("./production_files.hdf5", "r")
        # date = "20240328"
        # fbg_name_d = "fbg12"
        # fbg_name_e = "fbg15"
        # date = ""

        w_fbg_e_ = f["fbg_production/" + date + "/" + fbg_name_d + "/wavelength_m"][:]
        "wavelength in meters"
        fbg_e_ = f["fbg_production/" + date + "/" + fbg_name_d + "/reflectivity"][:, -1]

        w_fbg_d_ = f["fbg_production/" + date + "/" + fbg_name_e + "/wavelength_m"][:]
        "wavelength in meters"
        fbg_d_ = f["fbg_production/" + date + "/" + fbg_name_e + "/reflectivity"][:, -1]
        f.close()
        # find peak of power
        w_fbg_d_argmax = fbg_d_.argmax()
        self.w_fbg_max_d = w_fbg_d_[w_fbg_d_argmax]
        w_fbg_e_argmax = fbg_e_.argmax()
        self.w_fbg_max_e = w_fbg_e_[w_fbg_e_argmax]

        self.w_fbg_d_interp, self.w_fbg_d = self.extend_vector(w_fbg_d_, "wavelength")
        "wavelength in meters"
        self.fbg_d = self.extend_vector(fbg_d_, "reflectivity")
        self.fbg_d_interp = np.interp(self.w_fbg_d_interp, self.w_fbg_d, self.fbg_d)

        self.w_fbg_e_interp, self.w_fbg_e = self.extend_vector(w_fbg_e_, "wavelength")
        "wavelength in meters"
        self.fbg_e = self.extend_vector(fbg_e_, "reflectivity")
        self.fbg_e_interp = np.interp(self.w_fbg_e_interp, self.w_fbg_e, self.fbg_e)
        self.power_of_second_reflected_spectrum_mW = 0.0
        self.step_of_w_fbg = np.diff(self.w_fbg_d_interp).mean()
        "wavelength in meters"

        self.amplitude_w_by_m = 20e-3 / 50e-9  # 20mW distributed in  40nm
        self.translate_fbgs(0)

    def extend_vector(self, v: np.ndarray, type_of_data: str):
        if type_of_data == "wavelength":
            _w = np.hstack(
                (v - (v.size * np.diff(v).mean()), v, v + (v.size * np.diff(v).mean()))
            )
            return np.linspace(_w[0], _w[-1], 200_000), _w

        else:
            return np.hstack((np.zeros(v.size), v, np.zeros(v.size)))

    def translate_fbgs(self, delta_lambda_d: float, delta_lambda_e=None):
        index_to_shift_d = int(delta_lambda_d // self.step_of_w_fbg)
        if delta_lambda_e == None:
            index_to_shift_e = -index_to_shift_d
        else:
            index_to_shift_e = int(delta_lambda_e // self.step_of_w_fbg)

        self.fbg_d_shifted = shift(
            self.fbg_d_interp, index_to_shift_d, mode="grid-wrap"
        )
        self.fbg_e_shifted = shift(
            self.fbg_e_interp, index_to_shift_e, mode="grid-wrap"
        )
        self.make_dot_product()

    def make_dot_product(self):
        self.first_reflected_spectrum = self.amplitude_w_by_m * self.fbg_e_shifted

        self.second_reflected_spectrum = (
            self.first_reflected_spectrum * self.fbg_d_shifted
        )
        self.power_of_second_reflected_spectrum_W = (
            np.trapezoid(y=self.second_reflected_spectrum) * self.step_of_w_fbg
        )


def power_vs_delta_lambda_animation_two_fibers():
    fbgs = fbg_simulation(fbg_name_d="fbg12", fbg_name_e="fbg15",date='20240328')
    alpha = np.linspace(0.2, 1, 3)
    fig, ax = plt.subplots(
        2,
        1,
        num=1,
        sharex=True,
        figsize=(FIG_L, FIG_A),
        # dpi=72
    )
    marker_type = ["-D", "-o", "-^"]
    index = 0
    for i in np.linspace(-0.2, 0.2, 3):
        print(i)
        #
        # ax.plot(fbgs.w_fbg_d * 1e9, fbgs.fbg_d_shifted)
        # ax.plot(fbgs.w_fbg_e * 1e9, fbgs.fbg_e_shifted)
        # fig.clear()
        fbgs.translate_fbgs(i * 1e-9)

        fbgs.amplitude_w_by_m
        ax[0].plot(
            fbgs.w_fbg_e_interp * 1e9,
            fbgs.fbg_e_shifted,
            # lw=0.5,
            ls="-",
            color=my_colors[0],
            alpha=alpha[index],
        )
        ax[0].plot(
            fbgs.w_fbg_d_interp * 1e9,
            fbgs.fbg_d_shifted,
            # lw=0.5,
            ls="-",
            color=my_colors[3],
            alpha=alpha[index],
        )
        ax[1].plot(
            fbgs.w_fbg_d_interp * 1e9,
            fbgs.second_reflected_spectrum * 1e-6,
            # lw=0.5,
            ls="-",
            color=my_colors[2],
            alpha=alpha[index],
            label=locale.format_string(
                r"%.0f \unit{\micro\watt}",
                fbgs.power_of_second_reflected_spectrum_W * 1e6,
            ),
        )
        index += 1
    ax[0].set_ylabel("Refletividade")
    ax[1].legend()
    ax[1].set_ylabel(r"Potência óptica [\unit{\milli\watt\per\nano\meter}]")
    ax[1].set_xlabel(r"Comprimento de onda [\unit{\nano\meter}]")
    ax[0].set_xlim(left=1550, right=1560)
    # plt.pause(0.01)

    plt.savefig(
        TESE_FOLDER + "power_vs_delta_lambda_animation_two_fibers.pdf", format="pdf"
    )
    plt.close(1)


def power_vs_delta_lambda_animation_one_fiber():
    fbgs = fbg_simulation(fbg_name_d="fbg12", fbg_name_e="fbg15",date='20240328')
    alpha = np.linspace(0.2, 1, 3)
    fig, ax = plt.subplots(
        2,
        1,
        num=1,
        sharex=True,
        figsize=(FIG_L, FIG_A),
        # dpi=72
    )
    marker_type = ["-D", "-o", "-^"]
    index = 0
    fbgs.translate_fbgs(0)
    ax[0].plot(
        fbgs.w_fbg_e_interp * 1e9,
        fbgs.fbg_e_shifted,
        # lw=0.5,
        ls="-",
        color=my_colors[0],
    )
    for i in np.linspace(-0.2, 0.2, 3):

        #
        # ax.plot(fbgs.w_fbg_d * 1e9, fbgs.fbg_d_shifted)
        # ax.plot(fbgs.w_fbg_e * 1e9, fbgs.fbg_e_shifted)
        # fig.clear()
        fbgs.translate_fbgs(i * 1e-9, 0)
        ax[0].plot(
            fbgs.w_fbg_d_interp * 1e9,
            fbgs.fbg_d_shifted,
            # lw=0.5,
            ls="-",
            color=my_colors[3],
            alpha=alpha[index],
        )
        ax[1].plot(
            fbgs.w_fbg_d_interp * 1e9,
            fbgs.second_reflected_spectrum * 1e-6,
            # lw=0.5,
            ls="-",
            color=my_colors[2],
            alpha=alpha[index],
            label=locale.format_string(
                r"%.0f \unit{\micro\watt}",
                fbgs.power_of_second_reflected_spectrum_W * 1e6,
            ),
        )
        index += 1
    ax[0].set_ylabel("Refletividade")
    ax[1].legend()
    ax[1].set_ylabel(r"Potência óptica [\unit{\milli\watt\per\nano\meter}]")
    ax[1].set_xlabel(r"Comprimento de onda [\unit{\nano\meter}]")
    ax[0].set_xlim(left=1550, right=1560)
    # plt.pause(0.01)

    plt.savefig(
        TESE_FOLDER + "power_vs_delta_lambda_animation_one_fiber.pdf", format="pdf"
    )
    plt.close(1)


def plot_power_vs_accel_accel1(date, fbg_name_d, fbg_name_e, fig_name_to_save):

    fbgs = fbg_simulation(date=date, fbg_name_d=fbg_name_d, fbg_name_e=fbg_name_e)
    accel = AccelModelInertialFrame()

    def compute_delta_lambda(dd_r: float, delta_T=0.0):
        lambda_e = 1548e-9
        lambda_d = 1552e-9
        lambda_d = fbgs.w_fbg_max_d
        lambda_e = fbgs.w_fbg_max_e
        pe = 0.23
        alpha = 0.55e-6
        zeta = 8.60e-6
        # calc epsilon with epsilon assembly and epsilon of measurement
        # epsilon_m
        epsilon_m = (
            accel.seismic_mass * dd_r / (accel.E * np.pi * accel.fiber_diameter**2)
        )
        epsilon_a = 23e-4
        temp_var = (1.0 - pe) * (epsilon_m + epsilon_a) + (alpha + zeta) * delta_T
        delta_l_e = lambda_e * temp_var
        delta_l_d = lambda_d * temp_var
        return delta_l_e, delta_l_d

    dd_r_vec = np.arange(start=-100, stop=101, step=20)
    pot_vec = 0.0 * dd_r_vec
    delta_l_d = 0.0 * dd_r_vec
    delta_l_e = 0.0 * dd_r_vec

    for i, _ in enumerate(dd_r_vec):
        delta_l_e[i], delta_l_d[i] = compute_delta_lambda(dd_r_vec[i], 0)
        # print(delta_l_e[i], delta_l_d[i])
        fbgs.translate_fbgs(delta_lambda_d=delta_l_d[i], delta_lambda_e=-delta_l_e[i])
        pot_vec[i] = fbgs.power_of_second_reflected_spectrum_mW
    fig, ax = plt.subplots(2, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    ax[0].plot(dd_r_vec, pot_vec, "-*")
    ax[1].plot(dd_r_vec, -delta_l_e * 1e9, "-*")
    ax[1].plot(dd_r_vec, delta_l_d * 1e9, "-*")
    print(fbgs.w_fbg_max_d)
    ax[1].legend(
        [
            r"$\Delta\lambda_{g,"
            + fbg_name_e[3:]
            + "}$,"
            + locale.format_string(f="%.4f", val=fbgs.w_fbg_max_d * 1e9)
            + r"\unit{\nm}",
            r"$\Delta\lambda_{g,"
            + fbg_name_d[3:]
            + "}$"
            + locale.format_string(f="%.4f", val=fbgs.w_fbg_max_e * 1e9)
            + r"\unit{\nm}",
        ]
    )
    ax[0].set_ylabel(r"Potência óptica [\unit{\nano\watt}]")
    ax[1].set_ylabel("Variação do $\\lambda_{g} [\\unit{\\nano\\meter}]$")
    ax[1].set_xlabel(
        r"Aceleração ${}^\mathcal{B}\ddot{\mathbf{r}}$ [\unit{\meter\per\second\squared}]"
    )

    plt.savefig(
        "../tese/images/plot_power_vs_accel_" + fig_name_to_save + ".pdf", format="pdf"
    )
    plt.close(fig=1)
    print(
        "Accel is null in",
        np.where(dd_r_vec == 0)[0][0],
        "with power in [nW] equal ",
        pot_vec[np.where(dd_r_vec == 0)[0][0]],
    )
    # plt.show()
    # plt.pause(0.01)


def plot_power_vs_accel_complete_acc(
    fbg_name_d: list[str],
    fbg_name_e: list[str],
    fig_name_to_save: str,
    language: str,
    power_gradient=[1.0, 1.0, 1.0],
):
    plot_texts = {
        "pt": {
            "acceleration": r"Acelera\c{c}\~{a}o $[\si{\m\per\second\squared}]$",
            "optical_power_mW": r"Pot\^{e}ncia \^{o}ptica [\unit{\milli\watt}]",
            "locale": "pt_BR.UTF-8",
        },
        "en": {
            "acceleration": r"Acceleration $[\si{\m\per\second\squared}]$",
            "optical_power_mW": r"Optical power [\unit{\milli\watt}]",
            "locale": "en_US.UTF-8",
        },
    }
    texts = plot_texts[language]
    locale.setlocale(category=locale.LC_ALL, locale=texts["locale"])
    accel = AccelModelInertialFrame()
    fig, ax = plt.subplots(1, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    common_labels = {
        "x": r"$\mathbf{x}$",
        "y": r"$\mathbf{y}$",
        "z": r"$\mathbf{z}$",
        "m_per_ss": r"$[\si{\m\per\s\squared}]$",
    }
    cl = common_labels
    legend_name = [cl["x"], cl["y"], cl["z"]]
    f = h5py.File("./phd_data.hdf5", "a")
    # del f["20260108"]
    _ff = f.require_group("mounted_acc/acc_4/20260108")
    if "linearity_analysis" in _ff.keys():
        del _ff["linearity_analysis"]
    ff = _ff.require_group("linearity_analysis")
    dd_r_vec = np.arange(start=-400, stop=600, step=20)
    ff["dd_r_vec"] = dd_r_vec
    for left_fbg, right_fbg, row in zip(fbg_name_d, fbg_name_e, range(3)):
        fbgs = fbg_simulation(fbg_name_d=right_fbg, fbg_name_e=left_fbg)

        def compute_delta_lambda(dd_r: float, delta_T=0.0):
            lambda_e = 1548.5e-9
            lambda_d = 1551.75e-9
            lambda_d = fbgs.w_fbg_max_d
            lambda_e = fbgs.w_fbg_max_e
            pe = 0.23
            alpha = 0.55e-6
            zeta = 8.60e-6
            # calc epsilon with epsilon assembly and epsilon of measurement
            # epsilon_m
            epsilon_m = (
                accel.seismic_mass * dd_r / (accel.E * np.pi * accel.fiber_diameter**2)
            )
            epsilon_a = 23e-4
            temp_var = (1.0 - pe) * (power_gradient[row] * epsilon_m + epsilon_a) + (
                alpha + zeta
            ) * delta_T
            delta_l_e = lambda_e * temp_var
            delta_l_d = lambda_d * temp_var
            return delta_l_e, delta_l_d

        pot_vec = 0.0 * dd_r_vec
        delta_l_d = 0.0 * dd_r_vec
        delta_l_e = 0.0 * dd_r_vec

        for i, _ in enumerate(dd_r_vec):
            delta_l_e[i], delta_l_d[i] = compute_delta_lambda(dd_r_vec[i], 0)
            # print(delta_l_e[i], delta_l_d[i])
            fbgs.translate_fbgs(
                delta_lambda_d=delta_l_d[i], delta_lambda_e=-delta_l_e[i]
            )
            pot_vec[i] = fbgs.power_of_second_reflected_spectrum_W
        ff[list(common_labels.keys())[row]] = pot_vec
        ax.plot(dd_r_vec, pot_vec * 1e3, "-*", label=legend_name[row])

    # ax[1].set_ylabel(legend_name[row])
    ax.axvspan(-20 * 9.81, 20 * 9.81, color="k", alpha=0.08)

    fig.supylabel(texts["optical_power_mW"])
    ax.set_xlabel(texts["acceleration"])
    ax.legend()
    plt.savefig(TESE_FOLDER + fig_name_to_save + "_" + language + ".pdf", format="pdf")
    plt.close(fig=1)
    print("to aqui")

def simulation_push_pull_symbolic():
    def gaussian_func(wavelength, wavelength_center, standart_deviation):
        calc_a = 1 / (standart_deviation * sp.sqrt(2 * sp.pi))
        return calc_a * sp.exp(
            -(((wavelength - wavelength_center) / standart_deviation) ** 2) / 2
        )

    l = sp.symbols(r"\lambda")
    l_center_1 = sp.symbols(r"\lambda_{g\,1}", real=True)
    l_center_2 = sp.symbols(r"\lambda_{g\,2}", real=True)
    standart_deviation = sp.symbols(r"\sigma", real=True)
    f_1 = gaussian_func(l, l_center_1, standart_deviation)
    f_2 = gaussian_func(l, l_center_2, standart_deviation)
    # f_1 * f_2
    # integral_f1_f2 = sp.integrate(f_1 * f_2, l)
    integral_f1_f2 = risch_integrate(f_1 * f_2, l).doit()
    print(sp.latex(integral_f1_f2))
    print(sp.latex(f_2))


# def acc_4_analysis():
#     plot_power_vs_accel_accel1(
#         fbg_name_d="20240328/fbg15",
#         fbg_name_e="20240328/fbg12",
#         fig_name_to_save="acc_6_axis_x",
#     )
#     plot_power_vs_accel_accel1(
#         fbg_name_d="20240328/fbg10",
#         fbg_name_e="20240328/fbg13",
#         fig_name_to_save="acc_6_axis_y",
#     )
#     plot_power_vs_accel_accel1(
#         fbg_name_d="20240328/fbg6",
#         fbg_name_e="20240328/fbg11",
#         fig_name_to_save="acc_6_axis_z",
#     )


def acc_4_analysis(language: str):
    plot_power_vs_accel_complete_acc(
        fbg_name_d=["20240207/fbg3", "20231130/fbg7", "20240207/fbg2"],
        fbg_name_e=["20240207/fbg11", "20240207/fbg9", "20240207/fbg17"],
        fig_name_to_save="acc_4_power_vs_accel",
        power_gradient=[1.0, 1.0, 1.0],
        language=language,
    )


def acc_5_analysis(language="pt"):
    plot_power_vs_accel_complete_acc(
        fbg_name_d=["20240328/fbg15", "20240328/fbg10", "20240328/fbg6"],
        fbg_name_e=["20240328/fbg11", "20240328/fbg12", "20240328/fbg13"],
        fig_name_to_save="acc_5_power_vs_accel",
        language=language,
    )


def acc_6_analysis(language="pt"):
    plot_power_vs_accel_complete_acc(
        fbg_name_e=["20240207/fbg7", "20240328/fbg11", "20240328/fbg10"],
        fbg_name_d=["20240207/fbg12", "20240328/fbg13", "20240328/fbg9"],
        fig_name_to_save="acc_6_power_vs_accel",
        language=language,
    )


def linearity_analysis_acc_4(language:str):
    f = h5py.File("./phd_data.hdf5", "r")
    ff = f["mounted_acc/acc_4/20260108/linearity_analisys"]
    plot_texts = {
        "pt": {
            "domain_name": r"Fundo de escala [\si{\m\per\second\squared}]",
            "image_name": r"Erro de não linearidade [\si{\percent} FS]",
            "locale": "pt_BR.UTF-8",
        },
        "en": {
            "domain_name": r"Full scale [\si{\m\per\second\squared}]",
            "image_name": r"Nonlinearity error [\si{\percent} FS]",
            "locale": "en_US.UTF-8",
        },
    }
    texts = plot_texts[language]
    locale.setlocale(category=locale.LC_ALL, locale=texts["locale"])
    def perform_local_linear_fit(x_data, y_data, x_op, delta):
        """
        Perform a linear least squares fit around an operating point.

        Parameters:
        x_data (np.ndarray): Independent variable data.
        y_data (np.ndarray): Dependent variable data.
        x_op (float): Operating point x-coordinate.
        delta (float): Neighborhood range around the operating point.

        Returns:
        tuple: (angular_coefficient, linear_coefficient) of the local linear model.
        """
        # Filter data within the neighborhood [x_op - delta, x_op + delta]
        mask = (x_data >= x_op - delta) & (x_data <= x_op + delta)
        x_local = x_data[mask]
        y_local = y_data[mask]

        if len(x_local) < 2:
            raise ValueError("Insufficient data points in the specified neighborhood.")

        # Perform first-order polynomial fit (linear)
        # y = m*x + c
        coefficients = np.polyfit(x_local, y_local, 1)
        angular_coefficient, linear_coefficient = coefficients

        return angular_coefficient, linear_coefficient

    # data  = [ff["x"][mask], ff["y"][mask], ff["z"][mask]]
    if plt.fignum_exists("linear_graphics"):
        plt.close("linear_graphics")
    fig, ax = plt.subplots(
        nrows=1, ncols=1, num="linear_graphics", sharex=True, figsize=(FIG_L, FIG_A)
    )
    x_op = 0
    full_scale_vector = np.arange(start=20, stop=200, step=20)
    error =  [[],[],[]]
    axes = ["x", "y", "z"]
    dd_r_vec = ff["dd_r_vec"][:]
    # loop of axes
    for i in range(3):
        # loop to iterate over full scale
        for full_scale in full_scale_vector:
            mask = (dd_r_vec >= -full_scale) & (dd_r_vec <= full_scale)
            dd_r_vec_filtered = dd_r_vec[mask]
            angular_coefficient, linear_coefficient = perform_local_linear_fit(
                x_data=dd_r_vec_filtered,
                y_data=ff[axes[i]][mask],
                x_op=x_op,
                delta=full_scale,
            )
            image_of_full_scale = np.abs(ff[axes[i]][mask][0] - ff[axes[i]][mask][-1])
            error[i].append(100*np.abs(ff[axes[i]][mask] - (angular_coefficient * dd_r_vec_filtered + linear_coefficient)).max()/image_of_full_scale)
        ax.plot(full_scale_vector, error[i], '-*',ms=2,label=r"$\mathbf{"+axes[i]+r"}$")
        print(error[i])
    ax.legend()
    fig.supylabel(plot_texts[language]["image_name"])
    fig.supxlabel(plot_texts[language]["domain_name"])
    plt.savefig(TESE_FOLDER + "linearity_graphic_acc4"+"_"+language+".pdf", format="pdf")
    plt.close(fig="linear_graphics")


if __name__ == "__main__":
    pass
