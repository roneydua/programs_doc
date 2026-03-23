#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   graphics_optical_mechanical_identification.py
@Time    :   2023/07/17 18:34:34
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

from pydoc import text
from modeling.math_model_accel import AccelModelInertialFrame
import locale

# import uncertainties as un
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.linalg import eig

from common_functions.generic_functions import *

# locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
plt.style.use("./common_functions/roney3.mplstyle")

TESE_FOLDER = "../tese/images/not_used_on_thesis/"

my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.style.use("default")
FIG_L = 6.29
FIG_A = (90.0) / 25.4
am = AccelModelInertialFrame()
max_traction_10g = am.seismic_mass * 10.0 * 9.89 / 4.0
# load data


def plot_colleted_data_fbg_6():
    # load hdf5 file with source and fbg
    _f = h5py.File("./phd_data.hdf5", "r")
    f = _f["optical_mechanical_identification/test_fibers/fbg_6/test_003/"]
    fbg_data = []
    source_data = []
    for key in f.keys():
        if key.startswith("fbg_6"):
            fbg_data.append(key)
        else:
            source_data.append(key)
    # fig.clear()
    fo_area = (125e-6) ** 2 * np.pi * 0.25
    diff_start = 2650
    diff_end = 3000
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A))
    for fbg in reversed(fbg_data):
        deformation = (
            f[fbg].attrs["micrometer_position_um"]
            * 1e-3
            / f[fbg].attrs["initial_length_m"]
        )
        ax.plot(
            f[fbg]["wavelength"][:],
            f[fbg]["power_dbm"][:],
            # label=locale.format_string('T=%.2f', f[fbg].attrs['traction_N']) +
            # r"$\si{\newton}$(" +
            # locale.format_string('\\text{m}$\\varepsilon$=%.2f',
            # deformation)+")")
            label=locale.format_string("%.2f", -f[fbg].attrs["traction_N"])
            + r"$\si{\newton}$",
        )
    ax.axvspan(
        xmin=f[fbg]["wavelength"][diff_start],
        xmax=f[fbg]["wavelength"][diff_end],
        ymin=0,
        ymax=1,
        alpha=0.1,
        facecolor=my_colors[0],
        edgecolor=my_colors[0],
    )
    ax.plot(
        f[source_data[0]]["wavelength"][:],
        f[source_data[0]]["power_dbm"][:],
        label="Fonte",
    )
    ax.set_xlim(1500, 1575)
    ax.set_ylim(bottom=-45, top=-32)
    # ax.legend(ncols=2)
    ax.legend(ncol=6, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand")
    ax.set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    ax.set_ylabel(r"Potência óptica [$\unit{\dbm\per\nm}$]")
    plt.savefig(TESE_FOLDER+"plot_colleted_data_fbg_6.pdf", format="pdf")
    plt.close(fig=1)
    _f.close()


def plot_reflectivity_of_colleted_data_fbg_6():
    f = h5py.File("./phd_data.hdf5", "r")[
        "optical_mechanical_identification/test_fibers/fbg_6/test_003/"
    ]
    fbg_data = []
    source_data = []
    for key in f.keys():
        if key.startswith("fbg_6"):
            fbg_data.append(key)
        elif key.startswith("fonte"):
            source_data.append(key)
    # fig.clear()
    diff_start = 2600
    diff_end = 3000
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A))
    for fbg in reversed(fbg_data):
        diff = (
            f[source_data[0]]["power_dbm"][diff_start:diff_end]
            - f[fbg]["power_dbm"][diff_start:diff_end]
        ).mean()
        # fix bias between source and fbg collected data

        source_fix = f[source_data[0]]["power_dbm"] - diff
        # compute reflectivity
        r = 1.0 - 10.0 ** (0.1 * (f[fbg]["power_dbm"][:] - source_fix))
        ax.plot(
            f[source_data[0]]["wavelength"][:],
            100.0 * r,
            lw=0.5,
            label=locale.format_string("%.2f", -f[fbg].attrs["traction_N"])
            + r"$\si{\newton}$",
        )

    ax.set_xlim(1520, 1560)
    ax.set_ylim(bottom=-0.1, top=100)
    ax.legend(ncols=2)
    ax.set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    ax.set_ylabel(r"\text{r}\,[$\unit{\percent}$]")
    plt.savefig(
        TESE_FOLDER+"plot_reflectivity_of_colleted_data_fbg_6.pdf",
        format="pdf",
    )
    plt.close(fig=1)


def plot_reflectivity_of_colleted_data_fbg_6_zero_strain():
    f = h5py.File("./phd_data.hdf5", "r")[
        "optical_mechanical_identification/test_fibers/fbg_6/test_003/"
    ]
    fbg_data = []
    source_data = []
    for key in f.keys():
        if key.startswith("fbg_6"):
            fbg_data.append(key)
        elif key.startswith("fonte"):
            source_data.append(key)
    # fig.clear()
    diff_start = 2600
    diff_end = 3000
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A))
    r = []
    ax.set_xlim(1520, 1560)
    ax.set_ylim(bottom=-0.1, top=100)
    for fbg in reversed(fbg_data):
        diff = (
            f[source_data[0]]["power_dbm"][diff_start:diff_end]
            - f[fbg]["power_dbm"][diff_start:diff_end]
        ).mean()
        # fix bias between source and fbg collected data

        source_fix = f[source_data[0]]["power_dbm"] - diff
        # compute reflectivity
        r = 1.0 - 10.0 ** (0.1 * (f[fbg]["power_dbm"][:] - source_fix))
        break
        ax.plot(
            f[source_data[0]]["wavelength"][:],
            100.0 * r,
            lw=0.5,
            label=locale.format_string("%.2f", -f[fbg].attrs["traction_N"])
            + r"$\si{\newton}$",
        )
    index_ref = np.where(f[source_data[0]]["wavelength"][:] > 1545)[0][0]
    index_max_r = r[index_ref : index_ref + 100].argmax()
    # ax.plot(f[source_data[0]]['wavelength'][index_ref+index_max_r],
    #         100*r[index_ref+index_max_r], '*', label=locale.format_string('$\\lambda=$%.2f', f[source_data[0]]['wavelength'][index_ref+index_max_r])+locale.format_string(', $r=$ %.2f', r[index_ref+index_max_r]*100)+r'\%')
    l_bragg = 1545.68  # nm
    p_e = 0.2635
    traction = 0.27  # N
    young = 69e9
    fo_area = (125e-6) ** 2 * np.pi * 0.25
    delta_brag = (1.0 - p_e) * traction / young / fo_area * l_bragg
    ax.plot(
        -delta_brag + f[source_data[0]]["wavelength"][:],
        100 * r,
        label=locale.format_string(
            "$\\lambda_\\text{g}=$%.2f\\unit{\\nm}",
            -delta_brag + f[source_data[0]]["wavelength"][index_ref + index_max_r],
        )
        + locale.format_string(", $r=$ %.2f", r[index_ref + index_max_r] * 100)
        + r"\%",
    )
    # find fwhm
    index_fwhm = (
        index_ref
        + np.where(r[index_ref : index_ref + 100] < r[index_ref + index_max_r] * 0.5)[
            0
        ][0]
    )
    ax.plot(
        f[source_data[0]]["wavelength"][index_fwhm] - delta_brag,
        100 * r[index_fwhm],
        "+",
    )
    print(
        f[source_data[0]]["wavelength"][index_fwhm]
        - f[source_data[0]]["wavelength"][index_ref + index_max_r]
    )

    ax.legend(ncols=2)
    ax.set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    ax.set_ylabel(r"\text{r}\,[$\unit{\percent}$]")

    plt.savefig(
        TESE_FOLDER+"plot_reflectivity_of_colleted_data_fbg_6_zero_strain.pdf",
        format="pdf",
    )
    plt.close(fig=1)


def calcule_optical_effective_coeficiente():
    # estimation of E and l0
    _f = h5py.File("./../data/phd_data.hdf5", "r")
    f = _f["optical_mechanical_identification/test_fibers/fbg_6/test_003"]
    fbg_data = []
    fo_area = (125e-6) ** 2 * np.pi * 0.25
    for key in f.keys():
        if key.startswith("fbg"):
            fbg_data.append(key)
    _N = len(fbg_data)
    l_bragg = np.ones(_N)
    mat_a = np.ones((_N, 2))
    for i in range(len(fbg_data)):
        deformation = -f[fbg_data[i]].attrs["traction_N"] / 69e9 / fo_area
        mat_a[i, 1] = deformation
        index_max_ref = np.argmax(f[fbg_data[i]]["reflectivity"][:3500])
        l_bragg[i] = f[fbg_data[i]]["wavelength"][index_max_ref]
    lambda_bragg, _temp = (np.linalg.inv(mat_a.T @ mat_a) @ mat_a.T) @ l_bragg
    p_e = 1.0 - _temp / lambda_bragg


def traction_vs_deformation():
    data = h5py.File("./../data/phd_data.hdf5", "r")[
        "optical_mechanical_identification/test_fibers/fbg_6/"
    ]
    tests = ["test_003"]
    fo_area = (125.4e-6) ** 2 * np.pi * 0.25
    l0 = 39.7e-3
    for t in tests:
        traction = np.array([])
        delta_l = np.array([])
        mat_a = np.array([])
        vec_y = np.array([])
        _N = 0
        for key in data[t].keys():
            if key.startswith("fbg_6"):
                traction = np.append(traction, -data[t][key].attrs["traction_N"])
                delta_l = np.append(
                    delta_l, data[t][key].attrs["micrometer_position_um"]
                )
                _N += 1
            mat_a = np.append(mat_a, fo_area * (delta_l[-1]) * 1e-6 / l0)
            vec_y = np.append(vec_y, traction[-1])
        E = 1.0 / (mat_a.T @ mat_a) * mat_a.T @ vec_y
        print(1e-9 * E)

        plt.plot(1e-6 * delta_l / l0, traction)
        plt.plot(1e-6 * delta_l / l0, fo_area * E * delta_l * 1e-6 / l0, "*")


def calc_reflectivity(fbg_n, test_n, w_start=1556, w_end=1570, plot_graphic=False):
    """
    calc_reflectivity Calculates reflectivity and records in the HDF5 file

    Args:
        fbg_n: FBG number to be identified
        test_n: Test number in the HDF5 file
        w_start: Start wavelength. Defaults to 1556.
        w_end: Cutting wavelength. Defaults to 1570.
        plot_graphic: Presents or not the chart. Defaults to False.
    """
    _f = h5py.File("./../data/phd_data.hdf5", "a")
    f = _f["optical_mechanical_identification/test_fibers/" + fbg_n + "/" + test_n]

    fbg_data = []
    source_data = []
    for key in f.keys():
        if key.startswith("fbg"):
            fbg_data.append(key)
        elif key.startswith("fonte"):
            source_data.append(key)
    if plot_graphic:
        fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A))
    for fbg in reversed(fbg_data):

        l_start = np.where(f[source_data[0]]["wavelength"][:] > w_start)[0][0]
        l_end = np.where(f[source_data[0]]["wavelength"][:] > w_end)[0][0]
        diff = (
            f[source_data[0]]["power_dbm"][l_start:l_end]
            - f[fbg]["power_dbm"][l_start:l_end]
        ).mean()
        # fix bias between source and fbg collected data

        source_fix = f[source_data[0]]["power_dbm"] - diff
        # compute reflectivity
        r = 1.0 - 10.0 ** (0.1 * (f[fbg]["power_dbm"][:] - source_fix))
        f[fbg]["reflectivity"] = r
        if plot_graphic:
            ax.plot(
                f[source_data[0]]["wavelength"][:],
                100.0 * r,
                lw=0.5,
                label=locale.format_string("%.2f", -f[fbg].attrs["traction_N"])
                + r"$\si{\newton}$",
            )
    if plot_graphic:
        ax.set_xlim(1520, 1560)
        ax.set_ylim(bottom=-0.1, top=100)
    _f.close()


# calc_reflectivity('fbg_2', 'test_003',plot_graphic=True)


# p11 = 0.113
# p12 = 0.252
# v = 0.16
# n = 1.482
# 1550 * (1.0 - 0.5 * n**2 * (p12 - v * (p11 + p12)))


def identification_method_examples():
    f = h5py.File("./phd_data.hdf5", "r")[
        "optical_mechanical_identification/test_fibers/fbg_6/test_001/"
    ]
    w = f["fonte001/wavelength"][:]
    pi = f["fonte001/power_dbm"][:]
    r = f["fbg_6_001/reflectivity"][:]

    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, 0.75 * FIG_A))
    ax.plot(w, pi, label=r"$\mathbf{p}_{\mathrm{I}}$")
    ax.plot(w, mW_dbm(dbm_mW(dbm=pi) * 0.5), label=r"$\mathbf{p}_{\mathrm{II}}$")
    reflected_spectrum = r * dbm_mW(dbm=pi) * 0.5
    reflected_reference = dbm_mW(dbm=pi) * 0.5 * 0.04
    ax.plot(w, mW_dbm(reflected_spectrum * 0.5), label=r"$\mathbf{p}_{\mathrm{r}}$")
    ax.plot(
        w,
        mW_dbm((reflected_spectrum + reflected_reference) * 0.5),
        label=r"$\mathbf{p}_{\mathrm{r^{\prime}}}$ (Método 1)",
    )
    ax.plot(
        w,
        mW_dbm((reflected_reference) * 0.5),
        label=r"$\mathbf{p}_{\mathrm{r^{\prime}}}$ (Método 2)",
    )

    ax.set_xlim(1520, 1560)
    ax.legend(ncols=2)
    ax.set_ylim(bottom=-60)
    ax.set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    ax.set_ylabel(r"Potência óptica [\unit{\dbm}]")
    plt.savefig(
        TESE_FOLDER+"identification_method_examples.pdf",
        format="pdf",
    )
    plt.close(fig=1)
    # transmistion figure

    w = f["fonte001/wavelength"][:]
    pi = f["fonte001/power_dbm"][:]
    pii = f["fbg_6_001/power_dbm"][:]

    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, 0.75 * FIG_A))
    ax.plot(w, pi, label=r"$\mathbf{p}_{\mathrm{I}}$")
    ax.plot(w, pii, label=r"$\mathbf{p}_{\mathrm{II}}$")
    ax.set_xlim(1520, 1560)
    ax.legend(ncols=2)
    ax.set_ylim(bottom=-45)
    ax.set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    ax.set_ylabel(r"Potência óptica [\unit{\dbm}]")
    plt.savefig(
        TESE_FOLDER+"identification_method_examples_2.pdf",
        format="pdf",
    )
    plt.close(fig=1)


def plot_time_elapsed_fbg_production(language:str):
    f = h5py.File("./production_files.hdf5", "r")["fbg_production/20240328/fbg2"]
    w = f["wavelength_m"][:]
    pi = f["optical_power_dbm"][:]
    r = f["reflectivity"][:]
    texts_defs  = {
        "en":{
            'power_dmb':"Optical Power [dBm]",
            'reflectivity':"Reflectivity",
            "locale": "en_US.UTF-8",
        },
        "pt":{
            'power_dmb':"Potência óptica [dBm]",
            'reflectivity':"Refletividade",
            "locale": "pt_BR.UTF-8",
        }
    }
    texts = texts_defs[language]
    locale.setlocale(category=locale.LC_ALL,locale=texts["locale"])
    print(texts["locale"])
    if plt.fignum_exists(1):
        plot.close('all')
    fig, ax = plt.subplots(2, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    alpha_line = np.linspace(0, 1, 9,endpoint=True)
    for i in range(0, 17, 2):
        ax[0].plot(w*1e9, pi[:, i], "-", color=my_colors[0], alpha=alpha_line[i//2])
        ax[1].plot(w * 1e9, r[:, i], "-", color=my_colors[0], alpha=alpha_line[i // 2])
    # ax.set_ylim(bottom=-45)
    ax[0].set_ylabel(texts['power_dmb'])
    ax[1].set_ylabel(texts['reflectivity'])
    ax[1].set_xlabel(r"$\lambda\,[\unit{\nm}]$")
    plt.savefig(
        TESE_FOLDER+"plot_time_elapsed_fbg_production_"+language+".pdf",
        format="pdf",
    )
    plt.close(1)
