#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   erbium_laser_graphics.py
@Time    :   2023/05/16 07:54:45
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import numpy as np
import sympy as sp
import locale
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import glob

# InteractiveShell.ast_node_interactivity = "all"
plt.style.use("common_functions/roney3.mplstyle")
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = (90.0) / 25.4

folder = "./data/15052023/erbium_source_test/"

files = glob.glob(folder + "laser_bobina_15m**.csv")

current = np.arange(100, 1200, 50)

file_names = glob.glob(folder + "laser_bobina_1500mm_M12**.csv")


def get_data():
    list_of_data = []
    list_of_data.append([])
    for j in file_names:
        list_of_data[-1].append(pd.read_csv(j, index_col=[0]))
        print(list_of_data[-1][-1].ilx_current[0], "\t",
              list_of_data[-1][-1].erbium_fiber_size[0])

    for i in [2, 8, 13, 15, 17, 19, 21]:
        list_of_data.append([])
        file_names = glob.glob(folder + "laser_bobina_" + str(i) +
                               "m_M12**.csv")
        for j in file_names:
            list_of_data[-1].append(pd.read_csv(j, index_col=[0]))
            print(list_of_data[-1][-1].ilx_current[0], "\t",
                  list_of_data[-1][-1].erbium_fiber_size[0])
    return list_of_data


# fix power resolution
fix_power_resolution = lambda d: d.power_dbm - 10.0 * np.log10(
    d.actual_resolution)

dBm2mW = lambda dBm: 10.0**(dBm * 0.1)
mW2dBm = lambda mW: 10.0 * np.log10(mW)
power_mW = lambda d: np.trapezoid(x=d.wavelength,
                              y=(dBm2mW(d.power_dbm - 10.0 * np.log10(
                                  d.actual_resolution[0]))))


def find_wavelength_of_peak():
    list_of_data = get_data()
    wavelength_of_peak = lambda d: d.wavelength[d.power_dbm.argmax()]
    wavelength_of_peak_list = []
    for _fiber in range(len(list_of_data)):
        for _current in range(len(list_of_data[_fiber])):
            _t = wavelength_of_peak(list_of_data[_fiber][_current])
            if _t > 1545.0 and _t < 1552.0:
                wavelength_of_peak_list.append(_t)
    wavelength_of_peak = np.array(wavelength_of_peak_list)
    plt.plot(wavelength_of_peak, '*', lw=0.1)
    wavelength_of_peak.mean()
    wavelength_of_peak.std()


def laser_pump_vs_current():
    list_of_data = get_data()
    legend_fiber_size = ['1.5', '2', '8', '13', '15', '17', '19', '21']
    # fig.clear()
    fig, ax = plt.subplots(1,
                           1,
                           sharex=True,
                           num=1,
                           figsize=(FIG_L, 0.75 * FIG_A))
    # the first index of list_of_data is the fiber size, the sequence obeys the sequence of legend_fiber_size
    for _fiber in range(len(list_of_data)):
        # for _fiber in [0]:
        current_mA = []
        total_power_mW = []
        # the second index of list_of_data is the current with no order
        for _current in range(len(list_of_data[_fiber])):
            current_mA.append(list_of_data[_fiber][_current].ilx_current[0])
            # Fix resolution values
            total_power_mW.append(power_mW(list_of_data[_fiber][_current]))
        current_mA, total_power_mW = zip(
            *sorted(zip(current_mA, total_power_mW)))
        current_mA_1500m = np.array(current_mA)
        total_power_mW_1500m = np.array(total_power_mW)
        ax.plot(current_mA,
                total_power_mW,
                '*-',
                lw=.75,
                label=legend_fiber_size[_fiber] + 'm')
        # ax.plot(current_mA,
        #            mW2dBm(total_power_mW),
        #            '*-',
        #            lw=.75,
        #            label=legend_fiber_size[_fiber] + 'm')
    # ax.legend(ncol=3)
    ax.legend(ncol=8,
              bbox_to_anchor=(0, 1, 1, 0),
              loc="lower left",
              mode="expand")
    # fig.legend(ncol=3)
    # fig.supylabel('Potência')
    ax.set_ylabel('$\\si{\\milli\\watt}$')
    # ax.set_ylabel('$\\si{dBm}$')
    ax.set_xlabel("Corrente, \\si{\\milli\\ampere}")
    # ax.set_xlim(right=500)
    # ax.set_ylim(top=1.5)
    plt.savefig("../../images/laser_pump_vs_current.pdf", format="pdf")
    plt.close(fig=1)


# power_vs_current()


def transimpedance_tension_vs_source_current():
    ktr = 230300.0
    r = np.array([10, 20, 25]) * 0.01
    i = np.arange(0.15, 0.51, .05)
    s_i = lambda _i: 2.790 * _i - 0.393  # mW
    _i = 1.0 / 2.790 * (1500.0 / ((17. * ktr / 480.) * 0.2) + 0.393)
    # 0.00279005 - 0.39293894
    # 2.790 - 0.393
    if plt.fignum_exists(1):
        fig.clear()  # type: ignore
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, 0.75 * FIG_A))
    for index in range(r.size):
        v = 17. / 480.0 * ktr * r[index] * s_i(i)
        # i = 1.0/2.790 * (1.5/17/480*230300*r + 0.393)
        ax.plot(1e3 * i,
                1e-3 * v,
                label=r'r=' + '{:1.0f}'.format(100 * r[index]) +
                r"\unit{\percent}")
        ax.plot(1e3 * i, 1e-3 * v * 6, '--', color=my_colors[index])
    ax.set_ylim(bottom=0, top=5)
    # ax.legend(ncols=3)
    ax.legend(ncol=3,
              bbox_to_anchor=(0, 1, 1, 0),
              loc="lower left",
              mode="expand")
    ax.set_xlabel(r'$i^{\text{s}}$, \unit{\milli\ampere}')
    ax.set_ylabel(r"$v^{\text{tr}}$,\unit{\volt}")
    fig.set_dpi(72)
    plt.savefig("../../images/transimpedance_tension_vs_source_current.pdf",
                format="pdf")
    plt.close(fig=1)


plot_tension_vs_corrent()
# poly_coef = np.polyfit(x=current_mA_1500m, y=total_power_mW_1500m,deg=1)
# p = np.poly1d(poly_coef)
# plt.plot(current_mA_1500m, p(current_mA))
# plt.plot(current_mA_1500m, total_power_mW_1500m)


def transimpedance_tension_vs_source_current_transmission():
    ktr = 230300.0
    r = np.array([10, 20, 25]) * 0.01
    i = np.arange(0.15, 0.21, .05)
    s_i = lambda _i: 2.790 * _i - 0.393  # mW
    _i = 1.0 / 2.790 * (1500.0 / ((17. * ktr / 480.) * 0.2) + 0.393)
    # 0.00279005 - 0.39293894
    # 2.790 - 0.393
    if plt.fignum_exists(1):
        fig.clear()  # type: ignore
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, 0.75 * FIG_A))
    for index in range(r.size):
        v = 17. / 480.0 * ktr * (1.0 - r[index]) * s_i(i)
        # i = 1.0/2.790 * (1.5/17/480*230300*r + 0.393)
        ax.plot(1e3 * i,
                1e-3 * v,
                label=r'r=' + '{:1.0f}'.format(100 * r[index]) +
                r"\unit{\percent}")
        ax.plot(1e3 * i, 1e-3 * v * 6, '--', color=my_colors[index])
    ax.set_ylim(bottom=0, top=5)
    # ax.legend(ncols=3)
    ax.legend(ncol=3,
              bbox_to_anchor=(0, 1, 1, 0),
              loc="lower left",
              mode="expand")
    ax.set_xlabel(r'$i^{\text{s}}$, \unit{\milli\ampere}')
    ax.set_ylabel(r"$v^{\text{tr}}$,\unit{\volt}")

    plt.savefig(
        "../../images/transimpedance_tension_vs_source_current_transmission.pdf",
        format="pdf")
    plt.close(fig=1)





def transimpedance_tension_vs_gravity_with_gains():
    coef = pd.read_csv("./data/transimpedance_tension_vs_strain_coefficients.csv")
    p = lambda ind, deformation: coef['coef_ang'][ind]*deformation + coef['coef_lin'][ind]
    max_deformation = 3.4265e-5
    deformation_vector = np.linspace(-max_deformation,
                                     max_deformation,
                                     5,
                                     endpoint=True)
    gravity_vector = deformation_vector / max_deformation
    leg = [
        r'$y_-$', r'$z_{+}$', r'$z_-$', r'$y_+$', r'$x_{+}$', r'$x_-$'
    ]
    fig.clear()
    fig, ax = plt.subplots(1, 1, num=1 , figsize=(FIG_L, FIG_A))
    for i in range(6):
        graphic = p(i,deformation_vector)*1e-3
        ax.plot(gravity_vector,graphic,label=leg[i])

    fig.supxlabel(r"Aceleração,$\unit{g}$")
    fig.supylabel(r"$v^{\text{tr}}$, $\unit{\volt}$")
    ax.legend(ncols=2)
    plt.savefig("../../images/transimpedance_tension_vs_gravity_with_gains.pdf", format="pdf")
    plt.close(fig=1)



def transimpedance_tension_vs_gravity_with_constant_gains():
    coef = pd.read_csv('data/transimpedance_tension_vs_strain_coefficients_constant_gain.csv')
    p = lambda ind, deformation: coef['coef_ang'][ind]*deformation + coef['coef_lin'][ind]
    max_deformation = 3.4265e-5
    deformation_vector = np.linspace(-max_deformation,
                                     max_deformation,
                                     5,
                                     endpoint=True)
    gravity_vector = deformation_vector / max_deformation
    leg = [
        r'$y_-$', r'$z_{+}$', r'$z_-$', r'$y_+$', r'$x_{+}$', r'$x_-$'
    ]
    fig.clear()
    fig, ax = plt.subplots(1, 1, num=1 , figsize=(FIG_L, FIG_A))
    for i in range(6):
        graphic = p(i,deformation_vector)*1e-3
        print(1e3*(graphic.max()-graphic.min()))
        ax.plot(gravity_vector,graphic,label=leg[i])

    fig.supxlabel(r"Aceleração,$\unit{g}$")
    fig.supylabel(r"$v^{\text{tr}}$, $\unit{\volt}$")
    ax.legend(ncols=2)
    plt.savefig(
        "../../images/transimpedance_tension_vs_gravity_with_constant_gains.pdf",
        format="pdf")
    plt.close(fig=1)
