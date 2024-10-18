#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2023/02/26 17:39:37
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''
import locale
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import fixed, interactive
from matplotlib import ticker

from bragg.bragg import Bragg, OpticalCoupler

from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import fixed, interactive

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
# plt.style.use("default")
plt.style.use("common_functions/roney3.mplstyle")
cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
TESE_FOLDER = "../tese/images/not_used_on_thesis"
FIG_L = 6.29
# change for 16:10
FIG_A = FIG_L / 1.6

l_peak = 1550.0
delta_l = 5000

bragg = Bragg(fbg_size=6.5e-3,
              delta_n=1e-4,
              wavelength_peak=l_peak,
              delta_span_wavelength=delta_l,
              diff_of_peak=1)


# plt.plot(bragg.wavelength_span_nm, bragg.r0)
# bragg.r0.max()


def calc_laser(w, center_w, std=0.1 * 1e-9):
    return np.exp(-(w - center_w)**2 / (2.0 * std**2))


def plot_reflection_of_transmition(deformation=0.0):
    '''Graphic of reflection of transmition with two FBG with peaks of the same wavelength'''
    fig, ax = plt.subplots(2, 1, sharex=True, num=1)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax[0].set_ylabel('')
    ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
    ax[0].set_ylim((0, 1))
    ax[1].set_ylim((0, 1))
    ax[1].set_xlim((1550, 1551))
    laser = calc_laser(bragg.wavelength_span, 1550.5 * 1e-9, std=0.02 * 1e-9)

    def plot(_r, p, label=''):
        # ax.plot(bragg.wavelength_span_nm,np.ones(delta_l),ls+cores,label='Banda Larga')
        ax[1].plot(bragg.wavelength_span_nm, _r, color=cores[p], label=label)

    op_ratio = .5
    tm = np.array(
        [[1.0 / np.sqrt(1 - op_ratio), 0.0], [0.0, np.sqrt(1 - op_ratio)]],
        dtype=np.float16)
    S_II_I = np.zeros((2, 2), dtype=np.complex128)
    S_IV_III = np.zeros((2, 2), dtype=np.complex128)
    r_E_osa_minus = np.zeros(bragg.wavelength_span.size, dtype=np.float32)
    r_1 = np.zeros(bragg.wavelength_span.size, dtype=np.float32)
    r_2 = np.zeros(bragg.wavelength_span.size, dtype=np.float32)

    # deformation = 1e-5
    for i in range(bragg.wavelength_span.size):
        S_II_I = bragg.reflection_of_transmition(
            deformation=-deformation, wavelength=bragg.wavelength_span[i])

        S_IV_III = bragg.reflection_of_transmition(
            deformation=deformation, wavelength=bragg.wavelength_span[i])
        # r[i] = bragg.calc_reflectance(_S = S_II_I @ tm @ S_IV_III)
        # matrix of point IV to I
        S_IV_I = S_II_I @ tm @ S_IV_III
        # TODO: check the 0.5 factor!
        r_E_osa_minus[i] = np.linalg.norm(
            1j * np.sqrt(op_ratio) * S_IV_III[1, 0] / S_IV_I[0, 0]) * laser[i]
        r_1[i] = bragg.calc_reflectance(_S=S_II_I)
        r_2[i] = bragg.calc_reflectance(_S=S_IV_III)
    ax[0].plot(bragg.wavelength_span_nm, laser, label='Laser')
    ax[0].plot(bragg.wavelength_span_nm, r_1, label=r'$\text{FBG}_1$')
    ax[0].plot(bragg.wavelength_span_nm, r_2, label=r'$\text{FBG}_2$')
    plot(r_E_osa_minus, p=0, label='Ref of trans.')
    plot((1 - r_1) * laser, p=1, label='Transm.')
    plot(r_2 * laser, p=2, label='Ref2')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title(str(np.float16(deformation)))


# interactive(plot_reflection_of_transmition, deformation=(-2e-4, 2e-4, 1e-5))


def plot_bragg_spectrum():
    fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A*0.75))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(r'Refletividade [\unit{\percent}]')
    ax.set_xlabel(r'$\lambda [\unit{\nm}]$')

    def plot(_r, i, p):
        # ax.plot(bragg.wavelength_span_nm,np.ones(delta_l),ls+cores,label='Banda Larga')
        ax.plot(bragg.wavelength_span_nm,
                100.0*_r,
                color=cores[p],
                label=i + ' -- Reflex')
        ax.plot(bragg.wavelength_span_nm,
                100.0*(1.0 - _r),
                label=i + ' -- Trans',
                color=cores[p],
                ls='-.')

    plot(bragg.calc_bragg(-0.0001), r"$d^{\prime}$", 0)
    plot(bragg.r0, r"$\varepsilon=0$", 1)
    plot(bragg.calc_bragg(0.0001), r"$d^{\prime\prime}$", 2)
    ax.set_xlim(1549.5, 1550.5)
    plt.legend(ncols=3,
               bbox_to_anchor=(0, 1, 1, 0),
               loc="lower left",
               mode="expand")
    plt.savefig(TESE_FOLDER+"/bragg_spectrum.pdf", format="pdf")
    plt.close(fig=1)


def plot_drawFig6_spectres():

    def make_figure():
        fig, ax = plt.subplots(1, 1, sharey=True, num=1, figsize=(3.9, 1.9))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))

        ax.set_ylabel(r'Refletividade')
        ax.set_xlabel(r'$\lambda [\si{\nm}]$')
        ax.set_xlim(1549, 1551)
        return fig, ax

    def plot(_ax, _r, i, p):
        # ax.plot(bragg.wavelength_span_nm,np.ones(delta_l),ls+cores,label='Banda Larga')
        _ax.plot(bragg.wavelength_span_nm,
                 _r,
                 color=cores[p],
                 label=i + r' (Reflexão)',
                 lw=1.)
        _ax.plot(bragg.wavelength_span_nm,
                 1 - _r,
                 label=i + r' (Transmissão)',
                 color=cores[p],
                 ls='-.',
                 lw=1)

    # draw figure with zero strain
    fig, ax = make_figure()
    plot(ax, bragg.calc_bragg(0.), r"$\Delta\varepsilon = 0$", 0)
    # plot(bragg.r0, r"$\varepsilon=0$",1)
    plot(ax, bragg.calc_bragg(0.), r"$\Delta\varepsilon = 0$", 1)
    plt.legend(ncols=2,
               bbox_to_anchor=(0, 1, 1, 0),
               loc="lower left",
               mode="expand")
    plt.savefig(TESE_FOLDER+"/drawFig6_zero_spectre.pdf", format="pdf")
    plt.close(fig=1)
    # draw figure with positive strain on fbg1
    de = .25e-4
    fig, ax = make_figure()
    plot(ax, bragg.calc_bragg(-de), r"$\Delta\varepsilon \leq 0$", 0)
    plot(ax, bragg.calc_bragg(de), r"$\Delta\varepsilon \geq 0$", 1)
    plt.legend(ncols=2,
               bbox_to_anchor=(0, 1, 1, 0),
               loc="lower left",
               mode="expand")
    plt.savefig(TESE_FOLDER+"/drawFig6_positive_dx_spectre.pdf",
                format="pdf")
    plt.close(fig=1)
    # draw figure with negative strain on fbg1
    fig, ax = make_figure()
    plot(ax, bragg.calc_bragg(de), r"$\Delta\varepsilon \geq 0$", 0)
    plot(ax, bragg.calc_bragg(-de), r"$\Delta\varepsilon \leq 0$", 1)
    plt.legend(ncols=2,
               bbox_to_anchor=(0, 1, 1, 0),
               loc="lower left",
               mode="expand")
    plt.savefig(TESE_FOLDER+"/drawFig6_negative_dx_spectre.pdf",
                format="pdf")
    plt.close(fig=1)


def plot_response_transmition_reflection():
    fig, ax = plt.subplots(1, 1, num=1)
    laser = calc_laser(bragg.wavelength_span, 1550.05 * 1e-9, 0.02 * 1e-9)
    # ax.plot(bragg.wavelength_span_nm, laser ,label="Laser Source",lw=0.5)
    de = np.arange(-5e-5, 5e-5, 1e-6)
    max_E = np.zeros(de.size)
    for i in range(de.size):
        r_expand = bragg.calc_bragg(deformation=de[i])
        r_compression = bragg.calc_bragg(deformation=-de[i])
        max_E[i] = (laser * (r_expand) * (r_compression)).max()
    ax.plot(de, max_E)
    ax.set_xlabel(r'$\Delta x,\si{\um}$')
    # ax.plot(bragg.wavelength_span_nm, r_expand * r_compression)


def plot_pot_vs_deformation():

    def dbm2W(power):
        return 10.0**(power * 0.1)

    laser = pd.read_csv(
        Path('../../../../experimentos/25042023/laser_1549042523__131644.csv'))
    laser_power_w = dbm2W(laser.power)

    bragg = Bragg(fbg_size=1.85e-4,
                  delta_n=3.15e-3,
                  wavelength_peak=1546.25,
                  diff_of_peak=20)
    e = np.arange(-5e-5, 5e-5, 5e-6)
    pot_reflection = 0.0*e
    pot_transmition = 0.0*e
    for i in range(e.size):
        r = bragg.calc_bragg(deformation=e[i],
                            wavelength_vector=laser.wave_length * 1e-9)

        pot_reflection[i] = np.trapz(x=laser.wave_length * 1e-9, y=0.5*r * laser_power_w)
        pot_transmition[i] = np.trapz(x=laser.wave_length * 1e-9, y=1.0/6.0*(1.0 - r) * laser_power_w)

    fig, ax = plt.subplots(2, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    fig.supylabel("$\\si{\\pico\\watt}$")
    fig.supxlabel(r"$\si{\micro\varepsilon}$")
    ax[0].plot(1e6*e,pot_reflection*1e12,label="Reflection")
    ax[1].plot(1e6 * e, pot_transmition * 1e12, label="Transmition")
    ax[0].legend()
    ax[1].legend()
    # ax[1].set_xlim(-5e1, 5e1)
    plt.savefig("../../images/pot_vs_deformation.pdf", format="pdf")
    plt.close(fig=1)


def graphicsAnimation(_e=0):

    def dbm2W(power):
        return 10.0**(power * 0.1)

    laser = pd.read_csv(
        Path('../../../../experimentos/25042023/laser_1549042523__131644.csv'))
    laser_power_w = dbm2W(laser.power)
    fig, ax = plt.subplots(1, 1, num=1,figsize=(5,3))
    ax.set_ylabel("$\\si{\\watt\\per\\nm}$")
    ax.set_xlabel("$\\lambda,\\si{\\nm}$")
    ax.set_ylim(0, 2)
    ax.set_xlim(1548,1550)
    bragg = Bragg(fbg_size=1.85e-4,
                  delta_n=3.15e-3,
                  wavelength_peak=1546.25,
                  diff_of_peak=20)
    r = bragg.calc_bragg(deformation=_e,
                         wavelength_vector=laser.wave_length * 1e-9)
    pot = r * laser_power_w
    pot_total = np.trapz(x=laser.wave_length * 1e-9, y=pot)
    ax.plot(laser.wave_length, pot,label='{:2.4f}'.format(pot_total*1e12)+"$\\si{\\pico\\watt}$")
    ax.legend()
    # ax.plot(laser.wave_length,laser_power_w)


# w = interactive(graphicsAnimation, _e=(-5e-5, 5e-5, 1e-6))
# display(w)

# interactive(graphicsAnimation, _e=(-5e-5, 5e-5, 1e-6))

# plot_drawFig6_spectres()
# plot_bragg_spectrum()
# plot_bragg_spectrum()
