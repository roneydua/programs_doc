#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graphics_initial_identification.py
@Time    :   2023/03/02 17:31:36
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import numpy as np
import glob
import locale
import matplotlib.pyplot as plt
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.style.use("default")
plt.style.use("~/Dropbox/pacotesPython/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4
from acquisitionAnritsu import dataAcquisition
import os

def dbmToWatt(_t):
    return 1e-3 * 10.0**(_t*.1)

def readData(t):
    da = []
    files = glob.glob("data/"+t)
    print(files)
    for i in files:
        da.append(dataAcquisition(test_name=i))

    return da


def plot_Eirbium_source(dbm=True):
    fig, ax = plt.subplots(1, 1, num=3, sharex=True, figsize=(FIG_L, FIG_A),dpi=144)
    current_of_ILX = np.arange(100,525,25)
    def readCsv():
        for i in range(len(current_of_ILX)):
            t = sorted(glob.glob("data/csv_Prof_Nicolau/wave*_"+str(current_of_ILX[i])+"mA.csv"), key=os.path.getmtime)
            s =  sorted(glob.glob("data/csv_Prof_Nicolau/power*_" +
                                str(current_of_ILX[i]) + "mA.csv"),
                    key=os.path.getmtime)
            for j in range(1):
                wave_length_csv = np.loadtxt(t[j])
                power_csv = np.loadtxt(s[j])
                if dbm==True:
                    ax.plot(wave_length_csv,power_csv,label=str(current_of_ILX[i])+r"$\si{\mA}$")
                else:
                    ax.plot(wave_length_csv,dbmToWatt(power_csv),label=str(current_of_ILX[i]) + r"$\si{\mA}$")
    readCsv()
    ax.legend(ncol=5)
    # ax.set_xlim([1500,1580])
    ax.set_ylim([-60,-15])
    ax.set_ylabel(r"\si{\deci\bel\meter}")
    ax.set_xlabel(r"$\lambda, \si{\nano\meter}$")

plot_Eirbium_source(dbm=True)
    # for i in range(len(wave_length_csv)):


def twoColuns():
    fig, ax = plt.subplots(3, 2, num=1, sharex=True, figsize=(FIG_L, FIG_A),dpi=144)
    def plotAx(_ax, name, _label):
        _da = readData(name)
        for i in range(1):
            _ax.plot(_da[i].wave_length, _da[i].power,label=_label)
        _ax.legend(ncol=2)

    plotAx(ax[0,0], "X+*",r"$x_+$")
    ax[0,0].set_ylabel(r"\si{\decibel}")
    plotAx(ax[0,1], "X-*",r"$x_{-}$")
    plotAx(ax[1,0], "Y+*",r"$y_+$")
    ax[1, 0].set_ylabel(r"\si{\decibel}")
    plotAx(ax[1,1], "Y-*",r"$y_{-}$")
    plotAx(ax[2, 0], "Z+*", r"$z_+$")
    ax[2,0].set_ylabel(r"\si{\decibel}")
    plotAx(ax[2, 1], "Z-*", r"$z_{-}$")
    ax[2,0].set_xlabel(r"$\lambda,\si{\nano\meter}$")
    ax[2,1].set_xlabel(r"$\lambda,\si{\nano\meter}$")




fig, ax = plt.subplots(1, 1, num=2, sharex=True, figsize=(FIG_L, FIG_A),dpi=144)
def plotAx( name, _label):
    _da = readData(name)
    for i in range(1):
        ax.plot(_da[i].wave_length, _da[i].power,label=_label)
    ax.legend(ncol=2)

plotAx("X+*",r"$x_+$")
ax.set_ylabel(r"\si{\decibel}")
plotAx( "X-*",r"$x_{-}$")
plotAx( "Y+*",r"$y_+$")
plotAx( "Y-*",r"$y_{-}$")
plotAx( "Z+*", r"$z_+$")
plotAx("Z-*", r"$z_{-}$")
ax.set_xlabel(r"$\lambda,\si{\nano\meter}$")