#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   osa_interpratation.py
@Time    :   2023/04/18 19:06:57
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import numpy as np
import sympy as sp
import locale
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from acquisitionAnritsu import dataAcquisition
from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import interactive, fixed
import pandas as pd
from pathlib import Path
from matplotlib import ticker

# InteractiveShell.ast_node_interactivity = "all"
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.style.use("default")
plt.style.use("~/Dropbox/pacotesPython/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4


def dbm2W(power):
    return 10.0**(power * 0.1)


# %% Read with Pandas.


# ax.plot(da[0].wave_length, dbm2W(da[0].power - 10.0 * np.log10(0.945)))



# dw = 0.1


# interactive(graphicsAnimation, n=(1, 50, 2))
# graphicsAnimation(20)


data1 = '../../../../experimentos/19042023/0.1nm_041923__093424_200.0mA'
data2 = '../../../../experimentos/19042023/1nm_041923__093424_200.0mA'
data3 = '../../../../experimentos/19042023/0.1nm_actResOff_041923__102109_200.0mA'
data4 = '../../../../experimentos/19042023/1nm_actResOff_041923__102109_200.0mA'

# data5 = '../../../../experimentos/19042023/1nm_0945nm_041923__115444_200.0mA'
da = []
da.append(dataAcquisition(test_name=data1))
da.append(dataAcquisition(test_name=data2))
da.append(dataAcquisition(test_name=data3))
da.append(dataAcquisition(test_name=data4))
# da.append(dataAcquisition(test_name=data5))

ax.plot(da[1].wave_length, da[1].power)
ax.plot(da[0].wave_length, da[0].power)
ax.plot(da[0].wave_length, da[0].power - 10.0 * np.log10(0.102))
ax.plot(da[1].wave_length, da[1].power - 10.0 * np.log10(0.945))

def dbm2W(power):
    return 10.0**(power * 0.1)


# %% Read with Pandas.

def plot_laser():
    fig, ax = plt.subplots(1, 1, num=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    data1 = '../../../../experimentos/25042023/laser_1549042523__131644.csv'
    laser_data = []
    laser_data.append(pd.read_csv(Path(data1)))
    data2 = '../../../../experimentos/25042023/laser_1549042523__131735.csv'
    laser_data.append(pd.read_csv(Path(data2)))
    ax.set_ylabel(r'$\si{\watt\per\nm}$')
    ax.set_xlabel(r'$\lambda,\si{\nm}$')
    ax.set_xlim(left=1548.5,right=1550)
    for i in [0]:
        laser_watt = dbm2W(laser_data[i].power)
        pot = np.trapezoid(x=laser_data[i].wave_length * 1e-9, y=laser_watt)
        index_max_y = np.argmax(laser_watt)
        ax.plot(laser_data[i].wave_length, laser_watt, label="{:.2f}".format(pot*1e9)+r'$\si{\nano\watt}$,('+"{:.2f}".format(laser_data[i].wave_length[index_max_y])+r'$\si{nm}$)')
    ax.legend()
# ax.plot(da[0].wave_length, dbm2W(da[0].power - 10.0 * np.log10(0.945)))
