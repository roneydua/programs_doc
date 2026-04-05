import locale
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.style.use("common_functions/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4


def dbm2W(power):
    return 10.0**(power * 0.1)


def plot_laser():

    def calc_laser(w, center_w, std=0.1 * 1e-9):
        return np.exp(-(w - center_w)**2 / (2.0 * std**2))

    fig, ax = plt.subplots(1, 1, num=2,figsize=(FIG_L/2,FIG_A*.5))
    data1 = '../../../../experimentos/25042023/laser_1549042523__131644.csv'
    laser_data = []
    laser_data.append(pd.read_csv(Path(data1)))
    data2 = '../../../../experimentos/25042023/laser_1549042523__131735.csv'
    laser_data.append(pd.read_csv(Path(data2)))
    ax.set_ylabel(r'$\si{\watt\per\nm}$')
    ax.set_xlabel(r'$\lambda,\si{\nm}$')
    ax.set_xlim(left=1549, right=1549.5)
    for i in [1]:
        laser_watt = dbm2W(laser_data[i].power)
        pot = np.trapezoid(x=laser_data[i].wave_length * 1e-9, y=laser_watt)
        index_max_y = np.argmax(laser_watt)
        ax.plot(laser_data[i].wave_length,
                laser_watt,
                label="{:.2f}".format(pot * 1e9) + r'$\si{\nano\watt}$(' +
                "{:.2f}".format(laser_data[i].wave_length[index_max_y]) +
                r'$\si{nm}$)')

        # laser_legend = r'$e^{-\frac{1}{2}\frac{\left(\lambda-1549,28\right)^{2}}{0,025^{2}}}$'
        # laser_gaussian = laser_watt[index_max_y] * calc_laser(
        #     w=laser_data[i].wave_length,
        #     center_w= laser_data[i].wave_length[index_max_y],
        #     std=0.025)
        # ax.plot(laser_data[i].wave_length, laser_gaussian, label=laser_legend)
    ax.legend()
    plt.savefig("./../../images/laser.pdf", format="pdf")
    plt.close(fig=2)

plot_laser()
