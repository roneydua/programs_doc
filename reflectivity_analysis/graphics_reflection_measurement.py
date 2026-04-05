import locale
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bragg.bragg import Bragg

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
plt.style.use("./common_functions/roney3.mplstyle")

my_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = (90.0) / 25.4


a = pd.read_csv(
    Path('../experimentos/24042023/a_reflection_042423__130528.csv'))
a_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/a_reflection_4_percent_042423__130752.csv'))

b = pd.read_csv(
    Path('../experimentos/24042023/b_reflection_042423__124322.csv'))
b_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/b_reflection_4_percent_042423__124714.csv'
    ))

c = pd.read_csv(
    Path('../experimentos/24042023/c_reflection_042423__131535.csv'))
c_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/c_reflection_4_percent_042423__131725.csv'
    ))

d = pd.read_csv(
    Path('../experimentos/24042023/d_reflection_042423_133240.csv'))
d_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/d_reflection_4_percent_042423__133611.csv'
    ))
e = pd.read_csv(
    Path('../experimentos/24042023/e_reflection_042023__142133.csv'))
e_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/e_reflection_4_percent_042023__142133.csv'
    ))

f = pd.read_csv(
    Path('../experimentos/24042023/f_reflection_042423__140137.csv'))
f_4_percent = pd.read_csv(
    Path(
        '../experimentos/24042023/f_reflection_4_percent_042423__140332.csv'
    ))

r = [a, b, c, d, e, f]
r_ref = [
    a_4_percent, b_4_percent, c_4_percent, d_4_percent, e_4_percent,
    f_4_percent
]
# leg = ['A','B','C','D','E','F']
leg = [
    r'$y_-$', r'$z_{+}$', r'$z_-$', r'$y_{+}$', r'$x_{+}$', r'$x_-$',
    r'\text{\emph{Tap}}'
]

laser = pd.read_csv(
    Path('../experimentos/25042023/laser_1549042523__131644.csv'))


def calc_reflectivity(_r, _ref):
    return 3.3735943356934604 / (10.0**((_ref - _r) * 0.1) - 1.0)


def plot_reflection_dbm():
    fig, ax = plt.subplots(6,
                           1,
                           num=1,
                           sharex=True,
                           figsize=(FIG_L, FIG_A * 1.25))
    for lin in range(6):
        ax[lin].plot(r[lin]['wave_length'], r[lin]['power'])
        ax[lin].plot(r_ref[lin]['wave_length'], r_ref[lin]['power'])
        ax[lin].set_ylabel(leg[lin])
        ax[lin].set_ylim(bottom=-80)

    fig.supylabel(r'$\si{dBm\per\nm}$')
    fig.supxlabel(r'$\lambda, \si{\nm}$')
    ax[0].legend(['Reflexão da FBG', 'Reflexão da FBG + 3,37%'],
                 ncol=2,
                 loc='upper center',
                 bbox_to_anchor=(.5, 1.5))
    # plt.savefig("../images/reflection_dbm.pdf", format="pdf")
    # plt.close(fig=1)
    plt.savefig("../images/reflection_dbm.svg", format="svg")
    plt.close(fig=1)



def reflectivity_plots():
    # bragg.r0.max()
    # fig.clear(3)
    _laser_peak = 1549.3
    fig, ax = plt.subplots(1,
                           3,
                           num=3,
                           sharey=True,
                           sharex=True,
                           figsize=(FIG_L, 0.8*FIG_A))

    class linearization_data():

        def __init__(self):
            self.df = pd.DataFrame()
            """Wavelength vector in nano memeter"""
            self.df['r_a'] = np.zeros(6, dtype=np.float32)
            """angular coefficient"""
            self.df['r_l'] = np.zeros(6, dtype=np.float32)
            """linear coefficient"""
            self.df['reflectivity_peak'] = np.zeros(6, dtype=np.float32)
            """Peak de reflectivity"""
            self.df['reflectivity_peak_wavelength'] = np.zeros(
                6, dtype=np.float32)
            """Wavelength of the reflectivity"""

    approx = linearization_data()

    def plotax(axis_number, lin,color_number=0):

        reflectivity = calc_reflectivity(_r=r[lin]['power'], _ref= r_ref[lin]['power'])
        ##
        icf = np.where(r_ref[lin].wave_length >= _laser_peak)[0][0]
        '''Index center of fitting'''
        coef_fit = np.polyfit(x=r_ref[lin].wave_length[icf - 10:icf + 10],
                              y=reflectivity[icf - 10:icf + 10],
                              deg=1)
        approx.df.r_a[lin], approx.df.r_l[lin] = coef_fit/100.
        approx.df.reflectivity_peak_wavelength[
            lin], approx.df.reflectivity_peak[lin] = r_ref[lin].wave_length[
                reflectivity.argmax()], reflectivity.max()
        ax[axis_number].plot(
            r_ref[lin].wave_length[icf - 10:icf + 10],
            coef_fit[0] * r_ref[lin].wave_length[icf - 10:icf + 10] +
            coef_fit[1],
            lw=2, color=my_color[4])
        # label='{:2.2f}'.format(coef_fit[0]).replace(".",",") +
        # '$\\lambda+$' +
        # '{:2.2f}'.format(coef_fit[1]).replace(".",","))
        ax[axis_number].plot(r_ref[lin]['wave_length'],
                             reflectivity,
                             lw=.5,color=my_color[color_number],
                             label=leg[lin])


        ax[axis_number].plot(r_ref[lin].wave_length[icf],
                             reflectivity[icf],
                             '+',
                             ms=8,
                             color=my_color[-2])

        print(r_ref[lin].wave_length[reflectivity.argmax()], "\t",
              reflectivity.max())

        ax[axis_number].plot(r_ref[lin].wave_length[reflectivity.argmax()],
                             reflectivity.max(),
                             'x',
                             ms=8,
                             color=my_color[5])

    plotax(0, 0)
    plotax(0, 3,1)
    plotax(1, 1)
    plotax(1, 2,1)
    plotax(2, 4)
    plotax(2, 5,1)
    ax[0].legend(ncol=1)
    ax[1].legend(ncol=1)
    ax[2].legend(ncol=1)
    ax[0].set_ylim(bottom=0, top=100)
    ax[0].set_xlim(1539, 1552)
    fig.supxlabel(r'$\lambda, \si{\nm}$')
    fig.supylabel(r'Refletividade, \%')
    plt.savefig("../images/reflectivity_plots.pdf", format="pdf")
    approx.df.to_csv(
        "../experimentos/24042023/reflectivity_approximations_and_peaks.csv"
    )
    plt.close(fig=3)
    p = lambda lin: approx.df.r_a[lin] * _laser_peak + approx.df.r_l[lin]
    _str = ""
    for i in range(6):
        _str += r"\reflectivity^{" + leg[
            i] + r"}\left(\lambda\right) & \approx\num{" + str(
                approx.df['r_a'][i]) + r"}\lambda+\num{" + str(
                    approx.df['r_l'][i]) + r"} & \reflectivity^{" + leg[
                        i] + r"}\left(1549,3\right) & \approx" + '{:.0f}'.format(100.*p(i)) + r"\unit{\percent}\\ "
    print(_str)

reflectivity_plots()


def transmissivity_plots():
    # bragg.r0.max()
    # fig.clear()
    fig, ax = plt.subplots(1,
                           3,
                           num=3,
                           sharey=True,
                           sharex=True,
                           figsize=(FIG_L, FIG_A))

    class linearization_data():

        def __init__(self):
            self.df = pd.DataFrame()
            """Wavelength vector in nano memeter"""
            self.df['t_a'] = np.zeros(6, dtype=np.float32)
            """angular coefficient"""
            self.df['t_l'] = np.zeros(6, dtype=np.float32)
            """linear coefficient"""

    approx = linearization_data()

    def plotax(axis_number, lin):

        reflectivity = 100.0 - calc_reflectivity(r[lin]['power'], r_ref[lin]['power'])
        ##
        icf = np.where(r_ref[lin].wave_length >= 1549.3)[0][0]
        '''Index center of fitting'''
        coef_fit = np.polyfit(x=r_ref[lin].wave_length[icf - 10:icf + 10],
                              y=reflectivity[icf - 10:icf + 10],
                              deg=1)
        approx.df.t_a[lin], approx.df.t_l[lin] = coef_fit/100.
        ax[axis_number].plot(
            r_ref[lin].wave_length[icf - 10:icf + 10],
            coef_fit[0] * r_ref[lin].wave_length[icf - 10:icf + 10] +
            coef_fit[1],
            lw=2)
        # label='{:2.2f}'.format(coef_fit[0]).replace(".",",") +
        # '$\\lambda+$' +
        # '{:2.2f}'.format(coef_fit[1]).replace(".",","))

        ax[axis_number].plot(r_ref[lin]['wave_length'],
                             reflectivity,
                             lw=.5,
                             label=leg[lin])
        ax[axis_number].plot(r_ref[lin].wave_length[icf],
                             reflectivity[icf],
                             '+',
                             ms=8,
                             color=my_color[-2])
    for lin in [0, 3]:
        # ax[0].vlines(x=1549.3, ymin=0, ymax=40, color=my_color[-1])
        plotax(0, lin)

    for lin in [1, 2]:
        # ax[1].vlines(x=1549.3, ymin=0, ymax=40, color=my_color[-1])
        plotax(1, lin)
    for lin in [4, 5]:
        # ax[2].vlines(x=1549.3, ymin=0, ymax=40, color=my_color[-1])
        plotax(2, lin)

    ax[0].legend(ncol=1)
    ax[1].legend(ncol=1)
    ax[2].legend(ncol=1)
    # ax[0].set_ylim(bottom=0, top=100)
    ax[0].set_xlim(1539, 1552)
    fig.supxlabel(r'$\lambda, \si{\nm}$')
    fig.supylabel(r'Transmissão, \%')
    plt.savefig("../images/transmission_plots.pdf", format="pdf")
    approx.df.to_csv(
        "../../experimentos/24042023/transmission_approximations_and_peaks.csv"
    )
    plt.close(fig=3)
    p = lambda lin: approx.df.t_a[lin] * 1549.3 + approx.df.t_l[lin]
    _str = ""
    for i in range(6):
        _str += r"\transmittivity^{" + leg[
            i] + r"}\left(\lambda\right) & \approx\num{" + str(
                approx.df['t_a'][i]) + r"}\lambda \num{" + str(
                    approx.df['t_l'][i]
                ) + r"} & \transmittivity^{" + leg[
                    i] + r"}\left(1549,3\right) & \approx" + '{:.0f}'.format(
                        100.0*p(i)) + r"\unit{\percent}, \\ "
    print(_str)

def animation_to_tuning_model_of_fbg():
    """InteractiveShell Only """

    def animation(fbg_size=1.85e-4,
                  delta_n=3.15e-3,
                  peak=1546.25,
                  fbg_number=3):

        delta_l = 1000

        bragg = Bragg(fbg_size=fbg_size,
                      delta_n=delta_n,
                      wavelength_peak=peak,
                      delta_span_wavelength=delta_l,
                      diff_of_peak=5)

        reflectivity = bragg.calc_bragg(
            0, 1e-9 * r_ref[fbg_number]['wave_length'])
        plt.plot(r_ref[fbg_number]['wave_length'],
                 calc_reflectivity(r[fbg_number]['power'],
                                   r_ref[fbg_number]['power']),
                 label=leg[fbg_number])
        plt.plot(r_ref[fbg_number]['wave_length'], 100 * reflectivity)
        plt.title('FBG size: ' + "{:1.2e}".format(fbg_size) +
                  ', $\\Delta n$: ' + "{:1.2e}".format(delta_n) + ', Peak: ' +
                  str(peak) + "$\\si{\\nm}$")

    w = interactive(animation,
                    fbg_size=(1.5e-5, 1.5e-3, 1.e-5),
                    delta_n=(1e-5, 1e-2, 1.e-5),
                    peak=(1545, 1549, .25),
                    fbg_number=(0, 6, 1))

    display(w)


# ((1.0-1.45)/2.45)**2