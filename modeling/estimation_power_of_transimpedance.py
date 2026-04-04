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

folder_data = "data/19052023/"
source = np.loadtxt(folder_data + "source1.txt")
source_2 = pd.read_csv(
    "data/15052023/erbium_source_test/laser_bobina_1500mm_M12_051523__150433.csv"
)

x_minus_ref_1 = np.loadtxt(folder_data + "x_minus_ref_1.txt")
x_plus_ref_1 = np.loadtxt(folder_data + "x_plus_ref_1.txt")
y_minus_ref_1 = np.loadtxt(folder_data + "y_minus_ref_1.txt")
y_plus_ref_1 = np.loadtxt(folder_data + "y_plus_ref_1.txt")

dbm2mW = lambda dBm: 10.0**(dBm * 0.1)
lin2dBm = lambda lin: 10.0 * np.log10(lin)
normalize_spectrum = lambda d: np.array(
    [1e9 * d[1:, 0],
     dbm2mW(d[1:, 1] - lin2dBm(1e9 * np.diff(d[:, 0])))]).T
power_mW = lambda d: np.trapezoid(x=d[:, 0], y=1e3 * d[:, 1])

# Fix the data with resolution and convert data in dBm to mW

source_fixed = normalize_spectrum(source)
x_minus_ref_fixed = normalize_spectrum(x_minus_ref_1)
x_plus_ref_fixed = normalize_spectrum(x_plus_ref_1)
y_minus_ref_fixed = normalize_spectrum(y_minus_ref_1)
y_plus_ref_fixed = normalize_spectrum(y_plus_ref_1)

source_power_mW = power_mW(source_fixed)
x_minus_ref_power_mW = power_mW(x_minus_ref_fixed)
x_plus_ref_power_mW = power_mW(x_plus_ref_fixed)
y_minus_ref_power_mW = power_mW(y_minus_ref_fixed)
y_plus_ref_power_mW = power_mW(y_plus_ref_fixed)

fig, ax = plt.subplots(3, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
ax[0].plot(source_fixed[:, 0],
           source_fixed[:, 1],
           label='Fonte(' + '{:2.2f}'.format(source_power_mW) +
           r"\si{\micro\watt})")
ax[1].plot(x_minus_ref_fixed[:, 0],
           x_minus_ref_fixed[:, 1],
           label=r'$x_{-}($' + '{:2.2f}'.format(x_minus_ref_power_mW) +
           r"\si{\micro\watt})")
ax[1].plot(x_plus_ref_fixed[:, 0],
           x_plus_ref_fixed[:, 1],
           label=r'$x_{+}($' + '{:2.2f}'.format(x_plus_ref_power_mW) +
           r"\si{\micro\watt})")
ax[2].plot(y_minus_ref_fixed[:, 0],
           y_minus_ref_fixed[:, 1],
           label=r'$y_{-}($' + '{:2.2f}'.format(y_minus_ref_power_mW) +
           r"\si{\micro\watt})")
ax[2].plot(y_plus_ref_fixed[:, 0],
           y_plus_ref_fixed[:, 1],
           label=r'$y_{+}($' + '{:2.2f}'.format(y_plus_ref_power_mW) +
           r"\si{\micro\watt})")
ax[0].legend()
ax[1].legend()
ax[2].legend()
fig.supylabel(r"\si{\milli\watt}")
fig.supxlabel(r"$\lambda, \si{\nm}$")

fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A))
ax.plot(1e9 * source[1:, 0],
        source[1:, 1] - 10.0 * np.log10(1e9 * np.diff(source[:, 0])))

ax.plot(
    source_2.wavelength, source_2.power_dbm - 10.0 * np.log10(6.0) -
    10 * np.log10(source_2.actual_resolution[0]))
ax.set_xlim(1548, 1551)
