#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   live_production.py
@Time    :   2023/10/04 19:58:32
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

import numpy as np
import locale
import matplotlib.pyplot as plt
from time import sleep
# from acquisitions.Q8347 import Q8347
from acquisitions.AQ6370D import AQ6370D
from common_functions.generic_functions import reflectivity_transmition
from datetime import datetime
import h5py

# InteractiveShell.ast_node_interactivity = "all"
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
# plt.style.use("default")
plt.style.use("common_functions/roney3.mplstyle")
plt.rcParams["figure.dpi"] = 100
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = (90.0) / 25.4
SAVE_DATA = True
osa = AQ6370D(center=1551.5e-9, span=10e-9)

osa.read(trace='TRB')
osa.osa.query("TRACE:ACTIVE?")

osa.osa.write(":TRACE:ACTIVE TRA")
osa.simple_sweep('TRA')

osa.osa.write(":TRACE:ACTIVE TRB")
osa.simple_sweep("TRB")


wavelength_m = osa.wavelength_m
y = osa.optical_power_dbm

fig, ax = plt.subplots(1, 1, num=1, sharex=True, figsize=(FIG_L, 2 * FIG_A), dpi=100)
fig.supxlabel(r"$\lambda, [\unit{\nm}]$")
fig.supylabel(r"Potência óptica $[\si{\dbm\nm}]$")
ax.plot(wavelength_m * 1e9, y)    
plt.show()
osa.close()


if SAVE_DATA:
    print("save data on hdf file")
    f = h5py.File("acquisitions/fbg_acc_5_montado.hdf5", "a")
    now = datetime.now()
    # ff = f.require_group("reflection_with_seismic_mass_locked")
    # check number of fbg on group
    # number_of_dataset = len(ff.keys())
    # fff = ff.require_group("two_cleaved")
    fff = f.require_group("doped_fbg/fbg14")
    # fff.attrs["angle_cleaved_deg"] = 30
    # fff.attrs["note"] = "circulator"
    # fff.attrs["note"] = "coupler"
    fff.require_dataset("wavelength_m", wavelength_m.shape, dtype=np.float32)
    fff.require_dataset("optical_power_dbm", y.shape, dtype=np.float32)
    fff["wavelength_m"][...] = wavelength_m
    fff["optical_power_dbm"][...] = y
    f.close()
    # plt.savefig(
    #     "fbg_production"
    #     + "/fig_production_figs/"
    #     + now.strftime(r"%Y%m%d")
    #     + "/fbg"
    #     + str(1 + number_of_dataset)
    #     + ".png",
    #     format="png",
    #     transparent=False,
    # )
    # plt.close()

print("ok")
