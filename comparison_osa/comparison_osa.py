#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   comparison_osa.py
@Time    :   2023/08/28 14:12:15
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
import h5py
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

from common_functions.generic_functions import *

FIG_L = 6.29
FIG_A = (90.0) / 25.4
# End of header


f = h5py.File('phd_data.hdf5')
ff = f['osa_comparison']
advantest_1_hr = ff['advantest01/high_resolution']
advantest_1_lr = ff['advantest01/low_resolution']
advantest_1_hr = ff['advantest02/high_resolution']
advantest_1_lr = ff['advantest02/low_resolution']
anritsu = ff['anritsu']


fig.clear()
fig, ax = plt.subplots(1, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
ax.plot(1e9*advantest_1_hr['wavelength'][...],
        advantest_1_hr['power_dbm'][...], label='Advantest 1 (HR)')
ax.plot(1e9*advantest_1_lr['wavelength'][...], advantest_1_lr['power_dbm'][...], ':', color=my_colors[0], label='Advantest 1 (LR)')
ax.plot(1e9*advantest_2_hr['wavelength'][...],
        advantest_2_hr['power_dbm'][...], label='Advantest 2 (HR)')
ax.plot(1e9*advantest_2_lr['wavelength'][...], advantest_2_lr['power_dbm'][...],
        ':', color=my_colors[1], label='Advantest 2 (LR)')

ax.plot(anritsu['wavelength'][...],
        anritsu['power_dbm'][...], label='Anritsu')
ax.set_ylabel('Potência $[\\si{\\dbm}]$')
ax.set_xlabel(' $\\lambda [\\si{\\m}]$')
ax.legend()


# f = h5py.File('../data/phd_data.hdf5','a')
# for k in range(1,6):
#     print(k)
#     ff = f['fbg_production/test2/fbg_'+str(k)]
#     r = np.zeros(ff['optical_power'].shape)
#     for i in range(r.shape[1]):
#         r[:,i]=calc_reflectivity_by_transmission(ff['optical_power'][:,0],ff['optical_power'][:,i],None)
#         # plt.plot(ff['wavelength'][:],r[:,i])
#     ff['r'] = r
# f.close()