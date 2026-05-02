import h5py
import pandas as pd
from common_functions.QuatSymbolic import QuaternionSymbolic
from common_functions.generic_functions import *
import matplotlib.pyplot as plt




folder = 'aquisitions/dataTemp/'
source_2 = pd.read_csv(folder+'source.txt', delimiter='\\s', engine='python',names=['wavelength','power_dbm'])
fbg_2 = pd.read_csv(folder+'fbg_2.txt', delimiter='\\s',
                    engine='python', names=['wavelength', 'power_dbm'])
fbg_5 = pd.read_csv(folder+'fbg_5.txt', delimiter='\\s',
                    engine='python', names=['wavelength', 'power_dbm'])


min_w = 1510e-9
max_w = 1530e-9
r_fbg_2 = calc_reflectivity_by_transmission(source_2.wavelength,source_2.power_dbm,fbg_2.power_dbm,True,min_wavelength=min_w,max_wavelength=max_w)

r_fbg_5 = calc_reflectivity_by_transmission(source_2.wavelength,source_2.power_dbm,fbg_5.power_dbm,True,min_wavelength=min_w,max_wavelength=max_w)


f = h5py.File('./../data/phd_data.hdf5',mode='a')
ff = f.require_group('fbg_production/test2/fbg_2/after_dehydrogenation/')
ff['wavelength'] = fbg_2.wavelength
ff['power_dbm'] = fbg_2.power_dbm
ff['r'] = r_fbg_2

ff = f.require_group('fbg_production/test2/fbg_5/after_dehydrogenation/')
ff['wavelength'] = fbg_5.wavelength
ff['power_dbm'] = fbg_5.power_dbm
ff['r'] = r_fbg_5
f.close()

plt.plot(source_2.wavelength, r_fbg_5)
plt.plot(source_2.wavelength, r_fbg_2)
plt.xlim(1540e-9,1560e-9)
plt.ylim(0,1)

# plt.plot(source_2.wavelength, fbg_2.power_dbm)
# plt.plot(source_2.wavelength, source_2.power_dbm)


# index_min, index_max = find_index_of_x_span(1510e-9, 1545e-9, source_2.wavelength)
# bias_source = (source_2.power_dbm[index_min:index_max]-fbg_2.power_dbm[index_min:index_max]).mean()

# plt.plot(source_2.wavelength, source_2.power_dbm-bias_source)


q = QuaternionSymbolic()
q.quat @ q.quat.T


