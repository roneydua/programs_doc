#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   acquisitionAnritsu.py
@Time    :   2023/03/02 17:19:25
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

from pyvisa import ResourceManager
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymeasure.instruments.anritsu import AnritsuMS9710C
from auxiliaryClasses import Dynamometer
import pandas as pd
from pathlib import Path
mpl.rcParams['figure.dpi'] = 72
plt.style.use("../../../../programasComuns/roney3.mplstyle")

# for save data
from datetime import datetime
ilx_control = True
mike_control =False
dynamometer_control = False
# H - hour, M- minute, S - second
folder_save = './data/15052023/erbium_source_test/'
# letter = 'f'
# teste_save_name = letter+"_reflection_"+"4_percent_"
# teste_save_name = letter+"_reflection_4_percent_"
teste_save_name = 'laser_pump_1500mm_M12_'


def dbm2W(power):
    return 10.0**(power * 0.1)

class experiment_data_save():

    def __init__(self, size: int):
        self.df = pd.DataFrame()
        """Data frame with experiment data"""
        self.df['wavelength'] = 1.0 + np.zeros(size, dtype=np.float32)
        """Wavelength vector in nano memeter"""
        self.df['power_dbm'] = 2.0 + np.zeros(size, dtype=np.float16)
        """Power vector in dBm * actual_resolution. WARNING: ned correction with 'actual resolution'"""
        #Variable necessary for the correction of the power value in decibels
        self.df['actual_resolution'] = '0.05'
        self.df['ilx_current'] = 0.0
        self.df['resolution_nm'] = 1.0
        self.df['resolution_vbw'] = '100Hz'
        self.df['erbium_fiber_size'] = 21


    def save(self, name):
        # self.df['power_dbm'] -= 10.0 * np.log10(self.df['actual_resolution'])
        self.df.to_csv(name, index=True)


def save_temporary_graphic(da, graphic_name:str):
    fig, ax2 = plt.subplots(1, 1, num='temp',dpi=36)
    ax2.plot(da.df['wavelength'], da.df['power_dbm'])
    ax2.set_xlabel(r"$\lambda, \si{\nm}$")
    ax2.set_ylabel('dBm')
    plt.savefig(graphic_name, format="pdf")
    plt.close(fig='temp')

def test_setup():
    osa = AnritsuMS9710C("GPIB0::8::INSTR", timeout=15000)
    osa.clear()
    osa.write("DATE " + str(datetime.now().year)[2:] + ",0" +
              str(datetime.now().month) + "," + str(datetime.now().day))
    osa.write("time " + str(datetime.now().hour) + "," +
              str(datetime.now().minute))
    osa.write("SMT OFF")
    # osa.write("STA 1510")
    osa.write("STA 935")
    # osa.write("STO 1600")
    osa.write("STO 1025")
    # osa.wavelength_span = 90.0
    # osa.write("CNT 1549.0")
    osa.resolution_vbw = '1kHz'
    # osa.write("STA 1525")

    osa.write("MPT 2001")
    # osa.write('SSI')
    # get current resolution
    osa.write('ARES ON')  # osa.write("SRT")
    # osa.clear()
    osa.write(r'ARED?')
    actual_resolution = float(osa.read())
    ## ILX setup
    rm = ResourceManager()
    ilx = rm.open_resource('GPIB0::1::INSTR')
    # put TEC on temperature mode
    ilx.write("TEC:MODE:T<NL>") # type: ignore
    # enable tec
    ilx.write("TEC:OUT 1<NL>") # type: ignore
    # enable laser output
    ilx.write("LAS:OUT 1") # type: ignore
    # ilx.write("LAS:OUT 0")
    # set ILX current

    return osa, ilx

def make_test(ilx, osa, _ilx_current:float):

    ilx_current = '{:2.1f}'.format(_ilx_current)
    ilx.write("LAS:LDI " + ilx_current)
    # teste_save_name = "TEAP_"

    osa.single_sweep(n=100)
    # update time to save data
    now = datetime.now()
    current_time = now.strftime("%m%d%y__%H%M%S")
    # da = dataAcquisition(n=osa.sampling_points, test_name="TEAP")
    test_name = Path(folder_save + teste_save_name + current_time+ ".csv")
    # da = dataAcquisition(n=osa.sampling_points) # type: ignore
    sampling_points = int(osa.sampling_points) # type: ignore
    da = experiment_data_save(size=sampling_points)
    ## update Actual resolution, resolution and other variables
    osa.write(r'ARED?')
    actual_resolution = float(osa.read())
    da.df['actual_resolution'] = actual_resolution
    da.df['ilx_current'] = ilx_current
    resolution = '.05'
    osa.write("RES " + resolution)
    da.df['resolution_nm'] = float(resolution)
    da.df['resolution_vbw'] = osa.resolution_vbw
    # osa.wait_for_sweep()
    da.df['wavelength'], da.df['power_dbm'] = osa.read_memory(slot='A')

    da.save(test_name)
    save_temporary_graphic(
        da, folder_save + teste_save_name + current_time +
        ".pdf")
    plt.plot(da.df['wavelength'], da.df['power_dbm'], label=teste_save_name)
    plt.legend()
    # osa.write("SRT")



if __name__ == "__main__":
    osa, ilx = test_setup()
    ilx_current_values = np.arange(100.,1200.,50.)
    for i in ilx_current_values:
        print("make test with "+str(i)+" mA")
        make_test(ilx, osa, _ilx_current=i)
    ## Put current on 100mA
    ilx_current = '{:2.1f}'.format(100)
    ilx.write("LAS:LDI " + ilx_current) # type: ignore





# da = pd.read_csv(Path('/home/pegasus/Dropbox/doutorado/experimentos/25042023/laser_1549042523__131735.csv'))
# plt.plot(da['wavelength'],da['power_dbm'])
# plt.xlim([1548.5,1550])
# plt.ylim(bottom=-60,top=8)
# plt.savefig('laser_1594.jpg',format='jpg')


# test_name1 = Path("../../../../experimentos/20042023/a_reflection042423__130528.csv")
# test_name2 = Path("../../../../experimentos/20042023/a_reflection_4_percent_042423__130752.csv")

# da1 = pd.read_csv(test_name1)
# da2 = pd.read_csv(test_name2)

# plt.plot(da1['wavelength'].to_numpy(),da1['power_dbm'].to_numpy())
# plt.plot(da2['wavelength'].to_numpy(),da2['power_dbm'].to_numpy())

# plt.plot(da1['wavelength'],
#          (dbm2W(da2['power_dbm'])- dbm2W(da1['power_dbm']) ) / dbm2W(da2['power_dbm']))
