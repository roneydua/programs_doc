#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   article_graphics.py
@Time    :   2025/08/16 15:32:01
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""


from calibration_acc_methods.acc_calibration_4 import (
    calibration_with_temperature_dependency,
    plot_allan_deviation_20240814,
    linearity_analisys_20240828
)

# from calibration_acc_methods.acc_calibration_4 import *
from fbg_production.fbg_graphics_production import plot_graphics_with_pairs_acc_4
from interrogation_analysis.push_pull_interrogation_analysis import (
    acc_4_analysis,
    linearity_analisys_acc_4,
)
from reflectivity_analisys.graphics_optical_mechanical_identification import (
    plot_time_elapsed_fbg_production,
)

if __name__ == "__main__":
    # figure pairs of fbg:
    # plot_graphics_with_pairs_acc_4("en")
    # plot_time_elapsed_fbg_production("en")
    # plot_graphics_with_pairs_acc_4("en")
    acc_4_analysis(language="en")#acc_4_power_vs_accel_en
    linearity_analisys_acc_4("en")
    plot_allan_deviation_20240814("en")#data_allan_deviation_acc_4_20240814
    calibration_with_temperature_dependency(
        "en", save_calibrated_data=False
    )  # uncalibrated_data_en.pdf
    linearity_analisys_20240828("en")
