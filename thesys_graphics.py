#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   thesys_graphics.py
@Time    :   2025/08/16 15:28:23
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import argparse

import bragg.optical_simulation
from interrogation_analysis.interrogation_analysis import (
    interrogation_laser_one_fbg,
    interrogation_two_fbgs_reflection_of_reflection,
)
from interrogation_analysis.push_pull_interrogation_analysis import (
    power_vs_delta_lambda_animation_one_fiber,
    power_vs_delta_lambda_animation_two_fibers,
    acc_4_analysis,
)
from modeling.inverse_problem_solution import plot_inertial_states_difference
from reflectivity_analisys.graphics_optical_mechanical_identification import (
    identification_method_examples,
)
from reflectivity_analysis.reflectivity_estimation_error import plot_error_comparision
from fbg_production.fbg_graphics_production import plot_graphics_with_pairs_acc_4

from reflectivity_analysis.graphics_optical_mechanical_identification import plot_time_elapsed_fbg_production


from calibration_acc_methods.acc_calibration_4 import plot_allan_deviation_20240814, calibration_with_temperature_dependency

def introduction_chapter():
    """Cap. Introducao"""
    bragg.optical_simulation.plot_drawFig6_spectres()


def interrogation_chapter():
    """Cap. interrogacao"""
    bragg.optical_simulation.plot_bragg_spectrum()
    interrogation_laser_one_fbg()
    power_vs_delta_lambda_animation_two_fibers()
    power_vs_delta_lambda_animation_one_fiber()
    interrogation_two_fbgs_reflection_of_reflection()


def reflectivity_chapter():
    "sec analise de refletividade"
    plot_error_comparision()
    identification_method_examples()


def result_chaper():
    "Cap. Resultados"
    # plot_inertial_states()
    # plot_deformations()
    plot_inertial_states_difference()
    plot_graphics_with_pairs_acc_4('pt')
    acc_4_analysis(language="pt")
    plot_allan_deviation_20240814('pt')
    calibration_with_temperature_dependency('pt')
def appends():
    plot_time_elapsed_fbg_production('pt')


if __name__ == "__main__":
    chapter_to_update = argparse.ArgumentParser()
    chapter_to_update.add_argument("--chapter", nargs="+")
    chapter_choices = chapter_to_update.parse_args()
    if chapter_choices.chapter:
        for name in chapter_choices.chapter:
            if name == "all":
                introduction_chapter()
                interrogation_chapter()
                reflectivity_chapter()
                result_chaper()
                break
            elif name == "introduction":
                introduction_chapter()
            elif name == "results":
                result_chaper()
            elif name == "interrogation":
                interrogation_chapter()
            elif name == "reflectivity":
                reflectivity_chapter()
            elif name == "appends":
                appends()
