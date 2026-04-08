#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   thesys_graphics.py
@Time    :   2025/08/16 15:28:23
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import argparse

from bragg.optical_simulation import (plot_bragg_spectrum,
                                      plot_drawFig6_spectres)
from calibration_acc_methods.acc_calibration_4 import (
    calibration_with_temperature_dependency, linearity_analysis_20240828,
    plot_allan_deviation_20240814)
from fbg_production.fbg_graphics_production import \
    plot_graphics_with_pairs_acc_4
from interrogation_analysis.interrogation_analysis import (
    interrogation_laser_one_fbg,
    interrogation_two_fbgs_reflection_of_reflection)
from interrogation_analysis.push_pull_interrogation_analysis import (
    acc_4_analysis, linearity_analysis_acc_4,
    power_vs_delta_lambda_animation_one_fiber,
    power_vs_delta_lambda_animation_two_fibers)
from modeling.graphics_new_approach import plot_graphics
from modeling.otimization_of_model import \
    plot_otimizacao_parametrica_frequencia
# from modeling.inverse_problem_solution import plot_inertial_states_difference
from reflectivity_analysis.graphics_optical_mechanical_identification import (
    identification_method_examples, plot_time_elapsed_fbg_production)
from reflectivity_analysis.reflectivity_estimation_error import (
    plot_error_comparision, plot_error_comparision_delta_r)


def interrogation_chapter():
    """Cap. interrogacao"""
    plot_drawFig6_spectres()
    plot_bragg_spectrum()  # figure plot_bragg_spectrum.pdf
    interrogation_laser_one_fbg()
    power_vs_delta_lambda_animation_two_fibers()
    power_vs_delta_lambda_animation_one_fiber()
    interrogation_two_fbgs_reflection_of_reflection()


def reflectivity_chapter():
    "sec analise de refletividade"
    plot_error_comparision()
    identification_method_examples()
    plot_error_comparision_delta_r()


def result_chaper():
    "Cap. Resultados"
    # linearity_analysis_acc_4("pt")
    # plot_graphics_with_pairs_acc_4('pt')
    # acc_4_analysis(language="pt")
    # plot_allan_deviation_20240814('pt')
    # calibration_with_temperature_dependency("pt", save_calibrated_data=False)
    linearity_analysis_20240828("pt")  # linearity_with_respect_to_full_scale
    # plot_graphics("sinusoidal_with_temp_perturbation")
    # plot_otimizacao_parametrica_frequencia()
def appends():
    plot_time_elapsed_fbg_production('pt')


if __name__ == "__main__":
    chapter_to_update = argparse.ArgumentParser()
    chapter_to_update.add_argument("--chapter", nargs="+")
    chapter_choices = chapter_to_update.parse_args()
    if chapter_choices.chapter:
        for name in chapter_choices.chapter:
            if name == "all":
                interrogation_chapter()
                result_chaper()
                reflectivity_chapter()
                break
            elif name == "results":
                result_chaper()
            elif name == "interrogation":
                interrogation_chapter()
            elif name == "reflectivity":
                reflectivity_chapter()
            elif name == "appends":
                appends()
