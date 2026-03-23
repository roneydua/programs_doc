#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   common_functions.py
@Time    :   2023/05/08 16:12:32
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import fixed, interactive
from matplotlib import ticker

from bragg.bragg import Bragg

FIG_L = 6.29
FIG_A = (90.0) / 25.4
plt.rcParams["figure.dpi"] = 144
plt.rcParams["figure.figsize"] = (FIG_L, FIG_A)


# [markdown]
# All function with interrogations

transmission_interrogation = lambda r, optical_source: (1.0 - r
                                                        ) * optical_source
"""Math model to interrogation with pure transmission"""
reflection_interrogation = lambda r, optical_source: 0.25 * r * optical_source
"""Math model to interrogation with pure reflection"""
reflection_reflection_interrogation = lambda r1, r2, optical_source: 1. / 16. * r1 * r2 * optical_source
"""Math model to interrogation with pure reflection of reflection"""
transmission_transmission_interrogation = lambda r1, r2, optical_source: (
    1.0 - r1) * (1.0 - r2) * optical_source
"""Math model to interrogation with pure transmission of transmission"""
reflection_transmission_interrogation = lambda r1, r2, optical_source: 0.25 * (
    1.0 - r1) * r2 * optical_source
"""Math model to interrogation with pure reflection of transmission"""
transmission_reflection_interrogation = lambda r1, r2, optical_source: 0.25 * (
    1. - r2) * r1 * optical_source
"""Math model to interrogation with pure transmission of reflection"""


def calc_total_pot(bragg1: Bragg,
                   deformation_vector: np.ndarray,
                   optical_source: np.ndarray,
                   interrogation_function,
                   bragg2: Bragg = None):
    pot = np.zeros(deformation_vector.size)
    for i in range(deformation_vector.size):
        ref = bragg1.calc_bragg(deformation_vector[i])
        if bragg2 != None:
            ref2 = bragg2.calc_bragg(-deformation_vector[i])
            pot[i] = np.trapz(x=bragg1.wavelength_span,
                              y=interrogation_function(ref, ref2,
                                                       optical_source))
        else:
            pot[i] = np.trapz(x=bragg1.wavelength_span,
                              y=interrogation_function(ref, optical_source))

    return pot


# def pot_vs_gravity(deformation_vector, y, plot_label, save_name):
#     # Plot reflection vs strain
#     poly_coef = np.polyfit(x=deformation_vector / 9.89, y=y * 1e6, deg=1)
#     fig, ax = plt.subplots(1, 1, num=1, figsize=(FIG_L, FIG_A * 0.75))
#     ax.set_title(r'$p(g)[\si{\micro\watt}]=$' +
#                  '{:2.4f}'.format(poly_coef[0]) + r"$\cdot g+$"
#                  '{:2.4f}'.format(poly_coef[1]))
#     ax.plot(deformation_vector / 9.89, y * 1e6, label=plot_label)
#     poly_fit = np.poly1d(poly_coef)

#     ax.plot(deformation_vector / 9.89, poly_fit(deformation_vector / 9.89))
#     ax.set_xlabel(r"Aceleração [g]")

#     ax.set_ylabel("\\si{\\micro\\watt}")
#     ax.legend()
#     plt.savefig(save_name, format="pdf")
#     plt.close(fig=1)

# def calc_pot_transmission(bragg: Bragg, deformation_vector, laser: np.ndarray):

#     pot_transmission = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         pot_transmission[i] = np.trapz(x=bragg.wavelength_span,
#                                        y=transmission_interrogation(
#                                            ref, laser))
#     return pot_transmission

# def calc_pot_reflection(bragg: Bragg, deformation_vector, laser: np.ndarray):
#     pot_reflection = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         pot_reflection[i] = np.trapz(x=bragg.wavelength_span,
#                                      y=reflection_interrogation(ref, laser))
#     return pot_reflection

# def calc_pot_reflection_reflection(bragg: Bragg, bragg2: Bragg,
#                                    deformation_vector, laser: np.ndarray):
#     pot_reflection_reflection = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         ref2 = bragg2.calc_bragg(-deformation_vector[i],
#                                  wavelength_vector=bragg.wavelength_span)
#         pot_reflection_reflection[i] = np.trapz(
#             x=bragg.wavelength_span,
#             y=reflection_reflection_interrogation(ref, ref2, laser))
#     return pot_reflection_reflection

# def calc_pot_transmission_transmission(bragg: Bragg, bragg2: Bragg,
#                                        deformation_vector, laser: np.ndarray):
#     pot_transmission_transmission = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         ref2 = bragg2.calc_bragg(-deformation_vector[i],
#                                  wavelength_vector=bragg.wavelength_span)
#         pot_transmission_transmission[i] = np.trapz(
#             x=bragg.wavelength_span,
#             y=transmission_transmission_interrogation(ref, ref2, laser))
#     return pot_transmission_transmission

# def calc_pot_reflection_transmission(bragg: Bragg, bragg2: Bragg,
#                                      deformation_vector, laser: np.ndarray):
#     pot_reflection_transmission = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         ref2 = bragg2.calc_bragg(-deformation_vector[i],
#                                  wavelength_vector=bragg.wavelength_span)
#         pot_reflection_transmission[i] = np.trapz(x=bragg.wavelength_span,
#                                                   y=0.25 * ref2 * (1. - ref) *
#                                                   laser)
#     return pot_reflection_transmission

# def calc_pot_transmission_reflection(bragg: Bragg, bragg2: Bragg,
#                                      deformation_vector, laser: np.ndarray):
#     pot_transmission_reflection = np.zeros(deformation_vector.size)
#     for i in range(deformation_vector.size):
#         ref = bragg.calc_bragg(deformation_vector[i])
#         ref2 = bragg2.calc_bragg(-deformation_vector[i],
#                                  wavelength_vector=bragg.wavelength_span)
#         pot_transmission_reflection[i] = np.trapz(
#             x=bragg.wavelength_span,
#             y=transmission_reflection_interrogation(ref, ref2, laser))
#     return pot_transmission_reflection
