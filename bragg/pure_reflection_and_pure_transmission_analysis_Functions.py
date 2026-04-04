#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pure_reflection_and_pure_transmission_analysis_Functions.py
@Time    :   2023/05/09 09:34:38
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
'''

import locale

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import fixed, interactive
from matplotlib import ticker
from common_functions.generic_functions import *
from interrogation_analysis.common_functions_bragg_study import *

# InteractiveShell.ast_node_interactivity = "all"
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.style.use("default")
from bragg.bragg import Bragg

plt.style.use("./common_functions/roney3.mplstyle")
FIG_L = 6.29
FIG_A = 1.5 * (90.0) / 25.4
TESE_FOLDER = "../tese/images/not_used_on_thesis/"

class InterrogationClass(object):
    """docstring for Transmition."""

    def __init__(self,
                 bragg1: Bragg,
                 optical_source: np.ndarray,
                 deformation_vector: np.ndarray,
                 interrogation_function,
                 optical_source_label: str = "Laser",
                 bragg2: Bragg = None,
                 interrogation_1_reflection=True,
                 interrogation_2_reflection: bool or None = False,
                 ):
        '''
        __init__ Interrrogation class for all simulations.
        This class 
        Args:
            bragg1: Instance of FBG class
            optical_source: The vector with optical source spectrum
            deformation_vector: Deformation vector for simulation
            interrogation_function: The function on common_functions.py
            optical_source_label: To change graphic legend. Defaults to "Laser".
            bragg2: Instance of second FBG. Defaults to None.
            interrogation_1_reflection: The configuration of first FBG, for Transmission configuration set to False. Defaults to True.
            interrogation_2_reflection: The configuration of second FBG, for Transmission configuration set to False. Defaults to False.
        '''
        self.bragg1 = bragg1
        """ BRAGG instance of FBG1"""
        self.bragg2 = bragg2
        """ BRAGG instance of FBG1"""
        self.optical_source = optical_source
        self.optical_source_label = optical_source_label
        self.interrogation_function = interrogation_function
        """A lambda function that with interrogation math model."""
        self.deformation_vector = deformation_vector
        """Deformation vector with possible values for test"""
        self.interrogation_1_reflection = interrogation_1_reflection
        """FBG configuration.Reflection or transmission."""
        self.interrogation_2_reflection = interrogation_2_reflection
        """FBG configuration.Reflection, transmission or None."""
        self.pot_photodetector = calc_total_pot(bragg1,
                                                self.deformation_vector,
                                                self.optical_source,
                                                self.interrogation_function,
                                                self.bragg2)
        """Vector of pot_photodetector with respect of @self.deformation_vector"""

    def interrogation(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[0].set_ylabel(r'Reflectivity')
        ax[0].set_xlabel('$\\lambda, \\si{\\nm}$')
        ax[0].set_xlim((1548, 1550))
        ax[0].set_ylim((0, 1.0))

        ax[1].set_ylabel(r'\si{\micro\watt}')
        ax[1].set_xlabel(r'Deformation, \si{\micro\varepsilon}')

        ref1 = self.bragg1.calc_bragg(deformation=deformation)
        # Check if FBG 1 is as a reflection or transmission.
        if self.interrogation_1_reflection == True:
            y1 = self.bragg1.r0
            y1_ref = ref1
        else:
            y1 = 1.0 - self.bragg1.r0
            y1_ref = 1.0 - ref1

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   y1,
                   label="Null deformation of FBG 1",
                   color=my_colors[0],
                   ls=':')
        ax[0].plot(self.bragg1.wavelength_span_nm,
                   y1_ref,
                   label=r"$\varepsilon=$" + "{:2.2e}".format(deformation),
                   color=my_colors[0])
        if self.interrogation_2_reflection != None:
            ref2 = self.bragg2.calc_bragg(deformation=-deformation)
            if self.interrogation_2_reflection == True:
                y2 = self.bragg2.r0
                y2_ref = ref2
            else:
                y2 = 1.0 - self.bragg2.r0
                y2_ref = 1.0 - ref2
            ax[0].plot(self.bragg1.wavelength_span_nm,
                       y2,
                       label="Null deformation of FBG 2",
                       color=my_colors[1],
                       ls=':')
            ax[0].plot(self.bragg1.wavelength_span_nm,
                       y2_ref,
                       label=r"$\varepsilon=$" +
                       "{:2.2e}".format(-deformation),
                       color=my_colors[1])
            photodetector_power = self.interrogation_function(
                ref1, ref2, self.optical_source)
        else:
            photodetector_power = self.interrogation_function(
                ref1, self.optical_source)

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.optical_source / self.optical_source.max(),
                   label=self.optical_source_label)

        total_protodetector_power = np.trapezoid(x=self.bragg1.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.deformation_vector * 1e6, 1e6 * self.pot_photodetector)

        ax[1].plot(deformation * 1e6,
                   total_protodetector_power * 1e6,
                   'x',
                   ms=10,
                   label="Photodetector power= " +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")

        ax[0].legend()
        ax[1].legend()


class Transmition(object):
    """docstring for Transmition."""

    def __init__(self, bragg: Bragg, laser, deformation_vector: np.ndarray):
        self.bragg = bragg
        self.laser = laser

        self.deformation_vector = deformation_vector
        self.pot_transmission = calc_pot_transmission(self.bragg,
                                                      self.deformation_vector,
                                                      self.laser)

    def transmission(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[0].set_ylabel(r'Refletividade')
        ax[0].set_xlabel('$\\lambda, \\si{\\nm}$')
        ax[0].set_ylim((0, 1.0))

        ax[1].set_ylabel(r'\si{\micro\watt}')
        ax[1].set_xlabel(r'Deformação, \si{\micro\varepsilon}')
        # ax[1].set_xlim((1548, 1550))

        ax[0].plot(self.bragg.wavelength_span_nm, (1.0 - self.bragg.r0),
                   label="Deformação nula")
        ref = self.bragg.calc_bragg(deformation=deformation)

        ax[0].plot(self.bragg.wavelength_span_nm, (1. - ref),
                   label=r"$\varepsilon=$" + "{:2.2e}".format(deformation))
        ax[0].plot(self.bragg.wavelength_span_nm,
                   self.laser / self.laser.max(),
                   label="Laser")

        photodetector_power = transmission_interrogation(ref,self.laser)
        total_protodetector_power = np.trapezoid(x=self.bragg.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.deformation_vector * 1e6,
                   1e6 * self.pot_transmission,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")

        ax[1].plot(deformation * 1e6,
                   total_protodetector_power * 1e6,
                   'x',
                   ms=10)

        # ax[1].set_yticks([0, 1e-3 * self.pot_transmission.max()])
        ax[0].legend()
        ax[1].legend()

class Reflection(object):
    """docstring for Transmition."""

    def __init__(self, bragg: Bragg, laser):
        self.bragg = bragg
        self.laser = laser

    def reflection(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[0].set_ylabel(r'Refletividade')
        ax[1].set_ylabel(r'\si{\milli\watt\per\meter}')
        ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
        ax[1].set_xlim((1547, 1551))
        ax[0].set_ylim((0, self.bragg.r0.max()))
        ax[0].plot(self.bragg.wavelength_span_nm,
                   self.bragg.r0,
                   label="Deformação nula")
        ref = self.bragg.calc_bragg(deformation=deformation)

        photodetector_power = 0.25 * (ref) * self.laser
        total_protodetector_power = np.trapezoid(x=self.bragg.wavelength_span,
                                             y=photodetector_power)
        ax[0].plot(self.bragg.wavelength_span_nm,
                   ref,
                   label=r"$\varepsilon=$" + "{:2.2e}".format(deformation))
        ax[0].plot(self.bragg.wavelength_span_nm,
                   self.laser / self.laser.max() * self.bragg.r0.max(),
                   label="Laser")
        ax[1].plot(self.bragg.wavelength_span_nm,
                   1e-3 * photodetector_power,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")
        ax[1].set_yticks([0, 1e-3 * photodetector_power.max()])
        ax[0].legend()
        ax[1].legend()

class ReflectionReflection(object):
    """docstring for Transmition."""

    def __init__(self, bragg1: Bragg, bragg2: Bragg, laser: np.ndarray):
        self.bragg1 = bragg1
        self.bragg2 = bragg2
        self.laser = laser

    def reflection_reflection(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[0].set_ylabel(r'Refletividade')
        ax[1].set_ylabel(r'\si{\milli\watt\per\meter}')
        ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
        ax[0].set_ylim((0, self.bragg1.r0.max()))
        ax[1].set_xlim((1547, 1551))
        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.bragg1.r0,
                   color=my_colors[0],
                   ls=':',
                   label="Deformação nula FBG 1")
        ax[0].plot(self.bragg2.wavelength_span_nm,
                   self.bragg2.r0,
                   color=my_colors[1],
                   ls=':',
                   label="Deformação nula FBG 2")
        ref1 = self.bragg1.calc_bragg(deformation=deformation)
        ref2 = self.bragg2.calc_bragg(
            deformation=-deformation,
            wavelength_vector=self.bragg1.wavelength_span)

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref1,
                   color=my_colors[0],
                   label=r"$\varepsilon_1=$" + "{:2.2e}".format(deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref2,
                   color=my_colors[1],
                   label=r"$\varepsilon_2=$" + "{:2.2e}".format(-deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.laser / self.laser.max() * self.bragg1.r0.max(),
                   label="Laser")
        photodetector_power = 1.0 / 16.0 * (ref1 * ref2) * self.laser
        total_protodetector_power = np.trapezoid(x=self.bragg1.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.bragg1.wavelength_span_nm,
                   1e-3 * photodetector_power,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")

        ax[1].set_yticks([0, 1e-3 * photodetector_power.max()])
        ax[0].legend()
        ax[1].legend()

class TransmissionTransmission(object):
    """docstring for Transmition."""

    def __init__(self, bragg1: Bragg, bragg2: Bragg, laser: np.ndarray):
        self.bragg1 = bragg1
        self.bragg2 = bragg2
        self.laser = laser

    def transmission_transmission(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax[0].set_ylabel(r'Refletividade')
        ax[1].set_ylabel(r'\si{\milli\watt\per\meter}')
        ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
        ax[0].set_ylim((0, self.bragg1.r0.max()))
        ax[1].set_xlim((1547, 1551))
        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.bragg1.r0,
                   color=my_colors[0],
                   ls=':',
                   label="Deformação nula FBG 1")
        ax[0].plot(self.bragg2.wavelength_span_nm,
                   self.bragg2.r0,
                   color=my_colors[1],
                   ls=':',
                   label="Deformação nula FBG 2")
        ref1 = self.bragg1.calc_bragg(deformation=deformation)
        ref2 = self.bragg2.calc_bragg(
            deformation=-deformation,
            wavelength_vector=self.bragg1.wavelength_span)

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref1,
                   color=my_colors[0],
                   label=r"$\varepsilon_1=$" + "{:2.2e}".format(deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref2,
                   color=my_colors[1],
                   label=r"$\varepsilon_2=$" + "{:2.2e}".format(-deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.laser / self.laser.max() * self.bragg1.r0.max(),
                   label="Laser")
        ax[0].legend()
        photodetector_power = (1.0 - ref1) * (1.0 - ref2) * self.laser
        total_protodetector_power = np.trapezoid(x=self.bragg1.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.bragg1.wavelength_span_nm,
                   1e-3 * photodetector_power,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")
        # ax[1].set_ylim([0, photodetector_power.max()],)
        ax[1].set_yticks([0, 1e-3 * photodetector_power.max()])
        ax[1].legend()

class ReflectionTransmission(object):
    """docstring for Transmition."""

    def __init__(self, bragg1: Bragg, bragg2: Bragg, laser: np.ndarray):
        self.bragg1 = bragg1
        self.bragg2 = bragg2
        self.laser = laser

    def reflection_transmission(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax[0].set_ylabel(r'Refletividade')
        ax[1].set_ylabel(r'\si{\milli\watt\per\meter}')
        ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
        ax[0].set_ylim((0, self.bragg1.r0.max()))
        ax[1].set_xlim((1547, 1551))
        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.bragg1.r0,
                   color=my_colors[0],
                   ls=':',
                   label="Deformação nula FBG 1")
        ax[0].plot(self.bragg2.wavelength_span_nm,
                   self.bragg2.r0,
                   color=my_colors[1],
                   ls=':',
                   label="Deformação nula FBG 2")
        ref1 = self.bragg1.calc_bragg(deformation=deformation)
        ref2 = self.bragg2.calc_bragg(
            deformation=-deformation,
            wavelength_vector=self.bragg1.wavelength_span)

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref1,
                   color=my_colors[0],
                   label=r"$\varepsilon_1=$" + "{:2.2e}".format(deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref2,
                   color=my_colors[1],
                   label=r"$\varepsilon_2=$" + "{:2.2e}".format(-deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.laser / self.laser.max() * self.bragg1.r0.max(),
                   label="Laser")
        ax[0].legend()

        photodetector_power = 0.25 * ref2 * (1.0 - ref1) * self.laser
        total_protodetector_power = np.trapezoid(x=self.bragg1.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.bragg1.wavelength_span_nm,
                   1e-3 * photodetector_power,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")
        # ax[1].set_ylim([0, photodetector_power.max()],)
        ax[1].set_yticks([0, 1e-3 * photodetector_power.max()])
        ax[1].legend()

class TransmissionReflection(object):
    """docstring for Transmition."""

    def __init__(self, bragg1: Bragg, bragg2: Bragg, laser: np.ndarray):
        self.bragg1 = bragg1
        self.bragg2 = bragg2
        self.laser = laser

    def transmission_reflection(self, deformation=0.0):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(FIG_L, FIG_A))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax[0].set_ylabel(r'Refletividade')
        ax[1].set_ylabel(r'\si{\milli\watt\per\meter}')
        ax[1].set_xlabel(r'$\lambda, \si{\nm}$')
        ax[0].set_ylim((0, self.bragg1.r0.max()))
        ax[1].set_xlim((1547, 1551))
        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.bragg1.r0,
                   color=my_colors[0],
                   ls=':',
                   label="Deformação nula FBG 1")
        ax[0].plot(self.bragg2.wavelength_span_nm,
                   self.bragg2.r0,
                   color=my_colors[1],
                   ls=':',
                   label="Deformação nula FBG 2")
        ref1 = self.bragg1.calc_bragg(deformation=deformation)
        ref2 = self.bragg2.calc_bragg(
            deformation=-deformation,
            wavelength_vector=self.bragg1.wavelength_span)

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref1,
                   color=my_colors[0],
                   label=r"$\varepsilon_1=$" + "{:2.2e}".format(deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   ref2,
                   color=my_colors[1],
                   label=r"$\varepsilon_2=$" + "{:2.2e}".format(-deformation))

        ax[0].plot(self.bragg1.wavelength_span_nm,
                   self.laser / self.laser.max() * self.bragg1.r0.max(),
                   label="Laser")
        ax[0].legend()

        photodetector_power = 0.25 * (1.0 - ref2) * ref1 * self.laser
        total_protodetector_power = np.trapezoid(x=self.bragg1.wavelength_span,
                                             y=photodetector_power)
        ax[1].plot(self.bragg1.wavelength_span_nm,
                   1e-3 * photodetector_power,
                   label="Potência no fotodetector=" +
                   '{:2.4f}'.format(total_protodetector_power * 1e6) +
                   r"\si{\micro\watt}")
        # ax[1].set_ylim([0, photodetector_power.max()],)
        ax[1].set_yticks([0, 1e-3 * photodetector_power.max()])
        ax[1].legend()
