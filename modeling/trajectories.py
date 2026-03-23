#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Trajectories.py
@Time    :   2023/11/05 23:55:44
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from common_functions import quaternion_functions as fq

plt.style.use("common_functions/roney3.mplstyle")
FIG_L = 6.29
FIG_A = (90.0) / 25.4


class Trajectory(object):
    """docstring for Trajectory."""

    def __init__(self, time_vector: np.ndarray, **kwargs):
        self.time_vector = time_vector
        self.n = time_vector.size
        # inertial states
        self.acceleration_vector_i = np.zeros((3, self.n))
        """ inertial acceleration """
        self.velocity_vector_i = np.zeros((3, self.n))
        """ inertial velocity """
        self.position_vector_i = np.zeros((3, self.n))
        """ inertial position """
        self.q_b_i = np.zeros((4, self.n))
        self.q_b_i[:, 0] = fq.eulerQuaternion(yaw=0, pitch=0, roll=0)
        """ quaternion transform inertial to body """
        # body states
        self.acceleration_vector_b = np.zeros((3, self.n))
        """ body acceleration """
        self.velocity_vector_b = np.zeros((3, self.n))
        """ body velocity """
        self.position_vector_b = np.zeros((3, self.n))
        """ body position """
        self.angular_velocity_vector_b = np.zeros((3, self.n))
        """ body velocity """
        self.angular_acceleration_vector_b = np.zeros((3, self.n))
        """ body acceleration """

        if kwargs["test"] == "translational_movement":
            print("Linear pure trajectory")
            # a gravity value on z direction
            tau = 1e-2
            frequency_trajectory = 2*np.pi/(0.05)
            self.acceleration_vector_i[0, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 5e-3, tau)
            ) * np.sin(frequency_trajectory * self.time_vector*2)
            self.acceleration_vector_i[1, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 20e-3, tau)
            )*np.sin(frequency_trajectory*self.time_vector)
            self.acceleration_vector_i[2, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 50e-3, tau)
            ) * np.cos(frequency_trajectory * self.time_vector)

        elif kwargs["test"] == "angular_movement":
            print(kwargs["test"])

            tau = 5e-2
            # a gravity value on z direction
            self.angular_velocity_vector_b[0, 0] = 0.0
            self.angular_velocity_vector_b[1, 0] = 0.0
            self.angular_velocity_vector_b[2, 0] = 0.0
            self.angular_acceleration_vector_b[0, :] = (
                5.0 * self.make_delayed_exponential_step(self.time_vector, 2e-3, tau)
            )
            self.angular_acceleration_vector_b[1, :] = (
                2.0 * self.make_delayed_exponential_step(self.time_vector, 50e-3, tau)
            )
            self.angular_acceleration_vector_b[2, :] = (
                5 * self.make_delayed_exponential_step(self.time_vector, 100e-3, tau)
            )
        elif kwargs["test"] == 'complete_movement':

            print(kwargs["test"])
            # a gravity value on z direction
            tau = 1e-2
            frequency_trajectory = 2 * np.pi / (0.5)
            self.acceleration_vector_i[0, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 5e-3, tau)
            ) * np.sin(frequency_trajectory * self.time_vector * 2)
            self.acceleration_vector_i[1, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 20e-3, tau)
            ) * np.sin(frequency_trajectory * self.time_vector)
            self.acceleration_vector_i[2, :] = (
                10.0 * self.make_delayed_exponential_step(self.time_vector, 50e-3, tau)
            ) * np.cos(frequency_trajectory * self.time_vector)
            tau = 5e-2
            # a gravity value on z direction
            self.angular_velocity_vector_b[0, 0] = 0.0
            self.angular_velocity_vector_b[1, 0] = 0.0
            self.angular_velocity_vector_b[2, 0] = 0.0
            self.angular_acceleration_vector_b[0, :] = (
                5.0 * self.make_delayed_exponential_step(self.time_vector, 2e-3, tau)
            ) * np.sin(frequency_trajectory * self.time_vector * 2)
            self.angular_acceleration_vector_b[1, :] = (
                5.0 * self.make_delayed_exponential_step(self.time_vector, 50e-3, tau)
            ) * np.sin(frequency_trajectory * self.time_vector)
            self.angular_acceleration_vector_b[2, :] = (
                5 * self.make_delayed_exponential_step(self.time_vector, 100e-3, tau)
            )* np.cos(frequency_trajectory * self.time_vector)

        # Make integration of initial reference body states
        dt = self.time_vector[1] - self.time_vector[0]
        for idx in range(self.n - 1):
            self.velocity_vector_i[:, idx + 1] = (
                self.velocity_vector_i[:, idx]
                + (
                    self.acceleration_vector_i[:, idx]
                    + self.acceleration_vector_i[:, idx + 1]
                )
                * 0.5
                * dt
            )
            self.position_vector_i[:, idx + 1] = (
                self.position_vector_i[:, idx]
                + (self.velocity_vector_i[:, idx] + self.velocity_vector_i[:, idx + 1])
                * 0.5
                * dt
            )
            self.angular_velocity_vector_b[:, idx + 1] = (
                self.angular_velocity_vector_b[:, idx]
                + (
                    self.angular_acceleration_vector_b[:, idx]
                    + self.angular_acceleration_vector_b[:, idx + 1]
                )
                * 0.5
                * dt
            )
            self.q_b_i[:, idx + 1] = fq.mult_quat(
                p=self.q_b_i[:, idx],
                q=fq.expMap(self.angular_velocity_vector_b[:, idx + 1], dt=dt),
            )
            self.acceleration_vector_b[:, idx + 1] = (
                fq.rotationMatrix(self.q_b_i[:, idx + 1]).T
                @ self.acceleration_vector_i[:, idx]
            )

    def make_delayed_exponential_step(
        self, time_vector: np.ndarray | float, time_delay=0.0, tau=1.0
    ):
        y = 1 - np.exp(-(time_vector - time_delay) / tau)
        for i, _y in enumerate(y):
            if _y < 0.0:
                y[i] = 0.0
        return y

    def plot_trajectories(self):
        self.plot_acceleration()
        self.plot_position()
        self.plot_velocity()
        self.plot_quaternion()
        self.plot_angular_velocity()

    def plot_acceleration(self):
        fig, ax = plt.subplots(3, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))

        fig.supylabel("aceleração")
        fig.supxlabel(r"Tempo, [$\unit{\ms}$]")
        ax[0].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_i[0, :],
            label=r"Sistema $\mathcal{I}$",
        )
        ax[1].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_i[1, :],
            label=r"Sistema $\mathcal{I}$",
        )
        ax[2].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_i[2, :],
            label=r"Sistema $\mathcal{I}$",
        )
        ax[0].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_b[0, :],
            label=r"Sistema $\mathcal{B}$",
        )
        ax[1].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_b[1, :],
            label=r"Sistema $\mathcal{B}$",
        )
        ax[2].plot(
            1e3 * self.time_vector,
            self.acceleration_vector_b[2, :],
            label=r"Sistema $\mathcal{B}$",
        )
        ax[2].legend()
        plt.show()

    def plot_velocity(self):
        fig, ax = plt.subplots(3, 1, num=2, sharex=True, figsize=(FIG_L, FIG_A))

        fig.supylabel("velocidade")
        fig.supxlabel(r"Tempo, [$\unit{\ms}$]")
        ax[0].plot(1e3 * self.time_vector, self.velocity_vector_i[0, :])
        ax[1].plot(1e3 * self.time_vector, self.velocity_vector_i[1, :])
        ax[2].plot(1e3 * self.time_vector, self.velocity_vector_i[2, :])

        plt.show()

    def plot_position(self):
        fig, ax = plt.subplots(3, 1, num=3, sharex=True, figsize=(FIG_L, FIG_A))

        fig.supylabel("posição")
        fig.supxlabel(r"Tempo, [$\unit{\ms}$]")
        ax[0].plot(1e3 * self.time_vector, self.position_vector_i[0, :])
        ax[1].plot(1e3 * self.time_vector, self.position_vector_i[1, :])
        ax[2].plot(1e3 * self.time_vector, self.position_vector_i[2, :])

        plt.show()

    def plot_angular_velocity(self):
        fig, ax = plt.subplots(3, 1, num=4, sharex=True, figsize=(FIG_L, FIG_A))

        fig.supylabel("velocidade angular")
        fig.supxlabel(r"Tempo, [$\unit{\ms}$]")
        ax[0].plot(1e3 * self.time_vector, self.angular_velocity_vector_b[0, :])
        ax[1].plot(1e3 * self.time_vector, self.angular_velocity_vector_b[1, :])
        ax[2].plot(1e3 * self.time_vector, self.angular_velocity_vector_b[2, :])

    def plot_angular_acceleration(self):
        fig, ax = plt.subplots(3, 1, num=5, sharex=True, figsize=(FIG_L, FIG_A))

        # fig.supylabel(r" $\dot{\mathbf{\omega}$ $[\unit{\radian\per\squared\second}]$")
        fig.supxlabel(r"Tempo, [$\unit{\ms}$]")
        ax[0].plot(1e3 * self.time_vector, self.angular_acceleration_vector_b[0, :])
        ax[1].plot(1e3 * self.time_vector, self.angular_acceleration_vector_b[1, :])
        ax[2].plot(1e3 * self.time_vector, self.angular_acceleration_vector_b[2, :])

    def plot_quaternion(self):
        fig, ax = plt.subplots(4, 1, num=6, sharex=True, figsize=(FIG_L, FIG_A))
        fig.supxlabel(r"Tempo, $[\unit{\ms}]$")
        fig.supylabel("quatérnion")
        ax[0].plot(1e3 * self.time_vector, self.q_b_i[0, :])
        ax[1].plot(1e3 * self.time_vector, self.q_b_i[1, :])
        ax[2].plot(1e3 * self.time_vector, self.q_b_i[2, :])
        ax[3].plot(1e3 * self.time_vector, self.q_b_i[3, :])
