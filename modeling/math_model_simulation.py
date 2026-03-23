#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   math_model_simulation.py
@Time    :   2023/10/24 09:04:05
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
from modeling.math_model_accel import AccelModelInertialFrame
from modeling.trajectories import Trajectory
from common_functions.RK import RungeKutta
from scipy.integrate import solve_ivp
from common_functions import quaternion_functions as fq
import argparse
import modeling.inverse_problem_solution as ips


class statesOfSimulation_object(object):
    def __init__(
        self, dt=1e-5, tf=1.0, hf=h5py.Group, accel=AccelModelInertialFrame
    ) -> None:
        """
        __init__ states of simulation collect all states too simulate
        Args:
            dt: Time step for integration. Defaults to 1e-5.NOTE: With dt<1e-5 the simulation show artificia damper.
            tf: Time of simulation. Defaults to 1.0.
        """
        self.hf = hf
        self.hf.attrs["dt"] = dt
        """ Step of time of simulation."""
        self.hf["t"] = np.arange(0.0, tf, dt, dtype=np.float64)
        self.hf["t"].attrs["about"] = "Time of simulation in seconds"

        self.hf["x"] = np.zeros((26, self.hf["t"].size), dtype=np.float64)
        self.hf["x"].attrs[
            "about"
        ] = "States  d_rb[:3] d_rm[3:6] rb[6:9] rm[9:12] qb[12:16] qm[16:20] wb[20:23] wm[23:26]"
        self.hf["true_accel_i"] = np.zeros(
            shape=(3, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_accel_i"].attrs[
            "about"
        ] = "Exact translational acceleration on inertial frame"
        self.hf["true_accel_b"] = np.zeros(
            shape=(3, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_accel_b"].attrs[
            "about"
        ] = "Exact translational acceleration on body frame"
        self.hf["true_angular_velocity_b"] = np.zeros(
            shape=(3, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_angular_velocity_b"].attrs[
            "about"
        ] = "Exact angular velocity on body frame"
        self.hf["true_angular_acceleration_b"] = np.zeros(
            shape=(3, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_angular_acceleration_b"].attrs[
            "about"
        ] = "Exact angular acceleration on body frame"
        """ Relative deformation with respect of the initial length (l-l0)/l. """
        self.hf["f"] = np.zeros(shape=(12, 3, self.hf["t"].size), dtype=np.float64)
        self.hf["fiber_len"] = np.zeros(shape=(12, self.hf["t"].size), dtype=np.float64)
        self.hf["true_relative_position"] = np.zeros(
            shape=(3, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_relative_position"].attrs[
            "about"
        ] = "seismic mass position with respect to base sensor "
        self.hf["true_relative_orientation"] = np.zeros(
            shape=(4, self.hf["t"].size), dtype=np.float64
        )
        self.hf["true_relative_orientation"].attrs[
            "about"
        ] = "seismic mass orientation with respect to base sensor "
        self.hf.attrs["b_B"] = accel.b_B
        self.hf.attrs["m_M"] = accel.m_M
        self.hf.attrs["k"] = accel.k
        self.hf.attrs["seismic_mass"] = accel.seismic_mass
        self.hf.attrs["fiber_length"] = accel.fiber_length
        self.hf.attrs["density"] = accel.density

def main():
    # parse = argparse.ArgumentParser(description="Makes the sensor dynamics simulation")
    # parse.add_argument(
    #     "-t",
    #     "--test_name",
    #     choices=[
    #         "step_response",
    #         "complete_movement",
    #         "translational_movement",
    #         "angular_movement",
    #     ],
    #     help="type of test",
    # )
    # parse.add_argument(
    #     "-f",
    #     "--file_name",
    #     help="name of hdf5 file to save data",
    # )
    # args = parse.parse_args()
    # print("teste name ",args.test_name, "file name of hdf5 ",args.file_name)
    hdf5_file = 'temp_file_model_damped.hdf5'
    # test_name = "translational_movement"
    # test_name = "angular_movement"
    test_name = 'complete_movement'

    accel = AccelModelInertialFrame(seismic_edge=16.3e-3,damper_for_computation_simulations=.01, fiber_length=6e-3, density=8e3)

    f = h5py.File(name=hdf5_file, mode="a", driver="core")#, backing_store=False)
    if test_name in f.keys():
        del f[test_name]
    ff = f.require_group(test_name)
    s = statesOfSimulation_object(tf=1.0, dt=1e-5, hf=ff, accel=accel)
    traj = Trajectory(time_vector=s.hf["t"][:], test=test_name)
    # traj.plot_trajectories()

    RK = RungeKutta(s.hf["x"].shape[0], accel.func_dd_x)
    # initial conditions for all quaternions as [1, 0, 0, 0]
    # initial misalignment
    s.hf["x"][12:16, 0] = traj.q_b_i[:, 0]
    s.hf["x"][16:20, 0] = traj.q_b_i[:, 0]
    # Set body sensor and seismic mass initial conditions
    s.hf["x"][:3, 0] = traj.velocity_vector_i[:, 0]  # body velocity
    s.hf["x"][3:6, 0] = traj.velocity_vector_i[:, 0]  # seismic mass velocity

    s.hf["x"][6:9, 0] = traj.position_vector_i[:, 0]  # body position
    s.hf["x"][9:12, 0] = traj.position_vector_i[:, 0]  # seismic mass position
    # s.hf["x"][11-2, 0] = 1e-6
    s.hf["x"][20:23, 0] = traj.angular_velocity_vector_b[:, 0]
    s.hf["x"][23:, 0] = traj.angular_velocity_vector_b[:, 0]

    accel.update_states(
        rb=s.hf["x"][6:9, 0],
        rm=s.hf["x"][9:12, 0],
        qb=s.hf["x"][12:16, 0],
        qm=s.hf["x"][16:20, 0],
    )
    # update body velocity
    accel.sms.dr = traj.velocity_vector_i[:, 0]
    accel.sms.w = traj.angular_velocity_vector_b[:, 0]
    s.hf["f"][:, :, 0] = accel.f
    s.hf["fiber_len"][:, 0] = np.linalg.norm(accel.f, axis=1)
    s.hf["true_accel_i"][:] = traj.acceleration_vector_i
    s.hf["true_accel_b"][:] = traj.acceleration_vector_b
    s.hf["true_angular_acceleration_b"][:] = traj.angular_acceleration_vector_b
    s.hf["true_angular_velocity_b"][:] = traj.angular_velocity_vector_b
    s.hf["true_relative_orientation"][:, 0] = fq.mult_quat(
        p=fq.conj(accel.bss.q), q=accel.sms.q
    )
    s.hf["true_relative_position"][:, 0] = (
        fq.rotationMatrix(s.hf["x"][12:16, 0]).T @ (accel.sms.r - accel.bss.r)
    )
    # if s.hf["true_relative_orientation"][0, 0] < 0.1:
    #     s.hf["true_relative_orientation"][:, 0] *= -1
    for i in tqdm(range(s.hf["t"].size - 1)):
        # integration of states
        s.hf["x"][:, i + 1] = RK.integrates_states(
            s.hf["x"][:, i], h=s.hf.attrs["dt"]
        )
        # solution = solve_ivp(
        #     accel.func_dd_x,
        #     t_span=(s.hf["t"][i], s.hf["t"][i + 1]),
        #     y0=s.hf["x"][:, i],
        #     # method="RK23",
        #     # method="DOP853",
        #     method="RK45",
        #     # rtol=1e-10,
        #     # atol=1e-10,
        #     # start_step=1e-11,
        #     # max_step=1e-6,
        #     # dense_output=True,
        # )

        # s.hf["x"][:, i + 1] = solution.y[:, -1]
        # Put the body in specific trajectories
        s.hf["x"][:3, i + 1] = traj.velocity_vector_i[:, i + 1]
        s.hf["x"][6:9, i + 1] = traj.position_vector_i[:, i + 1]
        s.hf["x"][20:23, i + 1] = traj.angular_velocity_vector_b[:, i + 1]

        s.hf["x"][12:16, i + 1] = fq.mult_quat(
            p=s.hf["x"][12:16, i],
            q=fq.expMap(s.hf["x"][20:23, i + 1], dt=s.hf.attrs["dt"]),
        )
        # s.hf["x"][16:20, i + 1] = fq.mult_quat(
        #     p=s.hf["x"][16:20, i],
        #     q=fq.expMap(s.hf["x"][23:26, i + 1], dt=s.hf.attrs["dt"]),
        # )
        # quaternion normalize
        # s.hf["x"][12:16, i + 1] /= np.linalg.norm(s.hf["x"][12:16, i + 1])
        # s.hf["x"][16:20, i + 1] /= np.linalg.norm(s.hf["x"][16:20, i + 1])
        # # update class structs to compute f vectors
        accel.update_states(
            rb=s.hf["x"][6:9, i + 1],
            rm=s.hf["x"][9:12, i + 1],
            qb=s.hf["x"][12:16, i + 1],
            qm=s.hf["x"][16:20, i + 1],
        )
        # update body velocity
        accel.sms.dr = traj.velocity_vector_i[:, i + 1]
        accel.sms.w = traj.angular_velocity_vector_b[:, i + 1]
        ## take the f vector of integration
        s.hf["f"][:, :, i + 1] = accel.f
        s.hf["fiber_len"][:, i + 1] = np.linalg.norm(accel.f, axis=1)
        s.hf["true_relative_position"][:, i + 1] = fq.rotationMatrix(
            s.hf["x"][12:16, i + 1]
        ).T @ (accel.sms.r - accel.bss.r)
        s.hf["true_relative_orientation"][:, i + 1] = fq.mult_quat(
            p=fq.conj(accel.bss.q), q=accel.sms.q
        )
        # if s.hf["true_relative_orientation"][0, i + 1] < 0.0:
        #     s.hf["true_relative_orientation"][:, i + 1] *= -1.0

    f.close()
    # make inverse problem and plot graphics
    ips.main(_test_name=test_name,_h5py_file_name=hdf5_file)


if __name__ == "__main__":

    
    # print(parse.print_help())
    main()
