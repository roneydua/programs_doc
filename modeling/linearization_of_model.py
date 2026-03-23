#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   linearization_of_model.py
@Time    :   2023/11/16 09:06:10
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""

import locale

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp

from modeling.math_model_accel import AccelModelInertialFrame
from common_functions import QuatSymbolic
import scipy.sparse.linalg as la

accel = AccelModelInertialFrame()
_e = sp.Symbol(r"\ell_{m}")#self.seismic_edge / 2.0
_d = sp.Symbol(r"e_m") #self.seismic_edge / 2.0 - 1e-3
_m_M = np.array(
    [
        [_e, _d, 0.0],  # X+ GRADE
        [_e, -_d, 0.0],
        [-_e, _d, 0.0],
        [-_e, -_d, 0.0],  # X GRADE
        [0.0, _e, _d],  # Y
        [0.0, _e, -_d],  # Y+ GRADE
        [0.0, -_e, _d],  # Y- GRADE
        [0.0, -_e, -_d],
        [_d, 0, _e],
        [-_d, 0, _e],  # Z+ grade
        [_d, 0, -_e],  # Z- grade
        [-_d, 0, -_e],
    ]
)

"""Connections in the seismic mass """
l = sp.Symbol(r"\ell", real=True)
_f = sp.Symbol(r"\ell_{m}")+l#self.base_sensor_edge / 2.0
_b_B = np.array(
    [
        [_f, _d, 0.0],  # X
        [_f, -_d, 0.0],
        [-_f, _d, 0.0],
        [-_f, -_d, 0.0],
        [0.0, _f, _d],  # Y
        [0.0, _f, -_d],
        [0.0, -_f, _d],
        [0.0, -_f, -_d],
        [_d, 0, _f],
        [-_d, 0, _f],
        [_d, 0, -_f],
        [-_d, 0, -_f],
    ]
)


m_M = sp.Matrix(_m_M)
b_B = sp.Matrix(_b_B)

# m_M = sp.Matrix(accel.m_M)
# b_B = sp.Matrix(accel.b_B)
r = sp.Matrix([sp.Symbol(r"r_x"), sp.Symbol(r"r_y"), sp.Symbol(r"r_z")])
q = QuatSymbolic.QuaternionSymbolic(sub="M")


def calc_f(index: int):
    """calc_f calculate function j at point j
    Returns:
        The vector f on point j
    """
    return r + q.quat_rot().T @ m_M[index, :].T - b_B[index, :].T


def calc_dl(f_i: sp.matrices.dense.MutableDenseMatrix):
    return sp.sqrt((f_i.T @ f_i)[0]) - l


r_dd = 0.0 * r
""" Translacional acceleration function"""

for i in range(12):
    f_i = calc_f(i)
    r_dd += (f_i * calc_dl(f_i)) / sp.sqrt((f_i.T @ f_i)[0])


def calc_jacobian(_f, _r, _f_is_scalar=False):
    if _f_is_scalar:
        derivative = sp.zeros(3, 1)
        ncol = 1
        for row, col in [(row, col) for row in range(3) for col in range(ncol)]:
            derivative[row, col] = sp.diff(_f, _r[col])
        return derivative
    else:
        ncol = _r.shape[0]
        nrow = _f.shape[0]
        derivative = sp.zeros(nrow, ncol)
        for row, col in [(row, col) for row in range(nrow) for col in range(ncol)]:
            derivative[row, col] = sp.diff(_f[row], _r[col])
        return derivative


derivative_r = calc_jacobian(r_dd, r)
# Evaluation of the derivative at the operating point


def calc_function_on_operation_point(_f):
    n_cols, n_rows = _f.shape
    linear_matrix = sp.zeros(n_cols, n_rows)
    for row, col in [(row, col) for row in range(n_cols) for col in range(n_rows)]:
        linear_matrix[row, col] = _f[row, col].subs(
            [
                (r[0], 0.0),
                (r[1], 0.0),
                (r[2], 0.0),
                (q.quat[0], 1.0),
                (q.quat[1], 0.0),
                (q.quat[2], 0.0),
                (q.quat[3], 0.0),
                (l, 3e-3),
            ]
        )
    return linear_matrix

# Calculate the derivatives
linear_matrix_linear_acceleration_r_r = calc_function_on_operation_point(derivative_r)
""" linearized linear acceleration"""


derivative_q = calc_jacobian(r_dd, q.quat)
linear_matrix_linear_acceleration_r_q = calc_function_on_operation_point(derivative_q)


## Angular analysis
r_dd_a = 0.0 * r
""" Angular acceleration function"""


def calc_dfdq(q, v):
    """
    calc_dfdq Calculate jacobian of rotated vector
    Args:
        q: quaternion rotation
        v: vector 3x1
    Returns:
        Jacobian of rotation vector w.r.t quaterion.
        NOTE: This function return transpose of Jacobian
    """

    def screw(quat):
        q_x = sp.zeros(3)
        q_x[0, 1] = -quat[-1]
        q_x[1, 0] = quat[-1]
        q_x[0, 2] = quat[-2]
        q_x[2, 0] = -quat[-2]
        q_x[1, 2] = -quat[-3]
        q_x[2, 1] = quat[-3]
        return q_x

    dfdq = sp.zeros(4, 3)
    dfdq[0, :] = q.quat[0] * v.reshape(1, 3) - v.reshape(1, 3) @ screw(q.quat[1:])
    dfdq[1:, :] = (q.quat[1:, 0].T @ v.reshape(3, 1))[0] * sp.eye(3)
    dfdq[1:, :] += v.reshape(3, 1) @ q.quat[1:, 0].T
    dfdq[1:, :] -= q.quat[1:, 0] @ v.reshape(1, 3)
    dfdq[1:, :] += q.quat[0] * screw(v)
    return 2.0 * dfdq


for i in range(12):
    f_i = calc_f(i)
    r_dd_a += (
        calc_dl(f_i)
        * q.Q().T
        @ calc_dfdq(q, m_M[i, :])
        * f_i
        / sp.sqrt((f_i.T @ f_i)[0])
    )
# derivative of omega dot with respect to position
derivative_q_r = calc_jacobian(r_dd_a, r)
linear_matrix_linear_acceleration_q_r = calc_function_on_operation_point(derivative_q_r)

# derivative of omega dot with respect to quaterion
derivative_q_q = calc_jacobian(r_dd_a, q.quat)
linear_matrix_angular_acceleration_q_q = calc_function_on_operation_point(
    derivative_q_q
)

linear_matrix = sp.zeros(3, 4)

for i in range(12):
    fi = calc_f(i)
    linear_matrix += (
        q.Q().T
        @ calc_dfdq(q, m_M[i, :])
        @ fi
        @ (calc_dfdq(q, m_M[i, :]) @ fi).T
        / l**2
    )
# derivative of omega dot with respect to quaterion
linear_matrix_angular_acceleration_q_q = calc_function_on_operation_point(
    linear_matrix
)

mat_K = sp.zeros(6, 6)

mat_K[:3, :3] = linear_matrix_linear_acceleration_r_r
mat_K[3:, 3:] = linear_matrix_angular_acceleration_q_q[:, 1:]
print(sp.latex(mat_K))


mat_w = 0.5 * 0.00082944 * np.eye(3) * accel.k
mat_r = 4.0 * accel.k * np.eye(3)
mat_k_global = np.zeros((6, 6))
mat_k_global[:3, :3] = mat_r
mat_k_global[-3:, -3:] = mat_w


mat_m_global = np.zeros((6, 6))
mat_m_global[:3, :3] = np.eye(3) * accel.seismic_mass
mat_m_global[-3:, -3:] = accel.inertial_seismic_mass


d = np.linalg.inv(mat_m_global) @ mat_k_global
# 1.0/(np.sqrt(scipy.linalg.eig(d)[0]) / (2 * np.pi))


np.sqrt(0.5 * 0.00082944 * accel.k / accel.inertial_seismic_mass[0, 0]) / (np.pi * 2)
np.sqrt(4* accel.k / accel.seismic_mass)/(np.pi*2)

np.eye(4,3,k=-1).T
mat_temp = np.zeros((3,4))
mat_temp[:,1:] = np.eye(3)
# numerical analysis
_e = (accel.seismic_edge/2-1e-3)

mat_linear = np.zeros((13,13))
mat_linear[:3,3:6] = np.eye(3) 
mat_linear[3:6, :3] = 4.0 * accel.k / accel.seismic_mass * np.eye(3)
mat_linear[7:10,10:] = .5*np.eye(3)
mat_linear[10:,7:10] = 8*accel.k*_e**2*accel.inertial_seismic_mass_inv

np.linalg.eig(mat_linear).eigenvalues / (np.pi * 2)


np.sqrt(4.0 * accel.k / accel.seismic_mass) / (np.pi * 2)


mat_mass = np.zeros((6, 6))
mat_stiffness = np.zeros((12, 12))

mat_linear[:3, 3:6] = np.eye(3)
mat_linear[3:6, :3] = 4.0 * accel.k / accel.seismic_mass*np.eye(3)
mat_linear[6:9, 9:] = 0.5 * np.eye(3)
mat_linear[10:, 6:9] = 8 * accel.k * _e**2 * accel.inertial_seismic_mass_inv
mat_linear
np.linalg.eig(mat_linear).eigenvalues / (2.0 * np.pi)

AA = np.array([[0,1],[4*accel.k/accel.seismic_mass,0]])

np.linalg.eig(AA).eigenvalues/(2.0*np.pi)
print(np.array(mat_linear,dtype=np.float32))
