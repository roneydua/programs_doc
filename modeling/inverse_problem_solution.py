#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   inverse_problem_solution.py
@Time    :   2023/10/19 15:15:52
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""
import locale

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset


import common_functions.quaternion_functions as fq
from modeling import math_model_accel
from modeling.math_model_accel import InverseProblem, SimpleSolution

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 144
FIG_L = 6.29
# change for 16:10
FIG_A = FIG_L / 1.6
SAVE_FORMAT = "png"
# TEST_NAME = "translational_movement"
TEST_NAME = "complete_movement"
# TEST_NAME = "step_response"
# TEST_NAME = "angular_movement"
H5PY_FILE_NAME = "temp_file_model.hdf5"
TESE_FOLDER = "./../tese/images/used_on_thesis/"
GRAPHICS_IN_DEG = True
# ONLY_SHOW = True
ONLY_SHOW = False

if ONLY_SHOW:
    plt.rcParams["figure.dpi"] = 144

graphics = [
    # "one_fiber",
    "differential_aligned",
    "ls_translational",
    "ls_translational_angular_reduced",
    "ls_translational_angular_full",
    # "differential_cross",
]


def set_precision_of_graphcis(_d):
    # return np.float16(_d)
    return _d


def fix_unit(a):
    if GRAPHICS_IN_DEG:
        return 2 * np.pi * a
    else:
        return a


def recover_acceleration():
    # Load test to get parameters
    f = h5py.File(H5PY_FILE_NAME, "a")
    _ff = f[TEST_NAME]
    if "accel_recover" in _ff.keys():
        del _ff["accel_recover"]
    ff = _ff.require_group("accel_recover")
    if "ls_solution" in _ff.keys():
        del _ff["ls_solution"]
    fff = _ff.require_group("ls_solution")
    # _delta_time_size_new = int(1e-2 / (_ff["t"][1] - _ff["t"][0]))
    # if _ff.attrs["dt"] < 1e-4:
    #     new_dt = 1e-4
    # else:
    #     new_dt = _ff.attrs["dt"]
    # _delta_n = int(new_dt / (_ff["t"][1] - _ff["t"][0]))
    # _time_size = int(_ff["t"][:].size * (_ff["t"][1] - _ff["t"][0]) / new_dt)
    # # print(_delta_n, _time_size)
    # if _time_size < 1:
    #     print("_time_size muito pequeno")
    #     _delta_n = 1
    #     _time_size = _ff["t"][:].size
    _delta_n = 1
    _time_size = _ff["t"][:].size

    fibers_with_length_info = np.array([1, 5, 9, 11])
    ip_trans = InverseProblem(
        fibers_with_info=np.array([1, 5, 9, 11]),
        density=_ff.attrs["density"],
        fiber_length=_ff.attrs["fiber_length"],
    )
    ip_trans_ang_full = InverseProblem(
        fibers_with_info=np.arange(1, 13),
        recover_angular_accel=True,
        estimation="full",
        density=_ff.attrs["density"],
        fiber_length=_ff.attrs["fiber_length"],
    )
    ip_trans_ang_reduced = InverseProblem(
        fibers_with_info=np.array([1, 4, 5, 8, 9, 11, 12]),
        recover_angular_accel=True,
        estimation="reduced",
        density=_ff.attrs["density"],
        fiber_length=_ff.attrs["fiber_length"],
    )
    ss = SimpleSolution(
        np.array([1, 5, 9]),
        density=_ff.attrs["density"],
        fiber_length=_ff.attrs["fiber_length"],
    )

    ff.create_dataset("t", _time_size, dtype=np.float64)
    # Simple methods
    ff.create_dataset("one_fiber", (3, _time_size), dtype=np.float64)
    ff["one_fiber"].attrs["method_name"] = "Simples (1 FO)"
    ff.create_dataset("differential_aligned", (3, _time_size), dtype=np.float64)
    ff["differential_aligned"].attrs["method_name"] = "Diferencial alinhado"
    ff.create_dataset("differential_cross", (3, _time_size), dtype=np.float64)
    ff["differential_cross"].attrs["method_name"] = "Diferencial cruzado"
    # Least squared methods
    ff.create_dataset("ls_translational", (3, _time_size), dtype=np.float64)
    ff["ls_translational"].attrs["method_name"] = "MMQ translacional"
    ff["ls_translational"].attrs["fibers_with_info"] = ip_trans.fibers_with_info
    fff.create_dataset("ls_translational", (4, _time_size), dtype=np.float64)
    fff["ls_translational"].attrs["method_name"] = "MMQ translacional"

    ff.create_dataset(
        "ls_translational_angular_reduced", (6, _time_size), dtype=np.float64
    )
    ff["ls_translational_angular_reduced"].attrs["method_name"] = "MMQ reduzido"
    ff["ls_translational_angular_reduced"].attrs[
        "fibers_with_info"
    ] = ip_trans_ang_reduced.fibers_with_info
    fff.create_dataset(
        "ls_translational_angular_reduced", (7, _time_size), dtype=np.float64
    )
    fff["ls_translational_angular_reduced"].attrs["method_name"] = "MMQ reduzido"

    ff.create_dataset(
        "ls_translational_angular_full", (6, _time_size), dtype=np.float64
    )
    ff["ls_translational_angular_full"].attrs["method_name"] = "MMQ completo"
    ff["ls_translational_angular_full"].attrs[
        "fibers_with_info"
    ] = ip_trans_ang_full.fibers_with_info
    fff.create_dataset(
        "ls_translational_angular_full", (10, _time_size), dtype=np.float64
    )
    fff["ls_translational_angular_full"].attrs["method_name"] = "MMQ completo"
    fff.create_dataset(
        "ls_translational_angular_full", (10, _time_size), dtype=np.float64
    )
    fff["ls_translational_angular_full"].attrs["method_name"] = "MMQ completo"
    fff.create_dataset(
        "ls_translational_angular_reduced_attitude_error",
        (4, _time_size),
        dtype=np.float64,
    )
    fff["ls_translational_angular_reduced_attitude_error"].attrs[
        "method_name"
    ] = "MMQ reduzido"
    fff.create_dataset(
        "ls_translational_angular_full_attitude_error",
        (4, _time_size),
        dtype=np.float64,
    )
    fff["ls_translational_angular_full_attitude_error"].attrs[
        "method_name"
    ] = "MMQ completo"

    idx_new = 0
    for idx in tqdm(range(0, int(_ff["t"][:].size), _delta_n)):
        ff["t"][idx_new] = _ff["t"][idx]

        ff["one_fiber"][:, idx_new] = ss.estimated_ddrm_B(
            _ff["fiber_len"][:, idx], method="one_fiber"
        )
        ff["differential_aligned"][:, idx_new] = ss.estimated_ddrm_B(
            _ff["fiber_len"][:, idx], method="differential_aligned"
        )
        ff["differential_cross"][:, idx_new] = ss.estimated_ddrm_B(
            _ff["fiber_len"][:, idx], method="differential_cross"
        )
        # Least squared matrix
        ff["ls_translational"][:, idx_new] = ip_trans.compute_inverse_problem_solution(
            np.take(_ff["fiber_len"][:, idx], ip_trans.fibers_with_info_index)
        )
        fff["ls_translational"][:, idx_new] = ip_trans.var_gamma

        (
            ff["ls_translational_angular_reduced"][:3, idx_new],
            ff["ls_translational_angular_reduced"][3:, idx_new],
        ) = ip_trans_ang_reduced.compute_inverse_problem_solution(
            np.take(
                _ff["fiber_len"][:, idx], ip_trans_ang_reduced.fibers_with_info_index
            )
        )
        fff["ls_translational_angular_reduced"][
            :, idx_new
        ] = ip_trans_ang_reduced.var_gamma
        (
            ff["ls_translational_angular_full"][:3, idx_new],
            ff["ls_translational_angular_full"][3:, idx_new],
        ) = ip_trans_ang_full.compute_inverse_problem_solution(
            np.take(_ff["fiber_len"][:, idx], ip_trans_ang_full.fibers_with_info_index)
        )
        fff["ls_translational_angular_full"][:, idx_new] = ip_trans_ang_full.var_gamma
        idx_new += 1
    """compute error of attitude"""
    for j in ["ls_translational_angular_reduced", "ls_translational_angular_full"]:
        q = np.zeros((4, ff["t"][:].size))
        for k in range(ff["t"][:].size):
            q[0, k] = np.sqrt(
                1.0 - fff[j][-3, k] ** 2 - fff[j][-2, k] ** 2 - fff[j][-1, k] ** 2
            )
            q[1, k] = fff[j][-3, k]
            q[2, k] = fff[j][-2, k]
            q[3, k] = fff[j][-1, k]

        fff[j + "_attitude_error"][:, k] = fq.mult_quat(
            f["complete_movement"]["true_relative_orientation"][:, k],
            fq.conj(q[:, k]),
        )
        fff[j + "_attitude_error"].attrs["info"] = "diference between"
    f.close()


def plot_recover_acceleration():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(1)

    dt_sim = ff["t"][1] - ff["t"][0]
    dt_rec = fff["t"][1] - fff["t"][0]
    n_factor = int(dt_rec / dt_sim)
    index_vec = np.arange(0, ff["t"][:].size, n_factor)

    fig, ax = plt.subplots(3, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    y_axis_name = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for i in range(3):
        linewidth = 2
        # ax[i].set_yscale("log")
        ax[i].plot(
            ff["t"][:] * time_multiply,
            ff["true_accel_b"][i, :],
            label="Referência",
            lw=linewidth,
        )
        for j in graphics:
            if j in fff.keys():
                ax[i].plot(
                    fff["t"][:] * time_multiply,
                    fff[j][i, :],
                    label=fff[j].attrs["method_name"],
                    lw=linewidth,
                )
                linewidth -= 0.2
        ax[i].set_ylabel(y_axis_name[i])

    fig.supylabel(
        r"Aceleração translacional local $[\unit{\meter\per\second\squared}]$"
    )
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    ax[0].legend(
        ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    )
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "recover_translational_acceleration_"
            + TEST_NAME
            + "."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()
    f.close()


def plot_error_recover_acceleration():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(1)

    dt_sim = ff["t"][1] - ff["t"][0]
    dt_rec = fff["t"][1] - fff["t"][0]
    n_factor = int(dt_rec / dt_sim)
    index_vec = np.arange(0, ff["t"][:].size, n_factor)

    fig, ax = plt.subplots(3, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    y_axis_name = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for i in range(3):
        linewidth = 2
        # ax[i].set_yscale("log")
        # ax[i].plot(
        #     ff["t"][:] * time_multiply,
        #     ff["true_accel_b"][i, :],
        #     label="Referência",
        #     lw=linewidth,
        # )
        for j in graphics:
            if j in fff.keys():
                ax[i].plot(
                    fff["t"][:] * time_multiply,
                    np.abs(fff[j][i, :] - ff["true_accel_b"][i, :]),
                    label=fff[j].attrs["method_name"],
                    lw=linewidth,
                )
                linewidth -= 0.2
        ax[i].set_ylabel(y_axis_name[i])

    fig.supylabel(
        r"Erro da aceleração translacional $[\unit{\meter\per\second\squared}]$"
    )
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    ax[0].legend(
        ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    )
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "error_recover_translational_acceleration_"
            + TEST_NAME
            + "."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()
    f.close()


def plot_recover_angular_acceleration():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(1)
    y_axis_name = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        num=1,
        sharex="col",
        figsize=(FIG_L, FIG_A),
        # gridspec_kw=dict(width_ratios=[1, 0.5], height_ratios=[1, 1, 1]),
    )
    if fff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    # linewidth = []
    for i in range(3):
        ax[i].set_ylabel(y_axis_name[i])
        lw = 3.0
        for j in ["ls_translational_angular_reduced", "ls_translational_angular_full"]:
            ax[i].plot(
                fff["t"][:] * time_multiply,
                fix_unit(fff[j][i + 3, :]),
                label=fff[j].attrs["method_name"],
                lw=lw,
            )
            lw -= 1.0
        ax[i].plot(
            ff["t"][:] * time_multiply,
            fix_unit(ff["true_angular_acceleration_b"][i, :]),
            label="Referência",
            lw=lw,
        )
    ax[0].legend(
        ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    )
    if GRAPHICS_IN_DEG:
        fig.supylabel(r"Aceleração angular $[\unit{\degree\per\second\squared}]$")
    else:
        fig.supylabel(r"Aceleração angular $[\unit{\radian\per\second\squared}]$")
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "recover_angular_acceleration_"
            + TEST_NAME
            + "."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=1)
    else:
        plt.show()
    f.close()


def plot_error_recover_angular_acceleration():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(1)
    y_axis_name = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        num=1,
        sharex="col",
        figsize=(FIG_L, FIG_A),
        # gridspec_kw=dict(width_ratios=[1, 0.5], height_ratios=[1, 1, 1]),
    )
    if fff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    # linewidth = []
    for i in range(3):
        ax[i].set_ylabel(y_axis_name[i])
        # lw = 3.0
        for j in ["ls_translational_angular_reduced", "ls_translational_angular_full"]:
            ax[i].plot(
                fff["t"][:] * time_multiply,
                fix_unit(
                    np.abs(fff[j][i + 3, :] - ff["true_angular_acceleration_b"][i, :])
                ),
                label=fff[j].attrs["method_name"],
                # lw=lw,
            )
    ax[0].legend(
        ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    )
    if GRAPHICS_IN_DEG:
        fig.supylabel(
            r"Erro da aceleração angular $[\unit{\degree\per\second\squared}]$"
        )
    else:
        fig.supylabel(r"Aceleração angular $[\unit{\radian\per\second\squared}]$")
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "error_recover_angular_acceleration_"
            + TEST_NAME
            + "."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=1)
    else:
        plt.show()
    f.close()


def plot_deformations():

    f = h5py.File(H5PY_FILE_NAME, "r")
    accel = math_model_accel.AccelModelInertialFrame()
    ff = f[TEST_NAME]
    plt.close(2)
    fig, ax = plt.subplots(3, 4, num=2, sharex=True, figsize=(FIG_L, 2 * FIG_A))
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")

    for i, _ax in enumerate(ax.flat):
        print(accel.leg[i], r"\longrightarrow ", r"$\text{FBG}_{" + str(i + 1) + "}$")
    # _ax.set_xlim(50,55)
    # _ax.set_ylim(-50, 50)

    for i, _ax in enumerate(ax.flat):
        _ax.plot(
            time_multiply * ff["t"][...],
            1e6 * (ff["fiber_len"][i, :] / ff.attrs["fiber_length"] - 1.0),
            label=accel.leg[i],
        )
        # _ax.legend()
        _ax.set_title(r"$\text{FBG}_{" + str(i + 1) + "}$")
        # _ax.set_xlim(50,55)
        # _ax.set_ylim(-50, 50)

    fig.supylabel(r"Deformação $\frac{\ell-\ell_0}{\ell_0}\times{10}^{6}$")

    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER + "deformations_" + TEST_NAME + "." + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()
    f.close()


def plot_inertial_states():
    f = h5py.File(name=H5PY_FILE_NAME, mode="r", driver="core", backing_store=False)
    ff = f[TEST_NAME]
    plt.close(3)
    fig, ax = plt.subplots(4, 3, num=3, sharex=True, figsize=(FIG_L, FIG_A * 1.5))
    ind_x = [0, 6, 12, 20]
    ax[0][0].set_ylabel(r"${}^\mathcal{I}\dot{\mathbf{r}}[\si{\meter\per\second}]$")
    ax[1][0].set_ylabel(r"${}^\mathcal{I}\mathbf{r}[\si{\meter}]$")
    ax[2][0].set_ylabel(r"$\mathbf{q}$")
    if GRAPHICS_IN_DEG:
        ax[3][0].set_ylabel(r"$\boldsymbol{\omega}[\si{\degree\per\second}]$")
    else:
        ax[3][0].set_ylabel(r"$\boldsymbol{\omega}[\si{\radian\per\second}]$")
    ax[0, 0].set_title(r"$\hat{\boldsymbol{x}}$")
    ax[0, 1].set_title(r"$\hat{\boldsymbol{y}}$")
    ax[0, 2].set_title(r"$\hat{\boldsymbol{z}}$")
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for lin in range(4):
        if lin == 2:
            # pass
            for col in range(3):
                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(ff["x"][ind_x[lin] + col + 5, :]),
                    label=r"$q_{" + str(col + 1) + "}$ Massa sísmica",
                    lw=1.5,
                )
                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(ff["x"][ind_x[lin] + col + 1, :]),
                    label=r"$q_{" + str(col + 1) + "}$ Base",
                    lw=0.5,
                )
        else:
            for col in range(3):
                _d = ff["x"][ind_x[lin] + col + 3, :]
                _dd = ff["x"][ind_x[lin] + col, :]
                if GRAPHICS_IN_DEG and lin == 3:
                    _d *= 2.0 * np.pi
                    _dd *= 2.0 * np.pi
                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(_d),
                    lw=1.5,
                )
                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(_dd),
                    lw=0.5,
                )

    ax[0, 2].legend(["Massa sísmica", "Base"])
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER + "all_states_" + TEST_NAME + "." + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()

    f.close()


def plot_inertial_states_difference():
    f = h5py.File(name=H5PY_FILE_NAME, mode="r", driver="core", backing_store=False)
    ff = f[TEST_NAME]
    plt.close(3)
    fig, ax = plt.subplots(4, 3, num=3, sharex=True, figsize=(FIG_L, FIG_A * 1.5))
    ind_x = [0, 6, 12, 20]
    ax[0][0].set_ylabel(
        r"${}^\mathcal{I}\dot{\mathbf{r}_{m}}-{}^\mathcal{I}\dot{\mathbf{r}_{b}}[\si{\micro\meter\per\second}]$"
    )
    ax[1][0].set_ylabel(
        r"${}^\mathcal{I}\mathbf{r}_{m}-{}^\mathcal{I}\mathbf{r}_{b}[\si{\micro\meter}]$"
    )
    ax[2][0].set_ylabel(r"$\,_{\mathcal{M}}^\mathcal{B}\boldsymbol{q}_{v}$")
    if GRAPHICS_IN_DEG:
        ax[3][0].set_ylabel(
            r"$\boldsymbol{\omega}_{m}-\boldsymbol{\omega}_{b}[\si{\milli\degree\per\second}]$"
        )
    else:
        ax[3][0].set_ylabel(r"$\boldsymbol{\omega}[\si{\radian\per\second}]$")
    ax[0, 0].set_title(r"$\hat{\boldsymbol{x}}$")
    ax[0, 1].set_title(r"$\hat{\boldsymbol{y}}$")
    ax[0, 2].set_title(r"$\hat{\boldsymbol{z}}$")
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for lin in range(4):
        if lin == 2:
            # quaternion if
            for col in range(3):
                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(
                        ff["x"][ind_x[lin] + col + 5, :]
                        - ff["x"][ind_x[lin] + col + 1, :]
                    ),
                    label=r"$q_{" + str(col + 1) + "}$ Massa sísmica",
                    lw=1.5,
                )
                # massa sismica - base
        else:
            for col in range(3):
                _d = ff["x"][ind_x[lin] + col + 3, :] - ff["x"][ind_x[lin] + col, :]
                if GRAPHICS_IN_DEG and lin == 3:
                    _d *= 2.0 * np.pi * 1e3
                else:
                    _d *= 1e6

                ax[lin, col].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(_d),
                    lw=1.5,
                )
            if lin == 0 and col == 1:
                axis_in = ax[0, 1].inset_axes(
                    [0.5, 0.1, 0.4, 0.4], xlim=(0.2, 0.4), ylim=(-10, 10)
                )
                axis_in.plot(time_multiply * ff["t"][:], set_precision_of_graphcis(_d))

    fig.dpi = 144
    fig.supylabel("Diferença entre o estados da massa sísmica e a base")
    # ax[0, 2].legend(["Massa sísmica - Base"])
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER + "all_states_" + TEST_NAME + "_differences." + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()

    f.close()


def plot_length_of_fo():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    plt.close(2)
    fig, ax = plt.subplots(3, 4, num=2, sharex=True, figsize=(FIG_L, FIG_A))

    for i, _ax in enumerate(ax.flat):
        _ax.plot(
            1e3 * ff["t"][:],
            1e3 * ff["fiber_len"][i, :],
        )
        _ax.set_title(r"$\text{FBG}_{" + str(i + 1) + "}$")
        # _ax.set_xlim(50,55)
        _ax.set_ylim(2, 4)

    fig.supylabel(r"$\ell$")
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    # fig.legend()

    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER + "length_of_fo_" + TEST_NAME + "." + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=2)
    else:
        plt.show()
    f.close()


def plot_inertial_states_difference_one_column():
    f = h5py.File(name=H5PY_FILE_NAME, mode="r", driver="core", backing_store=False)
    ff = f[TEST_NAME]
    plt.close(3)
    fig, ax = plt.subplots(4, 1, num=3, sharex=True, figsize=(FIG_L, FIG_A * 1.5))
    ind_x = [0, 6, 12, 20]
    ax[0].set_ylabel(
        r"${}^\mathcal{I}\dot{\mathbf{r}_{m}}-{}^\mathcal{I}\dot{\mathbf{r}_{b}}[\si{\micro\meter\per\second}]$"
    )
    ax[1].set_ylabel(
        r"${}^\mathcal{I}\mathbf{r}_{m}-{}^\mathcal{I}\mathbf{r}_{b}[\si{\micro\meter}]$"
    )
    ax[2].set_ylabel(r"$\,_{\mathcal{M}}^\mathcal{B}\boldsymbol{q}_{v}$")
    if GRAPHICS_IN_DEG:
        ax[3].set_ylabel(
            r"$\boldsymbol{\omega}_{m}-\boldsymbol{\omega}_{b}[\si{\milli\degree\per\second}]$"
        )
    else:
        ax[3].set_ylabel(r"$\boldsymbol{\omega}[\si{\radian\per\second}]$")
    axis_label = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for lin in range(4):
        if lin == 2:
            # quaternion if
            for col in range(3):
                ax[lin].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(
                        ff["x"][ind_x[lin] + col + 5, :]
                        - ff["x"][ind_x[lin] + col + 1, :]
                    ),
                    label=axis_label[col],
                    lw=1.5,zorder=3-col,
                )
                # massa sismica - base
                ax[lin].legend()
        else:
            for col in range(3):
                _d = ff["x"][ind_x[lin] + col + 3, :] - ff["x"][ind_x[lin] + col, :]
                if GRAPHICS_IN_DEG and lin == 3:
                    _d *= 2.0 * np.pi * 1e3
                else:
                    _d *= 1e6

                ax[lin].plot(
                    time_multiply * ff["t"][:],
                    set_precision_of_graphcis(_d),
                    lw=1.5,
                    label=axis_label[col],
                    zorder=3 - col,
                )
    for lin in range(4):
        axins = inset_axes(ax[lin], width="25%", height="70%", loc="upper right")
        # Removendo números e ticks
        # axins.set_xticks([])
        axins.set_yticks([])
        # axins.set_xticklabels([])
        # axins.set_yticklabels([])
        mark_inset(ax[lin], axins, loc1=2, loc2=3, fc="none", ec="0.5")
        x_min, x_max = 0.59, 0.594
        axins.set_xlim(x_min, x_max)
        # Coletar os y visíveis no intervalo [x_min, x_max]
        y_visible = []

        for line in ax[lin].get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            axins.plot(xdata, ydata,color=line.get_color(), linewidth=line.get_linewidth(),zorder=line.get_zorder())
            # Máscara para dados no intervalo de zoom
            mask = (xdata >= x_min) & (xdata <= x_max)
            if np.any(mask):
                y_visible.extend(ydata[mask])
        # Ajuste automático do ylim com margem
        if y_visible:
            y_min = min(y_visible)
            y_max = max(y_visible)
            margin = 0.05 * (y_max - y_min) if y_max > y_min else 1e-6
            axins.set_ylim(y_min - margin, y_max + margin)
            # axins.set_ylim('auto','auto')
    fig.supylabel("Diferença entre o estados da massa sísmica e a base")
    # ax[0, 2].legend(["Massa sísmica - Base"])
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "all_states_"
            + TEST_NAME
            + "_differences_one_column."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()

    f.close()


def plot_accelerations():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    ac_translation = ff["true_accel_i"]
    ac_rotational = ff["true_angular_acceleration_b"]
    t = ff["t"]
    plt.close(1)

    fig, ax = plt.subplots(2, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    y_axis_name = [
        r"$\hat{\boldsymbol{x}}$",
        r"$\hat{\boldsymbol{y}}$",
        r"$\hat{\boldsymbol{z}}$",
    ]
    if ff["t"][:].max() > 1.0:
        time_multiply = 1e-3
        fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    else:
        time_multiply = 1.0
        fig.supxlabel(r"Tempo $[\unit{\s}]$")
    for i in range(3):
        linewidth = 2
        # ax[i].set_yscale("log")
        ax[0].plot(
            t[:] * time_multiply,
            ac_translation[i, :],
            label=y_axis_name[i],
            zorder = 3-i
        )
        ax[1].plot(
            t[:] * time_multiply,
            ac_rotational[i, :],
            label=y_axis_name[i],
            zorder=3 - i,
        )
        ax[0].set_ylabel(r"${}^\mathcal{I}\ddot{\mathbf{r}}_{b}[\si{\meter\per\second\squared}]$")
        ax[1].set_ylabel(
            r"$\boldsymbol{\omega}_{m}[\si{\radian\per\second}]$"
        )

    fig.supxlabel(r"Tempo $[\unit{\s}]$")
    # ax[0].legend(
    #     ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    # )
    ax[0].legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER
            + "translational_acceleration_"
            + TEST_NAME
            + "."
            + SAVE_FORMAT,
            format=SAVE_FORMAT,
        )
        plt.close(fig=3)
    else:
        plt.show()
    f.close()


def plot_ls_solution():
        f = h5py.File(H5PY_FILE_NAME, "r")
        ff = f[TEST_NAME]
        fff = ff["accel_recover"]
        y_axis_name = [
            r"$\hat{\boldsymbol{x}}$",
            r"$\hat{\boldsymbol{y}}$",
            r"$\hat{\boldsymbol{z}}$",
        ]

        def plot_method(plot_error: bool):
            ## position stage
            fig, ax = plt.subplots(3, 1, num=4, sharex=True, figsize=(FIG_L, FIG_A))

            for i in range(3):
                lin_width = 3
                ax[i].set_ylabel(y_axis_name[i])
                for j in [
                    "ls_translational",
                    "ls_translational_angular_reduced",
                    "ls_translational_angular_full",
                ]:
                    if plot_error:
                        ax[i].set_yscale("log")
                        ax[i].set_ylim(bottom=10 ** (-20), top=10 ** (-8))
                        ax[i].plot(
                            fff["t"][:],
                            np.abs(
                                ff["ls_solution/" + j][i + 1, :]
                                - ff["true_relative_position"][i, :]
                            ),
                            lw=lin_width,
                            label=ff["ls_solution/" + j].attrs["method_name"],
                        )
                    else:
                        # ax[i].set_ylim(bottom=10 ** (-12), top=10 ** (-6))
                        ax[i].plot(
                            fff["t"][:],
                            1e9 * (ff["ls_solution/" + j][i + 1, :]),
                            lw=lin_width,
                            label=ff["ls_solution/" + j].attrs["method_name"],
                        )
                    lin_width *= 0.5
            fig.supxlabel(r"Tempo $[\unit{\s}]$")
            ax[0].legend()
            if plot_error:
                fig.supylabel(
                    r"$\left|{}^{\mathcal{B}}\mathbf{r}_{m}-{}^{\mathcal{B}}\tilde{\mathbf{r}}_{m}\right|$  $[\unit{\m}]$"
                )
                error_name = "error_"
            else:
                fig.supylabel(r"$\tilde{\mathbf{r}}_{m}$  $[\unit{\nm}]$")
                error_name = ""
            if not ONLY_SHOW:
                plt.savefig(
                    TESE_FOLDER
                    + error_name
                    + "estimated_relative_position_"
                    + TEST_NAME
                    + "."
                    + SAVE_FORMAT,
                    format=SAVE_FORMAT,
                )
                plt.close(fig=4)
            else:
                plt.show()
            # attitude stage
            fig, ax = plt.subplots(3, 1, num=5, sharex=True, figsize=(FIG_L, FIG_A))
            ax[0].set_ylabel(r"$q_{e,1}$")
            ax[1].set_ylabel(r"$q_{e,2}$")
            ax[2].set_ylabel(r"$q_{e,3}$")
            if plot_error:
                for i in range(1, 4):
                    ax[i - 1].set_yscale("log")
                    # ax[0].set_yscale("linear")
                    ax[i - 1].set_ylim(bottom=10 ** (-15), top=10 ** (-6))
                    lin_width = 3
                    for j in [
                        "ls_translational_angular_reduced_attitude_error",
                        "ls_translational_angular_full_attitude_error",
                    ]:

                        ax[i - 1].plot(
                            fff["t"][:],
                            np.abs(ff["ls_solution/" + j][i, :]),
                            lw=lin_width,
                            label=ff["ls_solution/" + j].attrs["method_name"],
                        )
                        lin_width = 0.5
            else:
                for i in range(1, 4):
                    # ax[i - 1].set_yscale("log")
                    ax[0].set_yscale("linear")
                    # ax[i - 1].set_ylim(bottom=10 ** (-15), top=10 ** (-6))
                    lin_width = 3
                    for j in [
                        "ls_translational_angular_reduced",
                        "ls_translational_angular_full",
                    ]:

                        ax[i - 1].plot(
                            fff["t"][:],
                            (ff["ls_solution/" + j][i, :]),
                            lw=lin_width,
                            label=ff["ls_solution/" + j].attrs["method_name"],
                        )
                        lin_width = 0.5
            if plot_error:
                fig.supylabel(r"Erro da orientação relativa")
            else:
                fig.supylabel(r"Orientação relativa")
            fig.supxlabel(r"Tempo $[\unit{\s}]$")
            ax[2].legend()
            if not ONLY_SHOW:
                plt.savefig(
                    TESE_FOLDER
                    + error_name
                    + "estimated_relative_orientation_"
                    + TEST_NAME
                    + "."
                    + SAVE_FORMAT,
                    format=SAVE_FORMAT,
                )
                plt.close(fig=5)
            else:
                plt.show()

        plot_method(plot_error=False)
        plot_method(plot_error=True)
        f.close()


def main(_test_name, _h5py_file_name):
    global TEST_NAME, H5PY_FILE_NAME
    TEST_NAME = _test_name
    H5PY_FILE_NAME = _h5py_file_name
    # recover_acceleration()
    # plot_inertial_states()
    # plot_recover_acceleration()
    plot_error_recover_acceleration()
    # plot_recover_angular_acceleration()
    # plot_error_recover_angular_acceleration()
    # plot_ls_solution()
    # plot_deformations()
    # plot_inertial_states_difference_one_column()
    # plot_accelerations()

if __name__ == "__main__":
    main(_test_name=TEST_NAME, _h5py_file_name=H5PY_FILE_NAME)
