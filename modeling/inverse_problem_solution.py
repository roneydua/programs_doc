#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   inverse_problem_solution.py
@Time    :   2023/10/19 15:15:52
@Author  :   Roney D. Silva
@Contact :   roneyddasilva@gmail.com
"""
import locale
import os
from turtle import width

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from tqdm import tqdm

import common_functions.quaternion_functions as fq
import common_functions.generic_functions as gf
from modeling import math_model_accel
from modeling.math_model_accel import InverseProblem, SimpleSolution

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
# plt.style.use("default")
plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 1152
FIG_L = 6.29
FIG_A = (60.0) / 25.4

# TEST_NAME = "translational_movement"
TEST_NAME = "complete_movement"
# TEST_NAME = "angular_movement"
H5PY_FILE_NAME = "modeling_data.hdf5"


# ONLY_SHOW = True
ONLY_SHOW = False

if ONLY_SHOW:
    plt.rcParams["figure.dpi"] = 288

graphics = [
    # "one_fiber",
    # "differential_aligned",
    "ls_translational",
    "ls_translational_angular_reduced",
    "ls_translational_angular_full",
    "differential_cross",
]


def set_precision_of_graphcis(_d):
    return np.float16(_d)


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
    if _ff.attrs["dt"] < 1e-4:
        new_dt = 1e-4
    else:
        new_dt = _ff.attrs["dt"]
    _delta_n = int(new_dt / (_ff["t"][1] - _ff["t"][0]))
    _time_size = int(_ff["t"].size * (_ff["t"][1] - _ff["t"][0]) / new_dt)
    print(_delta_n, _time_size)
    if _time_size < 1:
        print("_time_size muito pequeno")
        _delta_n = 1
        _time_size = _ff["t"].size

    fibers_with_length_info = np.array([1, 5, 9, 11])
    ip_trans = InverseProblem(fibers_with_info=np.array([1, 5, 9, 11]),density=_ff.attrs["density"],fiber_length=_ff.attrs['fiber_length'])
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
        estimation="reduced",density=_ff.attrs["density"],fiber_length=_ff.attrs['fiber_length']
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
    idx_new = 0
    for idx in tqdm(range(0, int(_ff["t"].size) - 1, _delta_n)):
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
    f.close()


def plot_recover_acceleration():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(1)

    dt_sim = ff["t"][1] - ff["t"][0]
    dt_rec = fff["t"][1] - fff["t"][0]
    n_factor = int(dt_rec / dt_sim)
    index_vec = np.arange(0, ff["t"].size, n_factor)

    fig, ax = plt.subplots(3, 1, num=1, sharex=True, figsize=(FIG_L, FIG_A))
    y_axis_name = [
        r"${}^{\mathcal{B}}{r}_{x}$",
        r"${}^{\mathcal{B}}{r}_{y}$",
        r"${}^{\mathcal{B}}{r}_{z}$",
    ]
    for i in range(3):
        # ax[i].set_yscale("log")
        for j in graphics:
            if j in fff.keys():
                # print(j)
                # ax[i].plot(
                #     fff["t"][:] * 1e3,
                #     np.abs(fff[j][i, :] - np.take(ff["true_accel_b"][i, :], index_vec)),
                #     label=fff[j].attrs["method_name"],
                # )
                ax[i].plot(
                    fff["t"][:] * 1e3,
                    fff[j][i, :],
                    label=fff[j].attrs["method_name"],
                )
        ax[i].plot(ff["t"][:] * 1e3, ff["true_accel_b"][i, :], label="Referência")
        ax[i].set_ylabel(y_axis_name[i])
    # ax[0].set_xlim(left=90,right=100)

    # ax[2].legend()
    fig.supylabel(r"Aceleração $[\unit{\meter\per\second\squared}]$")
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    ax[0].legend(
        ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1, 1), mode="expand"
    )
    # fig.legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"recover_translational_acceleration_"
            + TEST_NAME
            + ".pdf",
            format="pdf",
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
    y_axis_name = [r"$\dot{\omega}_{x}$", r"$\dot{\omega}_{y}$", r"$\dot{\omega}_{z}$"]
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        num=1,
        sharex="col",
        figsize=(FIG_L, FIG_A),
        # gridspec_kw=dict(width_ratios=[1, 0.5], height_ratios=[1, 1, 1]),
    )
    xmin = 600
    xmax = 602
    ymin = 0
    ymax = 0
    for i in range(3):
        ax[i].set_ylabel(y_axis_name[i])
        # lw = 4
        for j in ["ls_translational_angular_reduced", "ls_translational_angular_full"]:
            ax[i].plot(
                fff["t"][:] * 1e3,
                fff[j][i + 3, :],
                label=fff[j].attrs["method_name"],
                # lw=lw,
            )
            ax[i, 1].plot(
                fff["t"][:] * 1e3,
                fff[j][i + 3, :],
                label=fff[j].attrs["method_name"],
                # lw=lw,
            )
            # sub_idx = np.argwhere(
            #     (fff["t"][:] * 1e3 >= xmin) & (fff["t"][:] * 1e3 <= xmax)
            # )
            sub_idx = gf.find_index_of_x_span(xmin,xmax,fff["t"][:] * 1e3)
            ymin = np.min(fff[j][i + 3, :][sub_idx[0]:sub_idx[1]])
            ymax = np.max(fff[j][i + 3, :][sub_idx[0]:sub_idx[1]])
            ax[i, 1].set_ylim(ymin, ymax)
        ax[i, 0].plot(
            ff["t"][:] * 1e3,
            ff["true_angular_acceleration_b"][i, :],
            label="Referência",
        )
        ax[i, 1].plot(
            ff["t"][:] * 1e3,
            ff["true_angular_acceleration_b"][i, :],
            label="Referência",
        )
        ax[i, 1].set_ylim(ymin, ymax)
    ax[0, 0].legend()
    ax[2, 1].set_xlim(xmin, xmax)
    fig.supylabel(r"Aceleração angular $[\unit{\radian\per\second\squared}]$")
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    # fig.legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"recover_angular_acceleration_" + TEST_NAME + ".pdf",
            format="pdf",
        )
        plt.close(fig=1)
    else:
        plt.show()
    f.close()


def plot_defromations():
    
    f = h5py.File(H5PY_FILE_NAME, "r")
    accel = math_model_accel.AccelModelInertialFrame()
    ff = f[TEST_NAME]
    plt.close(2)
    fig, ax = plt.subplots(3, 4, num=2, sharex=True, figsize=(FIG_L, FIG_A))

    for i, _ax in enumerate(ax.flat):
        _ax.plot(
            1e3 * ff["t"][...],
            1e6 * (ff["fiber_len"][i, :] / ff.attrs["fiber_length"] - 1.0),label=accel.leg[i]
        )
        _ax.legend()
        _ax.set_title(r"$\text{FBG}_{" + str(i + 1) + "}$")
        # _ax.set_xlim(50,55)
        # _ax.set_ylim(-50, 50)

    fig.supylabel(r"Deformação $\frac{\ell-\ell_0}{\ell_0}\times{10}^{6}$")
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    # fig.legend()

    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"deformations_" + TEST_NAME + ".pdf", format="pdf"
        )
        plt.close(fig=3)
    else:
        plt.show()
    f.close()


def plot_inertial_states():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    plt.close(3)
    fig, ax = plt.subplots(4, 3, num=3, sharex=True, figsize=(FIG_L, FIG_A * 1.75))
    ind_x = [0, 6, 12, 20]
    ax[0][0].set_ylabel(r"${}^\mathcal{I}\dot{\mathbf{r}}[\si{\meter\per\second}]$")
    ax[1][0].set_ylabel(r"${}^\mathcal{I}\mathbf{r}[\si{\meter}]$")
    ax[2][0].set_ylabel(r"$\mathbf{q}$")
    ax[3][0].set_ylabel(r"$\boldsymbol{\omega}[\si{\radian\per\second}]$")
    ax[0, 0].set_title("x")
    ax[0, 1].set_title("y")
    ax[0, 2].set_title("z")
    for lin in range(4):
        if lin == 2:
            # pass
            for col in range(3):
                ax[lin, col].plot(
                    1e3 * ff["t"][:],
                    set_precision_of_graphcis(ff["x"][ind_x[lin] + col + 5, :]),
                    label=r"$q_{" + str(col + 1) + "}$ Massa sísmica",
                    lw=1.0,
                )
                ax[lin, col].plot(
                    1e3 * ff["t"][:],
                    set_precision_of_graphcis(ff["x"][ind_x[lin] + col + 1, :]),
                    label=r"$q_{" + str(col + 1) + "}$ Base",
                )
            # ax[lin, 0].plot(
            #     1e3 * ff["t"][:],
            #     set_precision_of_graphcis(ff["x"][ind_x[lin] + 4, :]),
            #     label=r"$q_{0}$ Massa sísmica",
            # )
            # ax[lin, 0].plot(
            #     1e3 * ff["t"][:],
            #     set_precision_of_graphcis(ff["x"][ind_x[lin], :]),
            #     label=r"$q_{0}$ Base",
            # )
            # ax[lin, 0].legend()
        else:
            for col in range(3):
                ax[lin, col].plot(
                    1e3 * ff["t"][:],
                    set_precision_of_graphcis(ff["x"][ind_x[lin] + col + 3, :]),
                    lw=1.0,
                )
                ax[lin, col].plot(
                    1e3 * ff["t"][:], set_precision_of_graphcis(ff["x"][ind_x[lin] + col, :])
                )
    ax[0, 2].legend(["Massa sísmica", "Base"])
    # ax[0,1].legend(["Massa sísmica", "Base"],ncols=2, loc="lower left", bbox_to_anchor=(-.50, 1.5,3,1.5),mode="expand")
    # fig.supylabel(r'Aceleração $[\unit{\meter\per\second\squared}]$')
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    # fig.legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"all_states_" + TEST_NAME + ".pdf", format="pdf"
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
            TESE_FOLDER+"length_of_fo_" + TEST_NAME + ".pdf", format="pdf"
        )
        plt.close(fig=2)
    else:
        plt.show()
    f.close()


def plot_ls_solution():
    f = h5py.File(H5PY_FILE_NAME, "r")
    ff = f[TEST_NAME]
    fff = ff["accel_recover"]
    plt.close(4)
    fig, ax = plt.subplots(3, 1, num=4, sharex=True, figsize=(FIG_L, FIG_A))
    dt_sim = ff["t"][1] - ff["t"][0]
    dt_rec = fff["t"][1] - fff["t"][0]
    n_factor = int(dt_rec / dt_sim)
    index_vec = np.arange(0, ff["t"].size, n_factor)
    for i in range(3):
        # ax[i].plot(
        #     1e3 * ff["t"][:],
        #     1e6 * ff["true_relative_position"][i, :],
        #     label="Referência",
        # )
        ax[i].set_yscale("log")
        for j in ff["ls_solution"].keys():
            ax[i].plot(
                1e3 * fff["t"][:],
                1e6
                * (
                    ff["ls_solution/" + j][i + 1, :]
                    - np.take(ff["true_relative_position"][i, :-1], index_vec)
                ),
                label=ff["ls_solution/" + j].attrs["method_name"],
            )

    fig.supylabel(
        r"Erro da posição relativa ${}^{\mathcal{B}}\mathbf{r}_{m}$ $[\unit{\um}]$"
    )
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    ax[0].legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"estimated_relative_position_" + TEST_NAME + ".pdf",
            format="pdf",
        )
        plt.close(fig=4)
    else:
        plt.show()

    plt.close(fig=5)
    fig, ax = plt.subplots(3, 1, num=5, sharex=True, figsize=(FIG_L, FIG_A))
    ax[0].set_ylabel(r"$q_x$")
    ax[1].set_ylabel(r"$q_y$")
    ax[2].set_ylabel(r"$q_z$")
    for i in range(3):
        # ax[i].plot(
        #     1e3 * ff["t"][:],
        #     ff["true_relative_orientation"][i, :],
        #     label="Referência",
        # )
        ax[i].set_yscale("log")
        for j in ["ls_translational_angular_reduced", "ls_translational_angular_full"]:
            # if i == 0:
            # q0 = ff["ls_solution/" + j][-4, :]
            q = np.zeros((4, fff["t"].size))
            q_error = np.zeros((4, fff["t"].size))
            for k in range(fff["t"].size):
                q[0, k] = np.sqrt(
                    1.0
                    - ff["ls_solution/" + j][-3, k] ** 2
                    - ff["ls_solution/" + j][-2, k] ** 2
                    - ff["ls_solution/" + j][-1, k] ** 2
                )
                q[1, k] = ff["ls_solution/" + j][-3, k]
                q[2, k] = ff["ls_solution/" + j][-2, k]
                q[3, k] = ff["ls_solution/" + j][-1, k]
                q_error[:, k] = fq.mult_quat(
                    ff["true_relative_orientation"][:, k * n_factor],
                    fq.conj(q[:, k]),
                )
            # ax[i].plot(
            #     1e3 * fff["t"][:],
            #     q_error[0, :],
            #     label=ff["ls_solution/" + j].attrs["method_name"],
            # )
            else:
                ax[i].plot(
                    1e3 * fff["t"][:],
                    q_error[i + 1, :],
                    label=ff["ls_solution/" + j].attrs["method_name"],
                )

    fig.supylabel(
        r"Erro da orientação relativa $\,_{\mathcal{M}}^{\mathcal{B}}\mathbf{q}$"
    )
    fig.supxlabel(r"Tempo $[\unit{\ms}]$")
    ax[0].legend()
    if not ONLY_SHOW:
        plt.savefig(
            TESE_FOLDER+"estimated_relative_orientation_"
            + TEST_NAME
            + ".pdf",
            format="pdf",
        )
        plt.close(fig=5)
    else:
        plt.show()
    f.close()


if __name__ == "__main__":
    # recover_acceleration()
    # plot_recover_acceleration()
    # plot_inertial_states()
    # plot_recover_angular_acceleration()
    # plot_ls_solution()
    plot_defromations()

    duration = 1  # seconds
    freq = 100  # Hz
    os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))
    os.system('spd-say "Our program has finished"')


# f = h5py.File(H5PY_FILE_NAME, "a")
# ff = f[TEST_NAME]
# for i in tqdm(range(ff["true_relative_position"][0, :].shape[0])):
#     ff["true_relative_position"][:, i] = (
#         fq.rotationMatrix(ff["x"][12:16, i]).T @ ff["true_relative_position"][:, i]
#     )
# f.close()
