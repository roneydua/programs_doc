#!/usr/bin/env python
# from RK import RungeKutta
import locale

import dill
import matplotlib.pyplot as plt
import numpy as np

import common_functions.quaternion_functions as fq

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
plt.style.use("common_functions/roney3.mplstyle")
cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

FIG_L = 8.268 - 4 / 2.54
FIG_A = 0.75 * (90.0) / 25.4

legAccel = ["Real", "Estimated", "MMQ"]
TESE_IMAGE_FOLDER = "./../../images/"


# Load files.
def load_data(filename):
    filehandler = open(filename, "rb")
    return dill.load(filehandler)


def plot_all_states(s, name=""):
    if plt.fignum_exists(1):
        plt.close(1)
    fig, ax = plt.subplots(4, 3, num=1, sharex=True, figsize=(FIG_L, 1.75 * FIG_A))
    ax[3][0].plot(s.t, s.x[16, :].T, lw=2, label="Mass")
    ax[3][0].plot(s.t, s.x[12, :].T, lw=1, label="Base")

    def def_type(a):
        # return a
        return np.float16(a)

    for col in range(3):
        ax[0][col].plot(s.t, def_type(s.x[col + 3, :].T), lw=2, label="Mass")
        ax[0][col].plot(s.t, def_type(s.x[col, :].T), lw=1, label="Base")

        ax[1][col].plot(s.t, def_type(s.x[col + 9, :].T), lw=2, label="Mass")
        ax[1][col].plot(s.t, def_type(s.x[col + 6, :].T), lw=1, label="Base")
        ax[2][col].plot(s.t, def_type(s.x[col + 23, :].T), lw=2, label="Mass")
        ax[2][col].plot(s.t, def_type(s.x[col + 20, :].T), lw=1, label="Base")
        ax[3][col].plot(s.t, def_type(s.x[col + 17, :].T), lw=2, label="Mass")
        ax[3][col].plot(s.t, def_type(s.x[col + 13, :].T), lw=1, label="Base")
        ax[0][col].legend()
        ax[1][col].legend()
        ax[2][col].legend()
        ax[3][col].legend()
    ax[1][0].set_ylabel(r"$\mathbf{r},\si{\meter}$")
    ax[0][0].set_ylabel(r"$\dot{\mathbf{r}},\si{\meter\per\second}$")
    ax[2][0].set_ylabel(r"$\bf{\omega},\si{\radian\per\second}$")
    ax[3][0].set_ylabel(r"$\mathbf{q}$")
    if name != "":
        plt.savefig(TESE_IMAGE_FOLDER + "all_states_" + name + ".pdf", format="pdf")
        plt.close(1)
    return True


def plot_all_states_differences(s, name=""):
    if plt.fignum_exists(1):
        plt.close(1)
    fig, ax = plt.subplots(4, 3, num=1, sharex=True, figsize=(FIG_L, 1.75 * FIG_A))
    ax[3][0].plot(s.t, s.x[16, :].T, lw=2, label="Mass")
    ax[3][0].plot(s.t, s.x[12, :].T, lw=1, label="Base")

    def def_type(a):
        # return a
        return np.float16(a)

    # plot mass minus Base
    for col in range(3):
        ax[0][col].plot(
            # s.t, def_type(s.x[col, :].T - s.x[col + 3, :].T), lw=1, label="Base"
        )

        ax[1][col].plot(s.t, def_type(s.x[col + 6, :].T - s.x[col + 9, :].T), lw=1)
        ax[2][col].plot(
            s.t,
            def_type(s.x[col + 20, :].T - s.x[col + 23, :].T),
            lw=1,
            # label="Base"
        )

        ax[3][col].plot(
            s.t,
            def_type(s.x[col + 13, :].T - s.x[col + 17, :].T),
            lw=1,
            # label="Base"
        )
        ax[0][col].legend()
        ax[1][col].legend()
        ax[2][col].legend()
        ax[3][col].legend()
    ax[1][0].set_ylabel(r"$\mathbf{r},\si{\meter}$")
    ax[0][0].set_ylabel(r"$\dot{\mathbf{r}},\si{\meter\per\second}$")
    ax[2][0].set_ylabel(r"$\bf{\omega},\si{\radian\per\second}$")
    ax[3][0].set_ylabel(r"$\mathbf{q}$")
    if name != "":
        plt.savefig(TESE_IMAGE_FOLDER + "all_states_" + name + ".pdf", format="pdf")
        plt.close(1)
    return True


def plot_deformations(s, name=""):
    if plt.fignum_exists(2):
        plt.close(2)
    fig, ax = plt.subplots(3, 4, num=2, sharex=True, figsize=(FIG_L, FIG_A))
    for col in range(4):
        ax[0][col].plot(
            s.t[:-1], s.deformation[col, :].T, label=s.accel_instance.leg[col]
        )
        ax[0][col].set_yticks(
            [s.deformation[col, :].min(), s.deformation[col, :].max()]
        )
        ax[0][col].legend()
        ax[1][col].plot(
            s.t[:-1], s.deformation[col + 4, :].T, label=s.accel_instance.leg[col + 4]
        )
        ax[1][col].set_yticks(
            [s.deformation[col + 4, :].min(), s.deformation[col + 4, :].max()]
        )
        ax[1][col].legend()
        ax[2][col].plot(
            s.t[:-1], s.deformation[col + 8, :].T, label=s.accel_instance.leg[col + 8]
        )
        ax[2][col].set_yticks(
            [s.deformation[col + 8, :].min(), s.deformation[col + 8, :].max()]
        )
        ax[2][col].legend()
    plt.show()
    fig.supylabel("Deformação, $\\varepsilon$")
    fig.supxlabel(r"Tempo, $\unit{\second}$")
    if name != "":
        plt.savefig(TESE_IMAGE_FOLDER + "deformations_" + name + ".pdf", format="pdf")
        plt.close(2)
    return True


def plot_euler(s, name=""):
    if plt.fignum_exists(3):
        plt.close(3)
    fig, ax = plt.subplots(1, 3, num=3, sharex=True, figsize=(FIG_L, FIG_A))
    euler_b = np.zeros(shape=(3, s.t.size))
    euler_m = np.zeros(shape=(3, s.t.size))
    for i in range(s.t.size):
        euler_b[:, i] = fq.quat2Euler(s.x[16:20, i].T, deg=1)
        euler_m[:, i] = fq.quat2Euler(s.x[12:16, i].T, deg=1)
    for i in range(3):
        ax[i].plot(s.t, euler_b[i, :])
        ax[i].plot(s.t, euler_m[i, :])
    if name != "":
        plt.savefig(TESE_IMAGE_FOLDER + "euler_" + name + ".pdf", format="pdf")
        plt.close(3)
    return True


# plot_all_states(s)


def plot_accelerations(s, name=""):
    if plt.fignum_exists(4):
        plt.close(4)
    fig, ax = plt.subplots(1, 3, num=4, sharex=True, figsize=(FIG_L, FIG_A))
    ax[0].set_ylabel(r"Aceleração, $\si{\meter\per\square\second}$")
    for i in range(3):
        ax[i].plot(s.t, s.true_accel[i, :], label=legAccel[0], lw=2)
        ax[i].plot(s.t, s.recover_accel_simple[i, :], label=legAccel[1], lw=1)
        ax[i].legend()
        ax[i].set_xlabel(r"Tempo, $\si{\second}$")
    plt.show()
    if name != "":
        plt.savefig(TESE_IMAGE_FOLDER + "accelerations_" + name + ".pdf", format="pdf")
        plt.close(4)
    return True
    # ax[2].plot(s.t,s.t*0-9.89)


def plot_accelerations_error(s, name=""):
    if plt.fignum_exists(5):
        plt.close(5)
    fig, ax = plt.subplots(3, 1, num=5, sharex=True, figsize=(FIG_L, FIG_A))
    ylabel = [r"$a_{x}$", r"$a_{y}$", r"$a_{z}$"]
    for i in range(3):
        ax[i].plot(
            s.t,
            1e3 * (s.true_accel[i, :] - s.recover_accel_simple[i, :]),
            label=legAccel[1],
        )
        ax[i].legend()
        ax[i].set_ylabel(ylabel[i] + r"$\si{\milli\meter\per\square\second}$")

    ax[2].set_xlabel(r"Tempo, $\si{\second}$")
    if name != "":
        plt.savefig(
            TESE_IMAGE_FOLDER + "accelerations_errors_" + name + ".pdf", format="pdf"
        )
        plt.close(5)
    return True


if __name__ == "__main__":
    # accel = AccelModelInertialFrame(fibers_with_info=np.array([1, 3, 5, 9], dtype=np.int8), inverse_problem_full=False)

    # s = load_data('data/translation_motions.pickle')
    # # s = s.load_states(name='data/s1DoF_x.pickle')
    # plot_accelerations(s,name='pure_translational')
    # plot_accelerations_error(s,'pure_translational')
    # plot_all_states(s, name='pure_translational')
    # plot_deformations(s, name='pure_translational')
    s = load_data("data/translation_and_rotation_motions.pickle")
    # s = load_data('data/vertical_acceleration.pickle')
    # s = s.load_states(name='data/s1DoF_x.pickle')
    test = "translational_movement"
    plot_accelerations(s, name="translation_and_rotation_motions" + test)
    plot_accelerations_error(s, name="translation_and_rotation_motions" + test)
    plot_all_states(s, name="translation_and_rotation_motions" + test)
    plot_deformations(s, name="translation_and_rotation_motions" + test)
