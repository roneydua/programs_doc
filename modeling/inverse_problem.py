import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from modeling.accel_model import (
    accel_model_euler_poincare,
)  # ajuste o path do import conforme seu projeto


def objective_function(
    x: np.ndarray,
    measured_lengths: np.ndarray,
    active_fibers: np.ndarray,
    model: accel_model_euler_poincare,
    estimate_rotation: bool,
):
    """
    computes the residual between measured fiber lengths and the geometric model.
    """
    r_rel_b = x[0:3]

    if estimate_rotation:
        rot_vec = x[3:6]
        r_m_b = Rotation.from_rotvec(rot_vec).as_matrix()
    else:
        r_m_b = np.eye(3)

    residuals = []
    for j in active_fibers:
        fiber_vec_b = r_rel_b + r_m_b @ model.m_m[j] - model.b_b[j]
        l_model = np.linalg.norm(fiber_vec_b)
        residuals.append(measured_lengths[j] - l_model)

    return np.array(residuals) * 1e6


def solve_inverse_problem():
    """
    estimates relative pose and recovers base accelerations using quasi-static assumption.
    reads inputs from csv and saves the estimation to another csv for analysis.
    """
    # load simulation data
    df_sim = pd.read_csv("simulation_output.csv")

    time_vector = df_sim["time"].values

    # extract the 12 fiber lengths into a numpy array
    length_columns = [f"fiber_{j+1}_length" for j in range(12)]
    lengths_data = df_sim[length_columns].values

    model = accel_model_euler_poincare()

    # configuration parameters
    active_fibers = [
        0,
        # 1,
        2,
        # 3,
        4,
        # 5,
        6,
        # 7,
        8,
        # 9,
        10,
        11,
    ]  # subset of 7 fibers (0-indexed)
    estimate_rotation = True  # toggle to false for translation only

    num_steps = len(time_vector)
    estimated_f_b = np.zeros((num_steps, 3))

    estimated_dot_omega_b = np.zeros((num_steps, 3))
    estimated_r_rel = np.zeros((num_steps, 3))
    estimated_euler = np.zeros((num_steps, 3))
    # initial guess (translation + rotation vector)
    x0 = np.zeros(6) if estimate_rotation else np.zeros(3)

    print("solving inverse problem via nonlinear least squares...")
    for i in range(num_steps):
        measured_lengths = lengths_data[i, :]

        # least squares pose estimation
        res = least_squares(
            objective_function,
            x0,
            args=(measured_lengths, active_fibers, model, estimate_rotation),
            method="lm",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
        )

        r_rel_est = res.x[0:3]
        estimated_r_rel[i, :] = r_rel_est

        if estimate_rotation:
            rot_obj = Rotation.from_rotvec(res.x[3:6])
            r_m_b_est = rot_obj.as_matrix()
            estimated_euler[i, :] = rot_obj.as_euler("xyz", degrees=False)
            x0 = res.x  # warm start for next iteration (speeds up convergence)
        else:
            r_m_b_est = np.eye(3)
            x0 = res.x

        # algebraic inversion (quasi-static assumption)
        f_b, dot_omega_b = model.inverse_dynamics_quasi_static(r_rel_est, r_m_b_est)

        estimated_f_b[i, :] = f_b
        estimated_dot_omega_b[i, :] = dot_omega_b

    # compile estimated results into a pandas dataframe
    df_inv = pd.DataFrame(
        {
            "time": time_vector,
            "f_b_est_x": estimated_f_b[:, 0],
            "f_b_est_y": estimated_f_b[:, 1],
            "f_b_est_z": estimated_f_b[:, 2],
            "dot_omega_b_est_x": estimated_dot_omega_b[:, 0],
            "dot_omega_b_est_y": estimated_dot_omega_b[:, 1],
            "dot_omega_b_est_z": estimated_dot_omega_b[:, 2],
            "r_rel_est_x": estimated_r_rel[:, 0],
            "r_rel_est_y": estimated_r_rel[:, 1],
            "r_rel_est_z": estimated_r_rel[:, 2],
            "euler_est_x": estimated_euler[:, 0],
            "euler_est_y": estimated_euler[:, 1],
            "euler_est_z": estimated_euler[:, 2],
        }
    )

    # export estimation data
    df_inv.to_csv("inverse_output.csv", index=False)
    print("inverse problem solved. estimations saved to inverse_output.csv.")


if __name__ == "__main__":
    solve_inverse_problem()
