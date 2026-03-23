import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from modeling.accel_model import accel_model_euler_poincare  # Ajuste o path do import


def objective_function(
    x: np.ndarray,
    measured_lengths: np.ndarray,
    active_fibers: np.ndarray,
    model: accel_model_euler_poincare,
    mmq_mode,
):
    """
    computes the residual between measured fiber lengths and the geometric model
    based on the selected estimation scenario.
    """
    r_rel_b = x[0:3]

    if mmq_mode == 1:
        # scenario 1: translational only (section 3.5.1)
        r_m_b = np.eye(3)

    elif mmq_mode == 2:
        # scenario 2: complete coupled estimation (section 3.5.2)
        rot_vec = x[3:6]
        r_m_b = Rotation.from_rotvec(rot_vec).as_matrix()

    elif mmq_mode == 3:
        # scenario 3: reduced / angularly linearized (section 3.5.3)
        # small angle approximation: R = I + skew(theta)
        theta_x, theta_y, theta_z = x[3:6]
        skew_theta = np.array(
            [
                [0.0, -theta_z, theta_y],
                [theta_z, 0.0, -theta_x],
                [-theta_y, theta_x, 0.0],
            ]
        )
        r_m_b = np.eye(3) + skew_theta

    residuals = []
    for j in active_fibers:
        if mmq_mode == 3:
            # TODO: caso a formulação do seu MMQ reduzido omita algebricamente
            # a interação de r_rel_b com r_m_b, ajuste esta equação vetorial.
            fiber_vec_b = r_rel_b + r_m_b @ model.m_m[j] - model.b_b[j]
        else:
            fiber_vec_b = r_rel_b + r_m_b @ model.m_m[j] - model.b_b[j]

        l_model = np.linalg.norm(fiber_vec_b)
        residuals.append(measured_lengths[j] - l_model)

    # scale residuals to micrometers to prevent premature optimization convergence
    return np.array(residuals) * 1e6


def solve_inverse_problem():
    """
    estimates relative pose and recovers base accelerations using quasi-static assumption.
    evaluates user-defined mmq scenarios.
    """
    df_sim = pd.read_csv("simulation_output.csv")
    time_vector = df_sim["time"].values

    length_columns = [f"fiber_{j+1}_length" for j in range(12)]
    lengths_data = df_sim[length_columns].values

    model = accel_model_euler_poincare()

    # --- configuration parameters ---
    active_fibers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # 1: translational, 2: complete, 3: reduced
    mmq_mode = 3
    # --------------------------------

    num_steps = len(time_vector)
    estimated_f_b = np.zeros((num_steps, 3))
    estimated_dot_omega_b = np.zeros((num_steps, 3))
    estimated_r_rel = np.zeros((num_steps, 3))
    estimated_euler = np.zeros((num_steps, 3))
    estimated_fiber_lengths = np.zeros((num_steps, 12))

    print(f"solving inverse problem via nonlinear least squares (mode {mmq_mode})...")

    for i in range(num_steps):
        measured_lengths = lengths_data[i, :]
        x0 = np.zeros(3) if mmq_mode == 1 else np.zeros(6)

        # least squares pose estimation with strict tolerances
        res = least_squares(
            objective_function,
            x0,
            args=(measured_lengths, active_fibers, model, mmq_mode),
            method="lm",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
        )

        r_rel_est = res.x[0:3]
        estimated_r_rel[i, :] = r_rel_est

        if mmq_mode == 1:
            r_m_b_est = np.eye(3)
        elif mmq_mode == 2:
            rot_obj = Rotation.from_rotvec(res.x[3:6])
            r_m_b_est = rot_obj.as_matrix()
            estimated_euler[i, :] = rot_obj.as_euler("xyz", degrees=False)
        elif mmq_mode == 3:
            # retrieve linearized rotation for analytical consistency
            theta_x, theta_y, theta_z = res.x[3:6]
            skew_theta = np.array(
                [
                    [0.0, -theta_z, theta_y],
                    [theta_z, 0.0, -theta_x],
                    [-theta_y, theta_x, 0.0],
                ],
                dtype=np.float64,
            )
            r_m_b_est = np.eye(3, dtype=np.float64) + skew_theta
            estimated_euler[i, :] = res.x[3:6]  # small angles approx: rotvec ~ euler

        # algebraic inversion (quasi-static assumption)
        f_b, dot_omega_b, fiber_lengths = model.inverse_dynamics_quasi_static(
            r_rel_est, r_m_b_est
        )

        estimated_f_b[i, :] = f_b
        estimated_dot_omega_b[i, :] = dot_omega_b
        estimated_fiber_lengths[i, :] = fiber_lengths

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
    for i in range(12):
        df_inv[f"fiber_{i+1}_length_est"] = estimated_fiber_lengths[:, i]

    df_inv.to_csv("inverse_output.csv", index=False)
    print("inverse problem solved. estimations saved to inverse_output.csv.")


if __name__ == "__main__":
    solve_inverse_problem()
