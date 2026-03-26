import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from modeling.accel_model import accel_model_euler_poincare


class closed_form_estimator:
    """
    closed-form estimator for the optical accelerometer.
    implements the algebraic, non-iterative least squares for embedded systems.
    """

    def __init__(self, model: accel_model_euler_poincare, active_fibers: list = None):
        self.model = model
        self.n_fibers = 12

        # if no active fibers are specified, assume all 12 are working
        self.active_fibers = (
            active_fibers if active_fibers is not None else list(range(self.n_fibers))
        )
        self.num_active = len(self.active_fibers)
        # optical and thermal properties of silica fiber
        self.p_e = 0.22  # effective photoelastic coefficient
        self.alpha = 0.55e-6  # thermal expansion coefficient (1/°C)
        self.zeta = 8.60e-6  # thermo-optic coefficient (1/°C)

        # thermal-kinematic sensitivity constant (K_T)
        self.k_t = (self.alpha + self.zeta) / (1.0 - self.p_e)

        # pre-allocated regression matrices for active fibers only
        self.h_trans_matrix = np.zeros((self.num_active, 3))  # 3-DOF
        self.h_comp_matrix = np.zeros((self.num_active, 6))  # 6-DOF
        self.h_term_matrix = np.zeros((self.num_active, 7))  # 7-DOF

        # nominal lengths at static equilibrium
        self.l_0 = np.zeros(self.n_fibers)

        # compute static matrices offline
        self._precompute_static_matrices()

    def _precompute_static_matrices(self):
        """
        precomputes the constant regression matrices and their pseudoinverses
        based exclusively on the static anchor coordinates.
        this mimics the offline calibration phase of the embedded sensor.
        """
        for idx, j in enumerate(self.active_fibers):
            m_j = self.model.m_m[j]
            b_j = self.model.b_b[j]

            # nominal vector d_j at static equilibrium
            d_j = m_j - b_j
            self.l_0[j] = np.linalg.norm(d_j)

            # 1. translational mmq regression matrix (h_trans) -> 3 columns
            self.h_trans_matrix[idx, :] = d_j / self.l_0[j]

            # 2. complete linear mmq regression matrix (h_comp) -> 6 columns
            torque_vec = np.cross(b_j, m_j)
            self.h_comp_matrix[idx, 0:3] = d_j / self.l_0[j]
            self.h_comp_matrix[idx, 3:6] = torque_vec / self.l_0[j]

            # 3. thermokinematic mmq regression matrix (h_term) -> 7 columns
            self.h_term_matrix[idx, 0:3] = d_j / self.l_0[j]
            self.h_term_matrix[idx, 3:6] = torque_vec / self.l_0[j]
            self.h_term_matrix[idx, 6] = self.l_0[j] * self.k_t

        # compute and store pseudoinverses offline to save clock cycles in real-time
        # H^T * H dimension will be 3x3, 6x6, and 7x7 respectively.
        self.h_trans_pinv = np.linalg.pinv(self.h_trans_matrix)
        self.h_comp_pinv = np.linalg.pinv(self.h_comp_matrix)
        self.h_term_pinv = np.linalg.pinv(self.h_term_matrix)

    def estimate_translational(self, fiber_lengths: np.ndarray) -> np.ndarray:
        """
        estimates the 3d relative position (3-DOF). requires at least 3 active fibers.
        """
        delta_l = np.zeros(self.num_active)
        for idx, j in enumerate(self.active_fibers):
            delta_l[idx] = fiber_lengths[j] - self.l_0[j]

        return self.h_trans_pinv @ delta_l

    def estimate_complete(self, fiber_lengths: np.ndarray) -> np.ndarray:
        """
        estimates the full 6-dof kinematic state (6-DOF). requires at least 6 active fibers.
        """
        delta_l = np.zeros(self.num_active)
        for idx, j in enumerate(self.active_fibers):
            delta_l[idx] = fiber_lengths[j] - self.l_0[j]

        return self.h_comp_pinv @ delta_l

    def estimate_thermokinematic(self, fiber_lengths: np.ndarray) -> np.ndarray:
        """
        estimates the full 6-dof kinematic state + common-mode temperature gradient (7-DOF).
        requires at least 7 active fibers.
        """
        delta_l = np.zeros(self.num_active)
        for idx, j in enumerate(self.active_fibers):
            delta_l[idx] = fiber_lengths[j] - self.l_0[j]

        return self.h_term_pinv @ delta_l


def solve_inverse_problem_closed_form(mmq_mode:int):
    """
    estimates relative pose and recovers base accelerations using quasi-static assumption,
    driven by closed-form algebraic estimations.
    """
    df_sim = pd.read_csv("./modeling/data/simulation_output_perturbed.csv")
    time_vector = df_sim["time"].values

    length_columns = [f"fiber_{j+1}_length" for j in range(12)]
    lengths_data = df_sim[length_columns].values

    model = accel_model_euler_poincare()

    # --- configuration parameters ---
    # active_fibers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    active_fibers = [0, 1,  4, 5,   8, 9,11]
    # active_fibers = [0, 2, 4, 7, 8,10, 11]

    # 1: translational (3x3), 2: complete kinematic (6x6), 3: thermokinematic (7x7)

    # --------------------------------

    estimator = closed_form_estimator(model, active_fibers)

    num_steps = len(time_vector)
    estimated_f_b = np.zeros((num_steps, 3))
    estimated_dot_omega_b = np.zeros((num_steps, 3))
    estimated_r_rel = np.zeros((num_steps, 3))
    estimated_euler = np.zeros((num_steps, 3))
    estimated_dT = np.zeros(num_steps)
    estimated_fiber_lengths = np.zeros((num_steps, 12))

    print(f"solving inverse problem via closed-form estimation (mode {mmq_mode})...")

    for i in range(num_steps):
        measured_lengths = lengths_data[i, :]

        # 1. state estimation via algebraic pseudoinverse
        if mmq_mode == 1:
            r_rel_est = estimator.estimate_translational(measured_lengths)
            euler_est = np.zeros(3)
            dT_est = 0.0
            filename = "./modeling/data/inverse_output_closed_form_translacional.csv"

        elif mmq_mode == 2:
            x_est = estimator.estimate_complete(measured_lengths)
            r_rel_est = x_est[0:3]
            euler_est = x_est[3:6]
            dT_est = 0.0
            filename = (
                "./modeling/data/inverse_output_closed_form_translacional_angular.csv"
            )

        elif mmq_mode == 3:
            x_est = estimator.estimate_thermokinematic(measured_lengths)
            r_rel_est = x_est[0:3]
            euler_est = x_est[3:6]
            dT_est = x_est[6]
            filename = "./modeling/data/inverse_output_closed_form_translacional_angular_thermal.csv"

        # recreate the rotation matrix exactly as formulated in the linear model: R = I + skew(theta)
        theta_x, theta_y, theta_z = euler_est
        skew_theta = np.array(
            [
                [0.0, -theta_z, theta_y],
                [theta_z, 0.0, -theta_x],
                [-theta_y, theta_x, 0.0],
            ],
            dtype=np.float64,
        )

        # r_m_b_est = np.eye(3, dtype=np.float64) + skew_theta
        r_m_b_est = Rotation.from_euler('xyz',euler_est).as_matrix()

        estimated_r_rel[i, :] = r_rel_est
        estimated_euler[i, :] = euler_est
        estimated_dT[i] = dT_est

        # 2. algebraic inversion (quasi-static assumption) to recover accelerations
        f_b, dot_omega_b, fiber_lengths_model = model.inverse_dynamics_quasi_static(
            r_rel_est, r_m_b_est
        )

        estimated_f_b[i, :] = f_b
        estimated_dot_omega_b[i, :] = dot_omega_b
        estimated_fiber_lengths[i, :] = fiber_lengths_model

    # export data to csv
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
            "dT_true": df_sim["dT_true"],
        }
    )
    if mmq_mode ==3:
        df_inv["dT_est"] =  estimated_dT

    for i in range(12):
        df_inv[f"fiber_{i+1}_length_est"] = estimated_fiber_lengths[:, i]

    df_inv.to_csv(filename, index=False)
    print(f"closed-form inverse problem solved. estimations saved to {filename}.")


if __name__ == "__main__":
    solve_inverse_problem_closed_form(mmq_mode=1)
    solve_inverse_problem_closed_form(mmq_mode=2)
    solve_inverse_problem_closed_form(mmq_mode=3)
