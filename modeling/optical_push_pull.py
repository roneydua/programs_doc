import numpy as np
import pandas as pd
from modeling.accel_model import accel_model_euler_poincare

class optical_push_pull_estimator:
    """
    estimator for the optical push-pull experimental setup.
    uses differential optical measurements (double reflection) to estimate
    purely translational relative position (3-DOF), intrinsically rejecting
    common-mode thermal noise at the hardware level.
    """
    def __init__(self, model: accel_model_euler_poincare, pairs: list):
        self.model = model
        self.pairs = pairs
        self.num_pairs = len(pairs)
        self.n_fibers = 12

        # regression matrix for the differential measurement
        self.h_diff_matrix = np.zeros((self.num_pairs, 3))

        # nominal lengths at static equilibrium for all fibers
        self.l_0 = np.zeros(self.n_fibers)

        self._precompute_static_matrices()

    def _precompute_static_matrices(self):
        """
        precomputes the differential regression matrix by calculating the
        individual translational matrices (H_trans) and subtracting the pairs.
        """
        h_individual = np.zeros((self.n_fibers, 3))

        # 1. calculate individual H_trans for all 12 fibers
        for j in range(self.n_fibers):
            m_j = self.model.m_m[j]
            b_j = self.model.b_b[j]

            d_j = m_j - b_j
            self.l_0[j] = np.linalg.norm(d_j)

            # H_trans_j = (1 / l_0) * d_j^T
            h_individual[j, :] = d_j / self.l_0[j]

        # 2. compute the differential matrix H_diff for each pair
        for idx, pair in enumerate(self.pairs):
            fiber_a, fiber_b = pair
            # H_diff = H_a - H_b
            self.h_diff_matrix[idx, :] = h_individual[fiber_a, :] - h_individual[fiber_b, :]

        # 3. precompute pseudoinverse
        # if using 3 orthogonal pairs, this is a 3x3 invertible matrix
        self.h_diff_pinv = np.linalg.pinv(self.h_diff_matrix)

    def estimate_translational(self, fiber_lengths_corrupted: np.ndarray) -> np.ndarray:
        """
        estimates the 3d relative position from the raw corrupted lengths.
        this simulates the physical double reflection: the lengths are subtracted
        optically, canceling out the thermal drift before estimation.
        """
        delta_l_diff = np.zeros(self.num_pairs)

        for idx, pair in enumerate(self.pairs):
            fiber_a, fiber_b = pair

            # extract the individual apparent variations (including thermal noise)
            delta_l_a = fiber_lengths_corrupted[fiber_a] - self.l_0[fiber_a]
            delta_l_b = fiber_lengths_corrupted[fiber_b] - self.l_0[fiber_b]

            # the optical hardware naturally outputs this difference
            delta_l_diff[idx] = delta_l_a - delta_l_b

        # matrix-vector multiplication to get [r_x, r_y, r_z]
        r_rel = self.h_diff_pinv @ delta_l_diff

        return r_rel


def solve_inverse_problem_optical_push_pull(push_pull_pairs:list, name_save:str):
    """
    recovers the specific force using the experimental optical push-pull setup.
    """
    # we load the PERTURBED data to prove that the optical setup rejects temperature
    input_file = "./modeling/data/simulation_output_perturbed.csv"
    try:
        df_sim = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please run add_thermal_noise.py first.")
        return

    time_vector = df_sim["time"].values
    length_columns = [f"fiber_{j+1}_length" for j in range(12)]
    lengths_data = df_sim[length_columns].values

    model = accel_model_euler_poincare()

    # --- experimental configuration ---
    # physical pairs interrogated by double reflection
    #
    # ----------------------------------

    estimator = optical_push_pull_estimator(model, push_pull_pairs)

    num_steps = len(time_vector)
    estimated_f_b = np.zeros((num_steps, 3))
    estimated_dot_omega_b = np.zeros((num_steps, 3)) # will be zero for 3-DOF
    estimated_r_rel = np.zeros((num_steps, 3))
    estimated_fiber_lengths = np.zeros((num_steps, 12))

    print("solving inverse problem via optical push-pull (hardware thermal rejection)...")

    for i in range(num_steps):
        measured_lengths = lengths_data[i, :]

        # 1. state estimation (Translational 3-DOF)
        r_rel_est = estimator.estimate_translational(measured_lengths)
        r_m_b_est = np.eye(3) # no rotation estimated in 3-pair setup

        estimated_r_rel[i, :] = r_rel_est

        # 2. algebraic inversion (quasi-static assumption)
        f_b, dot_omega_b, fibers_lengths = model.inverse_dynamics_quasi_static(r_rel_est, r_m_b_est)

        estimated_f_b[i, :] = f_b
        estimated_dot_omega_b[i, :] = dot_omega_b
        estimated_fiber_lengths[i, :] = fibers_lengths
    # export data to csv
    df_inv = pd.DataFrame({
        "time": time_vector,
        "f_b_est_x": estimated_f_b[:, 0],
        "f_b_est_y": estimated_f_b[:, 1],
        "f_b_est_z": estimated_f_b[:, 2],
        "r_rel_est_x": estimated_r_rel[:, 0],
        "r_rel_est_y": estimated_r_rel[:, 1],
        "r_rel_est_z": estimated_r_rel[:, 2],
        # "dot_omega_b_est_x": estimated_dot_omega_b[:, 0],
        # "dot_omega_b_est_y": estimated_dot_omega_b[:, 1],
        # "dot_omega_b_est_z": estimated_dot_omega_b[:, 2],
        # euler angles and dT are not estimated in this 3-pair configuration
        # "euler_est_x": np.zeros(num_steps),
        # "euler_est_y": np.zeros(num_steps),
        # "euler_est_z": np.zeros(num_steps),
    })
    for i in range(12):
        df_inv[f"fiber_{i+1}_length_est"] = estimated_fiber_lengths[:, i]
    # output_filename = "modeling/data/inverse_output_optical_push_pull.csv"
    df_inv.to_csv(name_save, index=False)
    print(f"optical push-pull problem solved. estimations saved to {name_save}.")


if __name__ == "__main__":
    solve_inverse_problem_optical_push_pull(push_pull_pairs=[[0, 3], [4, 7], [8, 11]], name_save="modeling/data/inverse_output_optical_push_pull_cruzed.csv")
    solve_inverse_problem_optical_push_pull(push_pull_pairs=[[0, 2], [4, 6], [8, 10]], name_save="modeling/data/inverse_output_optical_push_pull_aligned.csv")
