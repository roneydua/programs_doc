import numpy as np
from numpy.linalg import norm


class accel_model_euler_poincare:

    def __init__(
        self,
        seismic_edge: float = 16.4e-3,
        fiber_diameter: float = 125e-6,
        fiber_length_0: float = 3e-3,
        density: float = 2.6989e3,
        e_modulus: float = 70e9,
    ):
        """
        initialization of the euler-poincare accelerometer model.
        """
        self.density = density
        self.e_modulus = e_modulus
        self.fiber_diameter = fiber_diameter
        self.fiber_length_0 = fiber_length_0
        self.seismic_edge = seismic_edge

        self.optical_fiber_area = 0.25 * np.pi * (self.fiber_diameter**2)
        self.k = (self.e_modulus * self.optical_fiber_area) / self.fiber_length_0

        self.seismic_mass = (self.seismic_edge**3.0) * self.density

        # isometric mass assumption: scalar moment of inertia
        self.i_m = self.seismic_mass * (self.seismic_edge**2) / 6.0

        self.base_sensor_edge = self.seismic_edge + 2.0 * self.fiber_length_0

        _d = self.seismic_edge / 2.0 - 1e-3
        _e = self.seismic_edge / 2.0

        # connections on the seismic mass (m frame)
        self.m_m = np.array(
            [
                [_e, _d, 0.0],
                [_e, -_d, 0.0],
                [-_e, _d, 0.0],
                [-_e, -_d, 0.0],
                [0.0, _e, _d],
                [0.0, _e, -_d],
                [0.0, -_e, _d],
                [0.0, -_e, -_d],
                [_d, 0.0, _e],
                [-_d, 0.0, _e],
                [_d, 0.0, -_e],
                [-_d, 0.0, -_e],
            ]
        )

        _f = self.base_sensor_edge / 2.0

        # connections on the base sensor (b frame)
        self.b_b = np.array(
            [
                [_f, _d, 0.0],
                [_f, -_d, 0.0],
                [-_f, _d, 0.0],
                [-_f, -_d, 0.0],
                [0.0, _f, _d],
                [0.0, _f, -_d],
                [0.0, -_f, _d],
                [0.0, -_f, -_d],
                [_d, 0.0, _f],
                [-_d, 0.0, _f],
                [_d, 0.0, -_f],
                [-_d, 0.0, -_f],
            ]
        )

    def compute_elastic_efforts(self, r_rel_b: np.ndarray, r_m_b: np.ndarray):
        """
        computes elastic forces and moments in the seismic mass frame.
        """
        r_b_m = r_m_b.T
        f_el_m = np.zeros(3)
        m_el_m = np.zeros(3)
        fiber_lengths = np.zeros(12)

        for i in range(12):
            anchor_m = self.m_m[i]
            anchor_b = self.b_b[i]

            # fiber vector in base frame
            fiber_vec_b = r_rel_b + r_m_b @ anchor_m - anchor_b
            l_i = norm(fiber_vec_b)
            fiber_lengths[i] = l_i

            # unit vector in mass frame
            unit_vec_m = r_b_m @ (fiber_vec_b / l_i)

            # elastic force for fiber i in mass frame
            f_i_m = -self.k * (l_i - self.fiber_length_0) * unit_vec_m

            f_el_m += f_i_m
            m_el_m += np.cross(anchor_m, f_i_m)

        return f_el_m, m_el_m, fiber_lengths

    def forward_dynamics(
        self,
        r_rel_b: np.ndarray,
        v_rel_b: np.ndarray,
        r_m_b: np.ndarray,
        omega_rel_m: np.ndarray,
        a_b_b: np.ndarray,
        omega_b_b: np.ndarray,
        dot_omega_b_b: np.ndarray,
        g_b: np.ndarray,
    ):
        """
        computes relative translational and angular accelerations for numerical integration.
        """
        f_el_m, m_el_m,_ = self.compute_elastic_efforts(r_rel_b, r_m_b)
        r_b_m = r_m_b.T

        # specific force of the base
        f_b_b = a_b_b - g_b

        # non-inertial kinematic accelerations
        euler_accel = np.cross(dot_omega_b_b, r_rel_b)
        coriolis_accel = 2.0 * np.cross(omega_b_b, v_rel_b)
        centrifugal_accel = np.cross(omega_b_b, np.cross(omega_b_b, r_rel_b))

        # translational dynamics solving for relative acceleration in base frame
        a_rel_m = (
            (f_el_m / self.seismic_mass)
            - r_b_m @ f_b_b
            - r_b_m @ (euler_accel + coriolis_accel + centrifugal_accel)
        )
        a_rel_b = r_m_b @ a_rel_m

        # rotational dynamics solving for relative angular acceleration in mass frame
        # gyroscopic term is zero due to isometric mass assumption
        dot_omega_rel_m = (
            (m_el_m / self.i_m)
            - r_b_m @ dot_omega_b_b
            + np.cross(omega_rel_m, r_b_m @ omega_b_b)
        )

        return a_rel_b, dot_omega_rel_m

    def inverse_dynamics_quasi_static(self, r_rel_b: np.ndarray, r_m_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        recovers specific force and angular acceleration of the base and fiber lengths assuming quasi-static regime.
        """
        f_el_m, m_el_m, fiber_lengths = self.compute_elastic_efforts(r_rel_b, r_m_b)

        # algebraic inversion for specific force
        specific_force_b = (1.0 / self.seismic_mass) * (r_m_b @ f_el_m)

        # algebraic inversion for angular acceleration
        angular_accel_b = (1.0 / self.i_m) * (r_m_b @ m_el_m)

        return specific_force_b, angular_accel_b, fiber_lengths
