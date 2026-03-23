import numpy as np
from numpy.linalg import inv
import common_functions.quaternion_functions as fq


# Constates coordinates of connections


class states:
    rot = np.eye(3)
    """ Rotation matrix"""
    euler = np.zeros(3)
    """Euler angules. psi, theta, phi [deg]"""

    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        """attitude quaternion"""
        self.w = np.array([0, 0, 0])
        """# angular body velcity"""
        self.r = np.array([0, 0, 0])
        """inertial position"""
        self.dr = np.array([0, 0, 0])
        """inertial velocity"""
        self.ddr = np.array([0, 0, 0])
        """inertial acceleration"""

    def updates_attitude(self, q):
        """
        updates_attitude Update states of attitude quaterion q, Euler angles and matrix rotation
        Args:
            q: Attitude quaterion.
        """
        self.q = q
        self.euler = fq.quat2Euler(q, deg=1)
        self.rot = fq.rotationMatrix(self.q)

    def psi(self):
        return self.euler[0]

    def theta(self):
        return self.euler[1]

    def phi(self):
        return self.euler[2]


class AccelModelInertialFrame(object):
    density = 2.6989e3
    """material density  (aluminum)"""
    E = 70e9  # GPa
    """Young's Module"""
    fiber_diameter = 125e-6
    """fiber diameter"""
    sms = states()
    """seismic mass states"""
    bss = states()
    """base sensor states"""
    G = 0.0  # -9.89
    """Gravity"""
    damper_for_computation_simulations = 0.0
    """Artificial damper for numerical simulations"""

    def __init__(
        self,
        seismic_edge=16.4e-3,
        fiber_diameter=fiber_diameter,
        fiber_length=3e-3,
        damper_for_computation_simulations=0.0,
        density=density,
    ):
        """
        __init__ Class to compute states wrt inertial frame

        Args:
            seismic_edge: Defaults to 16.4e-3.
            fiber_diameter: Defaults to 125e-6.
            fiber_length: Defaults to 3e-3.
            fibers_with_info: Defaults to np.arange(1, 13).
            damper_for_computation_simulations: Defaults to 0.0.
            density: density of seismic mass. Defaults =  2.6989e3
        """
        self.density = density
        self.fiber_diameter =fiber_diameter
        
        self.damper_for_computation_simulations = damper_for_computation_simulations
        
        # Fiber diameter
        self.fiber_length = fiber_length
        """initial fiber length"""
        self.optical_fiber_area = 0.25 * np.pi * (self.fiber_diameter**2)
        """fiber optical area in m^2"""
        self.k = (
            self.E * self.optical_fiber_area
        ) / self.fiber_length
        """stiffness of optical fiber"""
        self.seismic_edge = seismic_edge
        """Length of seismic edge"""
        self.seismic_mass = (self.seismic_edge**3.0) * self.density
        """Sismic mass"""
        self.external_base_sensor_edge = (
            self.seismic_edge + 2.0 * self.fiber_length + 4e-3
        )
        """ approximation of the base of the sensor as a cube.The value 6e-3 refers double the length of the fibers that support the cube.4e-3 is twice the thickness."""
        self.base_sensor_edge = self.seismic_edge + 2.0 * self.fiber_length
        self.base_sensor_mass = (
            1e3
            * (self.external_base_sensor_edge**3 - self.base_sensor_edge**3)
            * self.density
        )
        self.inertial_seismic_mass = (
            np.eye(3) * self.seismic_mass / 6.0 * self.seismic_edge**2
        )
        """ Matrix of moments of inertia da massa seismic"""
        self.inertial_base_sensor = 1e3 * (
            np.eye(3) * self.base_sensor_mass / 6.0 * self.seismic_edge**2
        )
        # NOTE: Only for computational economy
        self.inertial_base_sensor_inv = inv(self.inertial_base_sensor)
        self.inertial_seismic_mass_inv = inv(self.inertial_seismic_mass)
        # perpendicular distance of center
        _d = self.seismic_edge / 2.0 - 1e-3
        _e = self.seismic_edge / 2.0
        self.m_M = np.array(
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
        _f = self.base_sensor_edge / 2.0
        self.b_B = np.array(
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
        self.b_I = 1.0 * self.b_B
        """ Inertial vectors of connections of coils on sensor base"""
        self.m_I = 1.0 * self.m_M
        """ Inertial vectors of connections of coils on seismic mass"""
        self.f = 0.0 * self.m_M
        """ Vector f. This vector contains the length of the optical fiber at the current instant."""
        self.leg = [
            r"$x_{+}z_{+}$",
            r"$x_{+}z_{-}$",
            r"$x_{-}z_{+}$",
            r"$x_{-}z_{-}$",
            r"$y_{+}z_{+}$",
            r"$y_{+}z_{-}$",
            r"$y_{-}z_{+}$",
            r"$y_{-}z_{-}$",
            r"$x_{+}y_{+}$",
            r"$x_{+}y_{-}$",
            r"$x_{-}y_{+}$",
            r"$x_{-}y_{-}$",
        ]
        c_critical = 4.0*np.sqrt(self.k / self.seismic_mass)
        damping_ratio = 1.0 #zeta 
        self.damper_for_computation_simulations = c_critical * damping_ratio
        """ undamped natural frequency hz"""
        self.undamped_translation_natural_frequency_hz = np.sqrt(4*self.k/self.seismic_mass) / (np.pi*2.)
        self.undamped_rotational_natural_frequency_hz=np.sqrt(
            4 * self.k *_d**2/ self.inertial_seismic_mass[0,0]
        ) / (np.pi * 2.0)
        print("The natural translational frequency is %1.1f Hz \t" % self.undamped_translation_natural_frequency_hz)
        print(
            "The natural rotational frequency is %1.1f Hz \t"
            % self.undamped_rotational_natural_frequency_hz
        )
        """ undamped natural frequency hz"""
        self.minimal_recomendated_time_step = 1 / (10.*self.undamped_translation_natural_frequency_hz)
        print("Is is recomendated that time step of integrator is 10x(1/natural frequency) so %1.0e s \t" % self.minimal_recomendated_time_step)
        # legend of numerical point

    def update_inertial_coil_connections(self):
        """
        update the inertial coordinates of mass and body connections

        The deformation vector f is also updated.

        """
        for i in range(12):
            # inertial vectors of connections of coils on sensor base
            self.m_I[i, :] = self.sms.r + self.sms.rot @ self.m_M[i, :]
            # inertial vectors of connections of coils on seismic mass
            self.b_I[i, :] = self.bss.r + self.bss.rot @ self.b_B[i, :]

    def update_f_vector(self):
        """update deformation vector f"""
        self.f = self.m_I - self.b_I

    def update_states(self, rb, qb, rm, qm):
        """
        update_states(): Update the states of translation and rotation

        Note, this function call update_inertial_coil_connections() and update_f_vector() to update f vector.
        Args:
            rb: translation vector of body system with respect of inertial
            qb: quaternion of attitude of body sensor
            rm: translation vector of seismic mass with respect of inertial system
            qm: attitude quaternion of seismic mass.
        """
        self.bss.r = rb
        self.bss.updates_attitude(qb)
        self.sms.r = rm
        self.sms.updates_attitude(qm)
        self.update_inertial_coil_connections()
        self.update_f_vector()

    def get_d_rb(self, d_x: np.ndarray):
        """
        get_d_rb return inertial velocity of body sensor

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            d_rb: inertial velocity of body sensor
        """
        return d_x[:3]

    def get_d_rm(self, d_x: np.ndarray):
        """
        get_d_rm return inertial velocity of seismic mass

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            d_rm: inertial velocity of seismic mass
        """
        return d_x[3:6]

    def get_rb(self, d_x: np.ndarray):
        """
        get_rb return inertial  position of body sensor

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            rb: inertial  position of body sensor
        """
        return d_x[6:9]

    def get_rm(self, d_x: np.ndarray):
        """
        get_rm return inertial position of seismic mass

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            rm: inertial position of seismic mass
        """
        return d_x[9:12]

    def get_qb(self, d_x: np.ndarray):
        """
        get_qb return atitude quaternion of body sensor

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            qb: Atitude quaternion of body sensor
        """
        return d_x[12:16]

    def get_qm(self, d_x: np.ndarray):
        """
        get_qm return attitude quaternion of seismic mass

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            qm: Atitude quaternion of seismic mass
        """
        return d_x[16:20]

    def get_wb(self, d_x: np.ndarray):
        """
        get_wb return angular velocity of body sensor

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            wb: Angular velocity of body sensor
        """
        return d_x[20:23]

    def get_wm(self, d_x: np.ndarray):
        """
        get_wm return angular velocity of seismic mass

        Args:
            d_x: Complete vector of state space of problem in order:
                    [drb,drm,rb,rm,qb,qm,wb,wm]
        Returns:
            wm: Angular velocity of seismic mass
        """
        return d_x[23:26]

    def func_dd_x(self, t: float, d_x: np.ndarray):
        """
        dd_x_forced_body_state calc second order of model for numerical integration.
        Args:
            d_x: first order [drb,drm,rb,rm,qb,qm,wb,wm]
        """
        # d_x = np.arange(1, 21)
        dd_x = np.zeros(26)

        d_rb = d_x[:3]
        """inertial velocity of body sensor"""
        d_rm = d_x[3:6]
        """inertial velocity of seismic mass"""
        rb = d_x[6:9]
        """inertial position of body sensor"""
        rm = d_x[9:12]
        """inertial position of seismic mass"""
        qb = d_x[12:16]  # /np.linalg.norm(d_x[12:16])
        """Atitude quaternion of body sensor"""
        qm = d_x[16:20]  # /np.linalg.norm(d_x[16:20])
        """Atitude quaternion of seismic mass"""
        wb = d_x[20:23]
        """Angular velocity of body sensor"""
        wm = d_x[23:]
        """Angular velocity of seismic mass"""
        # calculate deformation vector
        f_hat_dell = np.zeros((3, 12))
        sum_f_hat_dell = np.zeros(3)
        sum_f_hat_dell_dfdq_M = np.zeros(4)
        sum_f_hat_dell_dfdq_B = np.zeros(4)
        rot_qm = fq.rotationMatrix(qm)
        rot_qb = fq.rotationMatrix(qb)
        for i in range(12):
            # compute f
            f_hat_dell[:, i] = (
                rm + rot_qm @ self.m_M[i, :] - rb - rot_qb @ self.b_B[i, :]
            )
            # calculate a norm of v to get deformation and versor of f
            f_norm = np.linalg.norm(f_hat_dell[:, i])
            # compute f_hat
            f_hat_dell[:, i] /= f_norm
            # compute f_hat_dell
            f_hat_dell[:, i] *= f_norm - self.fiber_length
            # sum for compute translational movements
            sum_f_hat_dell += f_hat_dell[:, i]
            sum_f_hat_dell_dfdq_M += (
                fq.calc_dfdq(qm, self.m_M[i, :]).T @ f_hat_dell[:, i]
            )
            # NOTE: important signal of dfdq
            sum_f_hat_dell_dfdq_B += (
                -fq.calc_dfdq(qb, self.b_B[i, :]).T @ f_hat_dell[:, i]
            )
        dd_x[:3] = self.k * sum_f_hat_dell / self.base_sensor_mass
        # dd_x[:3] += (
        #     self.damper_for_computation_simulations / self.base_sensor_mass
        # ) * (d_rm - d_rb)
        # calculate dd_rm
        dd_x[3:6] = -self.k * sum_f_hat_dell / self.seismic_mass
        # dd_x[5] += self.G
        # artificial damper
        dd_x[3:6] -= (self.damper_for_computation_simulations / self.seismic_mass) * (
            d_rm-d_rb
        )
        # calculate d_rb
        dd_x[6:9] = d_rb
        # calculate d_rm
        dd_x[9:12] = d_rm
        # Left quaternion matrix
        Qb = fq.matrixQ(qb)
        Qm = fq.matrixQ(qm)
        # calculate attitude quaternion of body sensor
        dd_x[12:16] = 0.5000000001 * Qb @ wb
        # calculate attitude quaternion of seismic mass
        dd_x[16:20] = 0.5000000001 * Qm @ wm
        # calculate a angular acceleration of body sensor
        # dd_x[20:23] = -self.inertial_base_sensor_inv @ (fq.screwMatrix(wb) @
        # self.inertial_base_sensor @ wb + 0.5 * self.k * Qb.T @ sum_f_hat_dell_dfdq_B - u[3:])
        dd_x[20:23] = -(
            0.5
            * self.inertial_base_sensor_inv[0, 0]
            * self.k
            * Qb.T
            @ sum_f_hat_dell_dfdq_B
        )
        dd_x[20:23] += (
            self.damper_for_computation_simulations
            * self.inertial_base_sensor_inv[0, 0]
        ) * (rot_qb.T@rot_qm@wm - wb)
        # calculate a angular acceleration of body sensor
        # ATTENTION! due to symmetry, the product fq.screwMatrix(wm) @ self.inertial_seismic_mass @ wm it is always zero!

        dd_x[23:26] = -(
            0.5
            * self.k
            * self.inertial_seismic_mass_inv[0, 0]
            * (Qm.T @ sum_f_hat_dell_dfdq_M)
        )
        # dd_x[23:26] -= (
        #     self.damper_for_computation_simulations
        #     * self.inertial_seismic_mass_inv[0, 0]
        # ) * (wm - rot_qm.T@rot_qb@wb)
        return dd_x


class InverseProblem(AccelModelInertialFrame):
    """docstring for inverse_problem."""

    recover_type_flag = ""
    """Type of recover. full_estimation for complete solution and full_estimation_reduced to recover angular acceleration. linear_estimation do not recover angular accelerations"""
    estimated_rm_B = np.zeros(3)
    # estimated relative position of body and seismic mass
    estimated_q_M_B = np.array([1.0, 0.0, 0.0, 0.0])
    # estimated atitude quaternion seismic mass with respect to body
    estimated_f_B = np.zeros((12, 3))
    # estimate f vector on B coordinate system
    norm_of_estimated_f_B = np.zeros((12, 1), dtype=np.float64)

    def __init__(
        self,
        density:float,
        fibers_with_info: np.ndarray,
        recover_angular_accel=False,
        fiber_length=3e-3,
        **kwarg
    ):
        """
        __init__ Constructor of inverse_problem.
        Args:
            density: density of seismic mass.
            fibers_with_info: fiber indices considered to solve the problem
            recover_angular_accel: Defaults to False.
            fiber_length: size of fiber. Defaults is 3mm or (0.003m)
        **kwargs: full_estimation: True to recover the term q_M_B cross r_m_B

        """
        super().__init__(fiber_length=fiber_length,density=density)
        print(self.k)
        self.fibers_with_info = fibers_with_info
        self.fibers_with_info_index = fibers_with_info - 1
        self.k_by_m = self.k / self.seismic_mass
        """Ratio between stiffness and mass to use on accel recover"""

        if recover_angular_accel:
            if "full" == kwarg["estimation"]:
                self.recover_type_flag = "full"
                # considers that the term q_M_B x r_m_B must be estimated
                _N = 10
            elif "reduced" == kwarg["estimation"]:
                self.recover_type_flag = "reduced"
                # considers that the term q_M_B x r_m_B it is much smaller than the other terms
                _N = 7
            else:
                print("not recognized recovery")
                quit()
            self.var_xi = np.ones((self.fibers_with_info.size, _N))
            self.var_gamma = np.zeros(_N)
            self.var_psi = np.zeros(self.fibers_with_info.size)
            # construct the constant var_gamma matrix and solution of least squared
            _aux_vector = np.ones((self.fibers_with_info.size, 3))
            """auxiliar vector to compute (m-b) with dimension fiber_with_sise by 3, used on var_xi and var_psi"""
            self.aux_var_psi_matrix = np.zeros(self.fibers_with_info.size)

            for i, j in enumerate(self.fibers_with_info_index):
                _aux_vector[i, :] = self.m_M[j, :] - self.b_B[j, :]
                self.aux_var_psi_matrix[i] = _aux_vector[i, :].dot(_aux_vector[i, :])
                self.var_xi[i, 1:4] = 2.0 * _aux_vector[i, :]
                if self.recover_type_flag == "full":
                    self.var_xi[i, 4:7] = -4.0 * self.m_M[j, :].T
                # self.var_xi[i, -3:] = -4.0 * np.cross(self.m_M[j, :], self.b_B[j, :])
                self.var_xi[i, -3:] = (
                    -4.0 * self.m_M[j, :].T @ fq.screw_matrix(self.b_B[j, :])
                )
        else:
            self.recover_type_flag = "linear_estimation"
            self.var_xi = np.ones((self.fibers_with_info.size, 4))
            self.var_gamma = np.zeros(4)
            self.var_psi = np.zeros(self.fibers_with_info.size)
            # construct the constant var_gamma matrix and solution of least squared
            _aux_vector = np.ones((self.fibers_with_info.size, 3))
            """auxiliar vector to compute (m-b) with dimension fiber_with_sise by 3, used on var_xi and var_psi"""
            self.aux_var_psi_matrix = np.zeros(self.fibers_with_info.size)

            for i, j in enumerate(self.fibers_with_info_index):
                _aux_vector[i, :] = self.m_M[j, :] - self.b_B[j, :]
                self.aux_var_psi_matrix[i] = _aux_vector[i, :].dot(_aux_vector[i, :])
            self.var_xi[:, 1:] = 2.0 * _aux_vector
            self.diff_m_M_b_B = self.m_M - self.b_B

        if self.var_xi.shape[0] == self.var_gamma.size:
            # in this case the least squared method use only the inverse matrix of var_gamma
            self.least_square_matrix = np.linalg.inv(self.var_xi)
        else:
            # It is necessary compute pseudo inverse of matrix
            self.least_square_matrix = np.linalg.pinv(self.var_xi)

    def compute_inverse_problem_solution(self, fiber_len: np.ndarray):
        """
        compute_inverse_problem_solution
        Args:
            fiber_len: vector of fiber_len is ((f).dot(f))^1/2
        """
        self.var_psi = np.square(fiber_len) - self.aux_var_psi_matrix
        self.var_gamma = self.least_square_matrix @ self.var_psi
        self.estimate_f_vector()
        if self.recover_type_flag == "linear_estimation":
            return self.estimate_ddrm_B()
        else:
            return self.estimate_ddrm_B(), self.estiamate_dw_B()

    def estimate_f_vector(self):
        """
        estimate_f_vector Estimation with estimated relative positions r_m_B

        Args:
            estimated_rm_B: the vectors solution

        Returns:
            _description_ the estimate f vector (12,3)
        """
        if self.recover_type_flag == "linear_estimation":
            for i in range(12):
                self.estimated_f_B[i, :] = self.var_gamma[1:] + self.diff_m_M_b_B[i, :]
            self.norm_of_estimated_f_B = np.linalg.norm(self.estimated_f_B, axis=1)
        elif self.recover_type_flag in ("full", "reduced"):
            ## compute relative attitude
            self.estimated_q_M_B[1:] = self.var_gamma[-3:]
            self.estimated_q_M_B[0] = np.sqrt(
                1.0 - self.estimated_q_M_B[1:].dot(self.estimated_q_M_B[1:])
            )

            _rot_M_B = fq.rotationMatrix(self.estimated_q_M_B)
            for i in range(12):
                self.estimated_f_B[i, :] = (
                    self.var_gamma[1:4] + _rot_M_B @ self.m_M[i, :] - self.b_B[i, :]
                )
            self.norm_of_estimated_f_B = np.linalg.norm(self.estimated_f_B, axis=1)

    def estimate_ddrm_B(self):
        _t = np.zeros(3)
        for i in range(12):
            _t += (
                (self.norm_of_estimated_f_B[i] - self.fiber_length)
                / self.norm_of_estimated_f_B[i]
            ) * self.estimated_f_B[i, :]
        return -self.k_by_m * _t

    def estiamate_dw_B(self):
        _t = np.zeros(4)
        _Q_Im_k = (
            -0.5
            * self.k
            * self.inertial_seismic_mass_inv[0, 0]
            * fq.matrixQ(self.estimated_q_M_B).T
        )

        for i in range(12):
            _t += (
                (self.norm_of_estimated_f_B[i] - self.fiber_length)
                / self.norm_of_estimated_f_B[i]
            ) * (
                fq.calc_dfdq(a=self.m_M[i, :], q=self.estimated_q_M_B).T
                @ self.estimated_f_B[i, :]
            )

        return _Q_Im_k @ _t


class SimpleSolution(AccelModelInertialFrame):
    """Implementação do método empregado no trabalho do Cazo."""

    norm_of_estimated_f_B = np.zeros((12, 1), dtype=np.float64)

    def __init__(self, fibers_with_info: np.ndarray, density:float,fiber_length=3e-3):
        """
        __init__ Constructor of inverse_problem.
        Args:
            fibers_with_info: fiber indices considered to solve the problem
            recover_angular_accel: Defaults to False.
        """
        super().__init__(fiber_length=fiber_length, density=density)
        # self.fibers_with_info = fibers_with_info
        self.fibers_with_info_index = fibers_with_info - 1
        self.coef_one_fiber = 4.0 * self.k / self.seismic_mass
        """ Coefficient to compute with only a one optical fiber. 
        4 k/m !!!!! Note the coefficient 4"""
        self.coef_differential = 2.0 * self.k / self.seismic_mass
        """Coefficient to compute accel with differential method"""

    def estimated_ddrm_B(self, fiber_len: np.ndarray, method: str):
        """
        estimated_ddrm_B Estimatio  n of accel with no cross effects
        This method use the fibers_with_info_index variables to extract correct dimensions
        Args:
            fiber_len: 12 dimension vector of current fiber lengths
            method: 'one_fiber', 'differential_aligned' or 'differential_cross'
        """
        if method == "one_fiber":
            _r = np.take(fiber_len, self.fibers_with_info_index) - self.fiber_length
            return self.coef_one_fiber * _r
        elif method == "differential_aligned":
            _fiber_length_push_pull = np.zeros(3)
            _fiber_length_push_pull[0] = fiber_len[0] - fiber_len[2]
            _fiber_length_push_pull[1] = fiber_len[4] - fiber_len[6]
            _fiber_length_push_pull[2] = fiber_len[8] - fiber_len[10]
            _r = _fiber_length_push_pull  # -2.0*self.fiber_length
            return self.coef_differential * _r
        elif method == "differential_cross":
            _fiber_length_push_pull = np.zeros(3)
            _fiber_length_push_pull[0] = fiber_len[0] - fiber_len[3]
            _fiber_length_push_pull[1] = fiber_len[4] - fiber_len[7]
            _fiber_length_push_pull[2] = fiber_len[8] - fiber_len[11]
            _r = _fiber_length_push_pull  # -2.0*self.fiber_length
            return self.coef_differential * _r
