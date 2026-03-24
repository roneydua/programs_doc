import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from modeling.accel_model import accel_model_euler_poincare # Adjust path if necessary

class SystemDynamicsAnalyzer:
    """
    Class to perform linearization and modal analysis of the accelerometer
    using the state-space representation (A, B, C, D).
    """
    def __init__(self, model: accel_model_euler_poincare):
        self.model = model
        
        # Global Mass Matrix (Thesis equation: M = diag(m*I, Im*I))
        self.M = np.diag([
            model.seismic_mass, model.seismic_mass, model.seismic_mass,
            model.i_m, model.i_m, model.i_m
        ])
        
        self.K = np.zeros((6, 6))
        self.C = np.zeros((6, 6))
        self.A = np.zeros((12, 12))
        
    def _compute_jacobians(self, h: float = 1e-6):
        """
        Calculates the linearized stiffness (K) and damping (C) matrices at the origin
        using central finite differences on the nonlinear elastic model.
        """
        for i in range(6):
            # -----------------------------------------------------------------
            # 1. Position / Attitude Perturbation (to obtain the K Matrix)
            # -----------------------------------------------------------------
            q_plus = np.zeros(6)
            q_plus[i] = h
            q_minus = np.zeros(6)
            q_minus[i] = -h
            
            # Positive evaluation (+h)
            r_plus = q_plus[0:3]
            R_plus = Rotation.from_euler('xyz', q_plus[3:6]).as_matrix()
            f_plus, m_plus, _ = self.model.compute_elastic_efforts(r_plus, np.zeros(3), R_plus, np.zeros(3))
            
            # Negative evaluation (-h)
            r_minus = q_minus[0:3]
            R_minus = Rotation.from_euler('xyz', q_minus[3:6]).as_matrix()
            f_minus, m_minus, _ = self.model.compute_elastic_efforts(r_minus, np.zeros(3), R_minus, np.zeros(3))
            
            # Stiffness Matrix K = - d(Restoring_Force) / dq
            self.K[0:3, i] = -(f_plus - f_minus) / (2.0 * h)
            self.K[3:6, i] = -(m_plus - m_minus) / (2.0 * h)
            
            # -----------------------------------------------------------------
            # 2. Linear / Angular Velocity Perturbation (to obtain the C Matrix)
            # -----------------------------------------------------------------
            v_plus = np.zeros(6)
            v_plus[i] = h
            v_minus = np.zeros(6)
            v_minus[i] = -h
            
            # Positive evaluation (+h)
            f_v_plus, m_v_plus, _ = self.model.compute_elastic_efforts(np.zeros(3), v_plus[0:3], np.eye(3), v_plus[3:6])
            
            # Negative evaluation (-h)
            f_v_minus, m_v_minus, _ = self.model.compute_elastic_efforts(np.zeros(3), v_minus[0:3], np.eye(3), v_minus[3:6])
            
            # Damping Matrix C = - d(Dissipative_Force) / dv
            self.C[0:3, i] = -(f_v_plus - f_v_minus) / (2.0 * h)
            self.C[3:6, i] = -(m_v_plus - m_v_minus) / (2.0 * h)

    def build_state_space(self) -> np.ndarray:
        """
        Assembles the state matrix A of dimensions 12x12.
        """
        self._compute_jacobians()
        
        M_inv = np.linalg.inv(self.M)
        
        # A = [ 0          I      ]
        #     [ -M^-1*K   -M^-1*C ]
        self.A[0:6, 6:12] = np.eye(6)
        self.A[6:12, 0:6] = -M_inv @ self.K
        self.A[6:12, 6:12] = -M_inv @ self.C
        
        return self.A

    def analyze_eigenvalues(self) -> pd.DataFrame:
        """
        Extracts the poles (eigenvalues) of matrix A and calculates the
        natural frequencies and structural damping factors of the sensor.
        """
        self.build_state_space()
        
        # Eigenvalues calculation
        eigenvalues, _ = np.linalg.eig(self.A)
        
        modes = []
        # Underdamped systems generate complex conjugate poles. 
        # We filter only the positive imaginary part to list the 6 physical modes.
        for ev in eigenvalues:
            if np.imag(ev) >= 0.0:
                sigma = np.real(ev)
                wd = np.imag(ev)
                
                # Undamped natural frequency (pole magnitude)
                omega_n = np.abs(ev)
                
                # Damping ratio zeta
                zeta = -sigma / omega_n if omega_n > 1e-12 else 0.0
                
                # Frequency in Hertz
                fn_hz = omega_n / (2.0 * np.pi)
                
                # Approximate identification of the dominant degree of freedom
                modes.append({
                    "Polo Real (σ)": sigma,
                    "Polo Imag (ωd)": wd,
                    "ω_n [rad/s]": omega_n,
                    "f_n [Hz]": fn_hz,
                    "ζ (Amortecimento)": zeta
                })
                
        # Sort by lowest frequency modes (translations usually come before rotations)
        modes = sorted(modes, key=lambda x: x["ω_n [rad/s]"])
        
        df_modes = pd.DataFrame(modes)
        df_modes.index.name = "Modo"
        df_modes.index += 1
        
        # Since there may be small numerical asymmetries, we will use aesthetic rounding in the print
        return df_modes

if __name__ == "__main__":
    # Instantiates the physical model with material and geometric properties
    model = accel_model_euler_poincare()
    
    # Runs the linear analyzer
    analyzer = SystemDynamicsAnalyzer(model)
    df_results = analyzer.analyze_eigenvalues()
    
    pd.set_option('display.float_format', lambda x: f'{x:.4e}' if abs(x) < 1e-3 else f'{x:.4f}')
    print("-------------------------------------------------------------------------")
    print("ANÁLISE ESTRUTURAL (POLOS E FREQUÊNCIAS MODAIS)")
    print("-------------------------------------------------------------------------")
    print(df_results.to_string())
    print("-------------------------------------------------------------------------")