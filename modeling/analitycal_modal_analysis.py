import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from modeling.accel_model import accel_model_euler_poincare

class SystemDynamicsAnalyzer:
    """
    Classe para realizar a linearização e análise modal do acelerômetro
    utilizando a representação no espaço de estados (A, B, C, D).
    Calcula as matrizes estruturais analiticamente baseada no Jacobiano Geométrico.
    """
    def __init__(self, model: accel_model_euler_poincare):
        self.model = model
        
        # Matriz de Massa Global (M = diag(m*I, Im*I))
        self.M = np.diag([
            model.seismic_mass, model.seismic_mass, model.seismic_mass,
            model.i_m, model.i_m, model.i_m
        ])
        
        self.K = np.zeros((6, 6))
        self.C = np.zeros((6, 6))
        self.A = np.zeros((12, 12))
        
    def _compute_analytical_matrices(self, r_rel_0=np.zeros(3), r_m_b_0=np.eye(3)):
        """
        Calcula as matrizes de Rigidez (K) e Amortecimento (C) analiticamente,
        utilizando o formalismo do Jacobiano cinemático dos pontos de ancoragem.
        """
        self.K = np.zeros((6, 6))
        r_b_m_0 = r_m_b_0.T
        
        # Constante de amortecimento de Rayleigh (Mesmo valor usado no modelo físico)
        alpha_damping = 5e-5 
        
        for j in range(12):
            m_j = self.model.m_m[j]
            b_j = self.model.b_b[j]
            
            # 1. Vetor da fibra no equilíbrio (no referencial da base)
            fiber_vec_b = r_rel_0 + r_m_b_0 @ m_j - b_j
            l_j = np.linalg.norm(fiber_vec_b)
            
            # 2. Vetor direcional unitário (projetado no referencial da massa)
            u_j = r_b_m_0 @ (fiber_vec_b / l_j)
            
            # 3. Vetor momento de alavanca (t_j = m_j x u_j)
            # Mapeia a conversão de forças longitudinais em torques
            t_j = np.cross(m_j, u_j)
            
            # 4. Construção dos blocos tensoriais exatos (Expansão de Taylor Analítica)
            # K_tt: Rigidez Translacional pura
            K_tt = np.outer(u_j, u_j)
            
            # K_tr e K_rt: Acoplamentos Translacionais-Rotacionais
            K_tr = np.outer(u_j, t_j)
            K_rt = np.outer(t_j, u_j)
            
            # K_rr: Rigidez Torcional/Rotacional pura
            K_rr = np.outer(t_j, t_j)
            
            # Matriz de rigidez individual da fibra j (6x6)
            K_fiber = np.block([
                [K_tt, K_tr],
                [K_rt, K_rr]
            ])
            
            # Acumula na rigidez global multiplicando pela constante de Hooke k_j
            self.K += self.model.k * K_fiber
            
        # Amortecimento de Rayleigh estritamente proporcional à rigidez
        self.C = alpha_damping * self.K

    def build_state_space(self) -> np.ndarray:
        """
        Monta a matriz de estado A de dimensões 12x12 na origem geométrica.
        """
        # Chama a avaliação analítica no ponto de equilíbrio estático nulo
        self._compute_analytical_matrices(r_rel_0=np.zeros(3), r_m_b_0=np.eye(3))
        
        M_inv = np.linalg.inv(self.M)
        
        # A = [ 0          I      ]
        #     [ -M^-1*K   -M^-1*C ]
        self.A[0:6, 6:12] = np.eye(6)
        self.A[6:12, 0:6] = -M_inv @ self.K
        self.A[6:12, 6:12] = -M_inv @ self.C
        
        return self.A

    def analyze_eigenvalues(self) -> pd.DataFrame:
        """
        Extrai os polos (autovalores) da matriz A e calcula as frequências
        naturais e fatores de amortecimento estruturais do sensor.
        """
        self.build_state_space()
        
        # Cálculo dos autovalores
        eigenvalues, _ = np.linalg.eig(self.A)
        
        modes = []
        # Sistemas subamortecidos geram polos complexos conjugados. 
        # Filtramos a parte imaginária positiva para os 6 modos físicos.
        for ev in eigenvalues:
            if np.imag(ev) >= 0.0:
                sigma = np.real(ev)
                wd = np.imag(ev)
                
                # Frequência natural não amortecida (módulo do polo)
                omega_n = np.abs(ev)
                
                # Razão de amortecimento zeta
                zeta = -sigma / omega_n if omega_n > 1e-12 else 0.0
                
                # Frequência em Hertz
                fn_hz = omega_n / (2.0 * np.pi)
                
                modes.append({
                    "Polo Real (sigma)": sigma,
                    "Polo Imag (omega_d)": wd,
                    "omega_n [rad/s]": omega_n,
                    "f_n [Hz]": fn_hz,
                    "zeta (Amortecimento)": zeta
                })
                
        # Ordena pelos modos de menor frequência
        modes = sorted(modes, key=lambda x: x["ω_n [rad/s]"])
        
        df_modes = pd.DataFrame(modes)
        df_modes.index.name = "Modo"
        df_modes.index += 1
        
        return df_modes

if __name__ == "__main__":
    # Instancia o modelo físico com as propriedades de material e geometria
    model = accel_model_euler_poincare()
    
    # Executa o analisador linear
    analyzer = SystemDynamicsAnalyzer(model)
    df_results = analyzer.analyze_eigenvalues()
    
    pd.set_option('display.float_format', lambda x: f'{x:.4e}' if abs(x) < 1e-3 else f'{x:.4f}')
    print("-------------------------------------------------------------------------")
    print("ANÁLISE ESTRUTURAL (Solução Analítica via Jacobiano - Kövecses)")
    print("-------------------------------------------------------------------------")
    print(df_results.to_string())
    print("-------------------------------------------------------------------------")
    
    # Print das Matrizes para colocar na Tese se precisar
    print("\nMatriz de Rigidez Translacional K_tt (N/m):")
    print(np.array2string(analyzer.K[0:3, 0:3], formatter={'float_kind':lambda x: "%.2e" % x}))
    
    
    # analitycal 