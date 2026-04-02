import numpy as np
from scipy.optimize import minimize

def otimizar_acelerometro(f_alvo):
    # Parâmetros Físicos e Constantes
    rho_al = 2698.9      # Densidade do alumínio (kg/m^3)
    E_silica = 70e9      # Módulo de Young da sílica (Pa)
    phi_fiber = 125e-6   # Diâmetro da fibra óptica (m)
    A_fiber = np.pi * (phi_fiber**2) / 4.0
    p_e = 0.22           # Constante fotoelástica efetiva da fibra
    g = 9.81             
    
    w_alvo = 2 * np.pi * f_alvo

    # (Como S depende apenas de M linearmente, maximizar M maximiza S)
    def objective(x):
        L, l0 = x
        M = rho_al * (L**3)
        return -M  # Retorna negativo para o minimizador

    # Restrição: f_n >= f_alvo  => w_n^2 - w_alvo^2 >= 0
    def constraint_freq(x):
        L, l0 = x
        M = rho_al * (L**3)
        w_n_sq = (4 * E_silica * A_fiber) / (l0 * M)
        return w_n_sq - (w_alvo**2)

    # Limites físicos de manufatura (em metros)
    # l0: Comprimento da fibra entre 1 mm e 10 mm
    bounds = [(0.005, 0.040), (0.001, 0.010)]

    # Chute inicial (os valores do seu protótipo atual)
    x0 = [0.0164, 0.003]

    # Dicionário de restrições para o solver SLSQP
    con = {'type': 'ineq', 'fun': constraint_freq}

    # Executa a otimização
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=con)

    # Extração dos resultados ótimos
    L_opt, l0_opt = res.x
    M_opt = rho_al * (L_opt**3)
    
    # Cálculos finais de validação
    w_n_opt = np.sqrt((4 * E_silica * A_fiber) / (l0_opt * M_opt))
    f_n_opt = w_n_opt / (2 * np.pi)
    
    # Sensibilidade Translacional Única (em /g)
    # epsilon/g = M_opt * g / (4 * E_silica * A_fiber)
    S_trans = (1 - p_e) * (M_opt * g) / (4 * E_silica * A_fiber)
    
    k = E_silica * A_fiber / l0_opt
    print(np.sqrt(k /M_opt))
    # Sensibilidade Diferencial Push-Pull (multiplicada por 2)
    S_push_pull = 2 * S_trans

    print(f"--- Resultados da Otimização para f_alvo = {f_alvo} Hz ---")
    print(f"Status da Otimização: {res.message}")
    print(f"Aresta ideal do cubo (L):  {L_opt*1000:.2f} mm")
    print(f"Comprimento da fibra (l0): {l0_opt*1000:.2f} mm")
    print(f"Massa sísmica ideal (M):   {M_opt*1000:.2f} g")
    print(f"Frequência natural final:  {f_n_opt:.2f} Hz")
    print(f"Sensibilidade (Push-Pull): {S_push_pull * 1e6:.2f} µstrain/g")
    # Para pm/g, basta multiplicar a strain_push_pull pelo lambda_B (ex: 1550 nm)
    sens_pm_g = (S_push_pull) * 1550e3
    print(f"Sensibilidade em Bragg:    {sens_pm_g:.2f} pm/g (para λB = 1550 nm)\n")

    return res.x

if __name__ == "__main__":
    # Testando para algumas frequências de projeto
    otimizar_acelerometro(f_alvo=1561.11)
    # otimizar_acelerometro(f_alvo=1500)
    # otimizar_acelerometro(f_alvo=2000)