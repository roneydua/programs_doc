import numpy as np


RHO = 2698.9
E_MODULE = 70e9
AREA = np.pi * (125e-6) ** 2 / 4
ARESTA = 0.0164
L0 = 0.003
e = ARESTA / 2
d = e - 0.001

mass = RHO * (ARESTA**3)
Im = (mass * ARESTA**2) / 6
k = E_MODULE * AREA / L0
alpha = 2.2e-6


wn_t = np.sqrt(3 * k**2 * alpha**2 + 4 * k * mass) / mass
fn_t = wn_t/ (2 * np.pi)
zeta_t = 2 * k * alpha / np.sqrt(3 * k**2 * alpha**2 + 4 * k * mass)

wn_r = np.sqrt(3 * (d * k* alpha)**2 + 4 * k * Im) *d/Im
fn_r = wn_r /(2 * np.pi)
zeta_r = 2 * d * k * alpha / np.sqrt(3 * (d * k * alpha)**2 + 4 * k * Im)


print(f"wn_t={wn_t}\t fn_t={fn_t}\t zeta_t={zeta_t}\t")
print(f"wn_r={wn_r}\t fn_r={fn_r}\t zeta_r={zeta_r}\t")
