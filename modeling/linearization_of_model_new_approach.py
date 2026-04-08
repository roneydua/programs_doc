import sympy as sp


def compute_symbolic_stiffness_matrix():
    # define symbolic variables
    e, d, l_0, k_stiffness = sp.symbols("e d l_0 k", real=True, positive=True)
    a_silic = sp.Symbol(r"$\alpha$", real=True, positive=True)
    # geometric relation: base semi-edge (f) is the mass semi-edge (e) plus the fiber length (l_0)
    f = e + l_0

    # connection points on the seismic mass (m_j) and base (b_j)
    connections = [
        (sp.Matrix([e, d, 0]), sp.Matrix([f, d, 0])),
        (sp.Matrix([e, -d, 0]), sp.Matrix([f, -d, 0])),
        (sp.Matrix([-e, d, 0]), sp.Matrix([-f, d, 0])),
        (sp.Matrix([-e, -d, 0]), sp.Matrix([-f, -d, 0])),
        (sp.Matrix([0, e, d]), sp.Matrix([0, f, d])),
        (sp.Matrix([0, e, -d]), sp.Matrix([0, f, -d])),
        (sp.Matrix([0, -e, d]), sp.Matrix([0, -f, d])),
        (sp.Matrix([0, -e, -d]), sp.Matrix([0, -f, -d])),
        (sp.Matrix([d, 0, e]), sp.Matrix([d, 0, f])),
        (sp.Matrix([-d, 0, e]), sp.Matrix([-d, 0, f])),
        (sp.Matrix([d, 0, -e]), sp.Matrix([d, 0, -f])),
        (sp.Matrix([-d, 0, -e]), sp.Matrix([-d, 0, -f])),
    ]

    # initialize 6x6 global stiffness matrix
    k_matrix_global = sp.zeros(6, 6)

    for m_j, b_j in connections:
        # relative position vector at equilibrium
        d_j = m_j - b_j

        # longitudinal unit vector (l_hat)
        l_hat = d_j / l_0

        # lever arm vector (t_hat)
        t_hat = m_j.cross(l_hat)

        # block submatrices
        k_tt = l_hat * l_hat.T
        k_tr = l_hat * t_hat.T
        k_rt = t_hat * l_hat.T
        k_rr = t_hat * t_hat.T

        # assemble local stiffness matrix for fiber j
        k_j = sp.Matrix.vstack(
            sp.Matrix.hstack(k_tt, k_tr), sp.Matrix.hstack(k_rt, k_rr)
        )

        # sum to the global matrix
        k_matrix_global += k_stiffness * k_j

    # simplify the final analytical matrix
    k_matrix_global = sp.simplify(k_matrix_global)

    return k_matrix_global, a_silic*k_matrix_global, a_silic


if __name__ == "__main__":
    mass, inertial = sp.symbols(r"$m$ $I_{m}$", real=True, positive=True)
    mass_matrix = sp.diag(mass, mass, mass, inertial, inertial, inertial)
    s = sp.Symbol("s", real=True, positive=True)
    k_sym, c_sym,a  = compute_symbolic_stiffness_matrix()
    mat_a = sp.zeros(12,12)
    mat_a[0:6,6:] = sp.ones(6)
    mat_a[6:,0:6] = -(mass_matrix**-1) @ k_sym
    mat_a[6:,6:] = -(mass_matrix**-1) @ c_sym
    matrix_eigen = s**2 * sp.eye(6) + (a * s + 1) * mass_matrix ** (-1) @ k_sym
    eigen_vals = (matrix_eigen).det().simplify()
    print(sp.latex(eigen_vals))
    print(sp.latex(matrix_eigen))
