import sympy as sp


def main():
    # Define independent variable for integration
    lam = sp.Symbol(r"\lambda", real=True)
    # Define physical and geometric symbols
    a_l = sp.Symbol(r"a_l", real=True, positive=True)
    a_l
    a_b = sp.Symbol(r"a_b", real=True, positive=True)
    r_max = sp.Symbol(r"r_{max}", real=True, positive=True)
    delta_w_0 = sp.Symbol(r"\Delta_{w_{0}}", real=True, positive=True)
    delta_lam_r = sp.Symbol(r"\Delta_{\lambda_{r}}", real=True, positive=True)
    k_epsilon = sp.Symbol(r"kappa_\epsilon", real=True, positive=True)
    k_t = sp.Symbol(r"kapp_{T}", real=True, positive=True)
    epsilon = sp.Symbol(r"\epsilon", real=True)
    delta_t = sp.Symbol(r"\Delta_{T}", real=True)

    # Auxiliary symbol for total bandwidth in double transmission
    lam_band = sp.Symbol(r"\lambda_banda", real=True, positive=True)
    a_0 = sp.Symbol(r"A_0", real=True, positive=True)

    # -------------------------------------------------------------------------
    # 1. LASER MODE MODELING (Point Evaluation via Kinematic Shifts)
    # -------------------------------------------------------------------------
    # Slope defined by catalog parameters
    slope_laser = r_max / delta_lam_r

    # Kinematic shifts evaluated at the nominal laser emission point
    delta_l_1 = delta_w_0 / 2 + k_epsilon * epsilon + k_t * delta_t
    delta_l_2 = delta_w_0 / 2 - k_epsilon * epsilon + k_t * delta_t

    # Reflectivity profiles evaluated punctually at the laser wavelength
    r1_laser = slope_laser * delta_l_1
    r2_laser = slope_laser * delta_l_2

    # -------------------------------------------------------------------------
    # 2. SLD MODE MODELING (Geometric Integration over the Overlap Base)
    # -------------------------------------------------------------------------
    # Base integration width (temperature naturally cancelled out)
    delta_w_dynamic = delta_w_0 + 2 * k_epsilon * epsilon

    # Parametrized linear equations for colliding slopes inside the overlap region
    # Local coordinate x ranges from 0 to delta_w_dynamic
    x = sp.Symbol("x", real=True)
    slope_sld = r_max / delta_lam_r
    r1_sld_func = slope_sld * (delta_w_dynamic - x)
    r2_sld_func = slope_sld * x

    # Integration helper function for the product of slopes
    def integrate_product_sld(h_expr_func):
        integrand = h_expr_func(r1_sld_func, r2_sld_func)
        return sp.integrate(integrand, (x, 0, delta_w_dynamic))

    # -------------------------------------------------------------------------
    # 3. TOPOLOGY EVALUATION & EXPANSION
    # -------------------------------------------------------------------------
    topologies = {
        "Dupla reflexão": {
            "h_laser": (1 / sp.Integer(16)) * r1_laser * r2_laser,
            "p_laser_factor": a_l,
            "sld_func": lambda r1, r2: (1 / sp.Integer(16)) * r1 * r2,
            "p_sld_base": lambda integral: a_b * integral,
        },
        "Transmissão-reflexão": {
            "h_laser": (1 / sp.Integer(4)) * (1 - r1_laser) * r2_laser,
            "p_laser_factor": a_l,
            "sld_func": lambda r1, r2: (1 / sp.Integer(4)) * (1 - r1) * r2,
            "p_sld_base": lambda integral: (1 / sp.Integer(4)) * a_b * a_0
            - a_b
            * (1 / sp.Integer(4))
            * sp.integrate(r1_sld_func * r2_sld_func, (x, 0, delta_w_dynamic)),
        },
        "Reflexão-transmissão": {
            "h_laser": (1 / sp.Integer(4)) * r1_laser * (1 - r2_laser),
            "p_laser_factor": a_l,
            "sld_func": lambda r1, r2: (1 / sp.Integer(4)) * r1 * (1 - r2),
            "p_sld_base": lambda integral: (1 / sp.Integer(4)) * a_b * a_0
            - a_b
            * (1 / sp.Integer(4))
            * sp.integrate(r1_sld_func * r2_sld_func, (x, 0, delta_w_dynamic)),
        },
        "Dupla transmissão": {
            "h_laser": (1 - r1_laser) * (1 - r2_laser),
            "p_laser_factor": a_l,
            "sld_func": lambda r1, r2: (1 - r1) * (1 - r2),
            "p_sld_base": lambda integral: a_b * (lam_band - 2 * a_0)
            + a_b * sp.integrate(r1_sld_func * r2_sld_func, (x, 0, delta_w_dynamic)),
        },
    }

    print("=" * 80)
    print("SYMBOLIC VERIFICATION: MATRIX OF OPTOMECHANICAL INTERROGATION")
    print("=" * 80)

    for name, data in topologies.items():
        print(f"\nTopology: {name}")
        print("-" * 40)

        # Process Laser Equation
        p_laser_raw = data["p_laser_factor"] * data["h_laser"]
        p_laser_expanded = sp.expand(p_laser_raw)

        # Process SLD Equation
        if name == "Dupla reflexão":
            raw_integral = integrate_product_sld(data["sld_func"])
            p_sld_expanded = sp.expand(data["p_sld_base"](raw_integral))
        elif name in ["Transmissão-reflexão", "Reflexão-transmissão"]:
            p_sld_expanded = sp.expand(data["p_sld_base"](None))
        else:  # Dupla transmissão
            p_sld_expanded = sp.expand(data["p_sld_base"](None))

        # Output explicit analytical structures matching Table 2.4
        print("Laser Analytical Expansion (p_pd_laser):")
        sp.pprint(p_laser_expanded)
        print("\nSLD Analytical Expansion (p_pd_sld):")
        sp.pprint(p_sld_expanded)
        print("-" * 80)


if __name__ == "__main__":
    main()
