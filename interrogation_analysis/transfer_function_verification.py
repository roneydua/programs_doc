import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import shift

plt.style.use("common_functions/roney3.mplstyle")
my_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FIG_L = 6.29
FIG_A = FIG_L / 1.6
IMAGE_FOLDER = "./../tese/images/used_on_thesis/"
class fbg_simulation(object):
    def __init__(self, fbg_name_d:str, fbg_name_e:str, use_ideal_model=True, **kwargs):
        # load data from hdf5
        # 1. Cria o vetor de comprimento de onda estático para a simulação
        if "center_w" in kwargs:
            center_w = kwargs["center_w"]
        else:
            center_w = 1550e-9
        if "span" in kwargs:
            span = kwargs["span"]
        else:
            span = 50e-9
        self.w_fbg_d_interp = np.linspace(center_w - span/2, center_w + span/2, 200_000)
        self.w_fbg_e_interp = self.w_fbg_d_interp.copy()

        self.step_of_w_fbg = np.diff(self.w_fbg_d_interp).mean()
        self.sld_total_power = 20e-3
        self.amplitude_w_by_m = self.sld_total_power / 50e-9

        if use_ideal_model:
            # Parâmetros desejados para a validação teórica
            if "r_max_target" in kwargs:
                r_max_target = kwargs["r_max_target"]
            else:
                r_max_target = 1.0
            if "fwhm_target" in kwargs:
                fwhm_target = kwargs["fwhm_target"]
            else:
                fwhm_target =3e-9
            if "split" in kwargs:
                split = kwargs["split"]
            else:       
                split = fwhm_target*.45
                
            peak_e = center_w - split
            peak_d = center_w + split

            # Ordem = 4 cria um perfil trapezoidal suave (apodizado)
            self.fbg_e_interp = self.generate_ideal_fbg(self.w_fbg_e_interp, peak_e, r_max_target, fwhm_target, order=1.)
            self.fbg_d_interp = self.generate_ideal_fbg(self.w_fbg_d_interp, peak_d, r_max_target, fwhm_target, order=1.)
        else:
            f = h5py.File("./production_files.hdf5", "r")
            w_fbg_e_ = f["fbg_production/" + fbg_name_e + "/wavelength_m"][:]
            fbg_e_raw = f["fbg_production/" + fbg_name_e + "/reflectivity"][:, -1]

            w_fbg_d_ = f["fbg_production/" + fbg_name_d + "/wavelength_m"][:]
            fbg_d_raw = f["fbg_production/" + fbg_name_d + "/reflectivity"][:, -1]
            f.close()

            # baseline removal (condition the signal to start at true zero)
            # calculates the median of the bottom 5% of the spectrum to find the noise floor
            baseline_e = np.median(np.sort(fbg_e_raw)[: int(len(fbg_e_raw) * 0.05)])
            baseline_d = np.median(np.sort(fbg_d_raw)[: int(len(fbg_d_raw) * 0.05)])

            fbg_e_ = np.clip(fbg_e_raw - baseline_e, 0, None)
            fbg_d_ = np.clip(fbg_d_raw - baseline_d, 0, None)
            fbg_e_ = self.isolate_main_lobe(fbg_e_)
            fbg_d_ = self.isolate_main_lobe(fbg_d_)

            # interpolation and extension
            self.w_fbg_d_interp, self.w_fbg_d = self.extend_vector(w_fbg_d_, "wavelength")
            self.fbg_d = self.extend_vector(fbg_d_, "reflectivity")
            self.fbg_d_interp = np.interp(self.w_fbg_d_interp, self.w_fbg_d, self.fbg_d)

            self.w_fbg_e_interp, self.w_fbg_e = self.extend_vector(w_fbg_e_, "wavelength")
            self.fbg_e = self.extend_vector(fbg_e_, "reflectivity")
            self.fbg_e_interp = np.interp(self.w_fbg_e_interp, self.w_fbg_e, self.fbg_e)

            self.step_of_w_fbg = np.diff(self.w_fbg_d_interp).mean()
            self.amplitude_w_by_m = 20e-3 / 50e-9

        # extract geometric model parameters
        self.extract_model_parameters()
        # self.calibrate_initial_overlap(target_ratio=1)
        self.setup_laser_interrogation()

    def setup_laser_interrogation(self):
        # Localiza o cruzamento das grades em repouso para fixar o Laser
        peak_e_idx = np.argmax(self.fbg_e_interp)
        peak_d_idx = np.argmax(self.fbg_d_interp)

        idx_min = min(peak_e_idx, peak_d_idx)
        idx_max = max(peak_e_idx, peak_d_idx)

        diff_array = np.abs(
            self.fbg_e_interp[idx_min:idx_max] - self.fbg_d_interp[idx_min:idx_max]
        )
        self.idx_laser = idx_min + np.argmin(diff_array)
        self.w_laser = self.w_fbg_d_interp[self.idx_laser]
        print(self.w_laser)
        # Amplitude da potência de emissão do Laser (ex: 1 mW)
        self.a_l = 1e-3

    def evaluate_laser_power(self):
        # punctual reading of the ramp evolution at the laser wavelength
        r1_num = self.fbg_e_shifted[self.idx_laser]
        r2_num = self.fbg_d_shifted[self.idx_laser]

        # topology 1: single fbg
        self.p_single_laser_num = self.a_l * r1_num

        # topology 2: double reflection
        self.p_dr_laser_num = (1 / 16) * self.a_l * r1_num * r2_num

        # topology 3: transmission-reflection
        self.p_tr_laser_num = (1 / 4) * self.a_l * (1 - r1_num) * r2_num

        # topology 4: double transmission
        self.p_dt_laser_num = self.a_l * (1 - r1_num) * (1 - r2_num)

    def compute_analytical_laser_power(self, delta_lambda_shift: float, delta_t:float=0.0):
        # constants for silica fiber
        alpha = 0.55e-6
        zeta = 8.60e-6
        lambda_center = 1550e-9

        # common mode thermal shift calculation
        shift_t = lambda_center * (alpha + zeta) * delta_t

        a = self.r_max / self.delta_lambda_r
        dw_0 = self.delta_w_0

        # symmetric displacement evaluation for each ramp
        delta_l_1 = (dw_0 / 2) + (delta_lambda_shift +shift_t)
        delta_l_2 = (dw_0 / 2) + (delta_lambda_shift -shift_t)

        r1_ana = a * delta_l_1
        r2_ana = a * delta_l_2

        r1_ana = np.clip(r1_ana, 0, self.r_max)
        r2_ana = np.clip(r2_ana, 0, self.r_max)

        p_single_ana = self.a_l * r1_ana
        p_dr_ana = (1 / 16) * self.a_l * r1_ana * r2_ana
        p_tr_ana = (1 / 4) * self.a_l * (1 - r1_ana) * r2_ana
        p_dt_ana = self.a_l * (1 - r1_ana) * (1 - r2_ana)

        return p_single_ana, p_dr_ana, p_tr_ana, p_dt_ana

    def generate_ideal_fbg(self, w_vector, peak_wavelength, r_max, fwhm, order=4.0):
        """
        Generates an ideal fbg spectrum without side lobes (super-gaussian).
        """
        # calculates the equivalent standard deviation
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # absolute value prevents nan when raising negative numbers to fractional powers
        normalized_w = np.abs((w_vector - peak_wavelength) / sigma)
        # super-gaussian equation
        reflectivity = r_max * np.exp(-0.5 * (normalized_w ** (2.0 * order)))
        return reflectivity

    def isolate_main_lobe(self, reflectivities, threshold_ratio=0.2):
        # extracts the main lobe and suppresses side lobes
        r_max = np.max(reflectivities)
        peak_idx = np.argmax(reflectivities)
        threshold = r_max * threshold_ratio
        cleaned_refs = np.zeros_like(reflectivities)
        # traverse left to find the start of the main lobe
        idx_left = peak_idx
        while idx_left > 0 and reflectivities[idx_left] > threshold:
            idx_left -= 1
        # traverse right to find the end of the main lobe
        idx_right = peak_idx
        while (
            idx_right < len(reflectivities) - 1
            and reflectivities[idx_right] > threshold
        ):
            idx_right += 1
        # preserve only the main lobe data
        cleaned_refs[idx_left : idx_right + 1] = reflectivities[
            idx_left : idx_right + 1
        ]
        # ---------------------------------------------------------------------
        # Smooth Tapering (Roll-off) para eliminar as linhas verticais
        # ---------------------------------------------------------------------
        # Define a largura da cauda de suavização (ex: 15% da largura do lóbulo)
        taper_width = int(0.15 * (idx_right - idx_left))

        # Cauda suave para a borda esquerda (descendo até zero)
        if idx_left - taper_width >= 0:
            # Janela cosseno ao quadrado subindo de 0 a 1
            taper_l = np.sin(np.linspace(0, np.pi / 2, taper_width)) ** 2
            cleaned_refs[idx_left - taper_width : idx_left] = (
                reflectivities[idx_left] * taper_l
            )

        # Cauda suave para a borda direita (descendo até zero)
        if idx_right + taper_width < len(reflectivities):
            # Janela cosseno ao quadrado descendo de 1 a 0
            taper_r = np.cos(np.linspace(0, np.pi / 2, taper_width)) ** 2
            cleaned_refs[idx_right + 1 : idx_right + 1 + taper_width] = (
                reflectivities[idx_right] * taper_r
            )

        return cleaned_refs

    def calibrate_initial_overlap(self, target_ratio=0.5):
        # target_ratio = 0.5 centers the operation in the middle of the ramp
        target_w_0 = self.delta_lambda_r * target_ratio

        # calculate the required mechanical translation in meters
        shift_needed_m = self.delta_w_0 - target_w_0

        # convert translation to array indices
        # moving fbg_e to the left (negative shift) decreases the overlap
        idx_shift = int(shift_needed_m / self.step_of_w_fbg)

        self.fbg_e_interp = shift(self.fbg_e_interp, -idx_shift, mode="grid-wrap")

        print("\n--- After Virtual Mechanical Calibration ---")
        self.extract_model_parameters()

    def extend_vector(self, v: np.ndarray, type_of_data: str):

        if type_of_data == "wavelength":
            _w = np.hstack(
                (v - (v.size * np.diff(v).mean()), v, v + (v.size * np.diff(v).mean()))
            )
            return np.linspace(_w[0], _w[-1], 200_000), _w
        else:
            return np.hstack((np.zeros(v.size), v, np.zeros(v.size)))

    def extract_edge_parameters(self, wavelengths, reflectivities, edge_type):
        r_max = np.max(reflectivities)
        peak_idx = np.argmax(reflectivities)

        # linear region thresholds
        lower_bound = 0.4 * r_max
        upper_bound = 0.6 * r_max

        idx = peak_idx

        if edge_type == "falling":
            # search to the right of the peak (FBG E)
            while idx < len(reflectivities) - 1 and reflectivities[idx] > upper_bound:
                idx += 1
            idx_start = idx

            while idx < len(reflectivities) - 1 and reflectivities[idx] > lower_bound:
                idx += 1
            idx_end = idx

            valid_waves = wavelengths[idx_start : idx_end + 1]
            valid_refs = reflectivities[idx_start : idx_end + 1]

        elif edge_type == "rising":
            # search to the left of the peak (FBG D)
            while idx > 0 and reflectivities[idx] > upper_bound:
                idx -= 1
            idx_end = idx

            while idx > 0 and reflectivities[idx] > lower_bound:
                idx -= 1
            idx_start = idx

            valid_waves = wavelengths[idx_start : idx_end + 1]
            valid_refs = reflectivities[idx_start : idx_end + 1]

        # axis centralization to prevent numerical instability in polyfit
        w_offset = valid_waves.mean()
        shifted_waves = valid_waves - w_offset

        # linear fit: r = a * lambda_shifted + b
        coefficients = np.polyfit(shifted_waves, valid_refs, 1)
        a, b = coefficients

        # calculate geometric x-intercepts
        lambda_zero_shifted = -b / a
        lambda_top_shifted = (r_max - b) / a

        # restore global coordinate system
        lambda_zero = lambda_zero_shifted + w_offset
        lambda_top = lambda_top_shifted + w_offset

        delta_lambda_r = np.abs(lambda_top - lambda_zero)

        return r_max, delta_lambda_r, lambda_zero

    def extract_model_parameters(self):
        # fbg_e is on the left (falling edge is the active one)
        r_max_e, dl_r_e, lambda_f1 = self.extract_edge_parameters(
            self.w_fbg_e_interp, self.fbg_e_interp, "falling"
        )

        # fbg_d is on the right (rising edge is the active one)
        r_max_d, dl_r_d, lambda_i2 = self.extract_edge_parameters(
            self.w_fbg_d_interp, self.fbg_d_interp, "rising"
        )

        # average geometric parameters for the theoretical model
        self.r_max = (r_max_e + r_max_d) / 2.0
        self.delta_lambda_r = (dl_r_e + dl_r_d) / 2.0
        self.delta_w_0 = lambda_f1 - lambda_i2

        print(f"Geometric Parameters Extracted:")
        print(f"R_max = {self.r_max:.4f}")
        print(f"Delta_lambda_r = {self.delta_lambda_r * 1e9:.4f} nm")
        print(f"Delta_w_0 = {self.delta_w_0 * 1e9:.4f} nm")

    def translate_fbgs(self, delta_lambda_shift: float, shift_t:float=0.0):
        # constants for silica fiber
        alpha = 0.55e-6
        zeta = 8.60e-6
        lambda_center = 1550e-9
        shift_t = lambda_center * (alpha+zeta)*shift_t
        # fbg_e (left) moves right to increase overlap
        index_to_shift_e = int((delta_lambda_shift+shift_t) / self.step_of_w_fbg)

        # fbg_d (right) moves left to increase overlap
        index_to_shift_d = int((-delta_lambda_shift+shift_t) / self.step_of_w_fbg)

        self.fbg_d_shifted = shift(
            self.fbg_d_interp, index_to_shift_d, mode="grid-wrap"
        )
        self.fbg_e_shifted = shift(
            self.fbg_e_interp, index_to_shift_e, mode="grid-wrap"
        )

        self.make_dot_product()

        # evaluates punctual intersection for laser interrogation
        self.evaluate_laser_power()

    def make_dot_product(self):
        self.first_reflected_spectrum = self.fbg_e_shifted
        self.second_reflected_spectrum = (
            self.first_reflected_spectrum * self.fbg_d_shifted
        )

        self.cross_integral_num = np.trapezoid(
            y=self.second_reflected_spectrum, dx=self.step_of_w_fbg
        )

        # topology 2: double reflection
        self.p_dr_num = (1 / 16) * self.amplitude_w_by_m * self.cross_integral_num

        # topology 3: transmission-reflection
        tr_integrand = (1 - self.fbg_e_shifted) * self.fbg_d_shifted
        self.p_tr_num = (
            (1 / 4)
            * self.amplitude_w_by_m
            * np.trapezoid(y=tr_integrand, dx=self.step_of_w_fbg)
        )

        # topology 4: double transmission
        dt_integrand = (1 - self.fbg_e_shifted) * (1 - self.fbg_d_shifted)
        self.p_dt_num = self.amplitude_w_by_m * np.trapezoid(
            y=dt_integrand, dx=self.step_of_w_fbg
        )

    def compute_analytical_cross_integral(self, delta_lambda_shift: float,delta_t:float=0.0):
        # dynamic integration base (push-pull)
        delta_w = self.delta_w_0 + 2 * delta_lambda_shift

        if delta_w <= 0:
            return 0.0

        r_max = self.r_max
        dl_r = self.delta_lambda_r
        a = r_max / dl_r

        if delta_w <= dl_r:
            integral = (a**2 / 6) * (delta_w**3)
        elif delta_w <= 2 * dl_r:
            lim_inf = delta_w - dl_r
            lim_sup = dl_r
            area_flats = 2 * (r_max * a * (lim_inf**2) / 2)

            def int_ramps(x):
                return a**2 * (delta_w * (x**2) / 2 - (x**3) / 3)

            area_ramps = int_ramps(lim_sup) - int_ramps(lim_inf)
            integral = area_flats + area_ramps
        else:
            integral = (r_max**2) * (delta_w - dl_r)

        return self.amplitude_w_by_m * integral

    def compute_analytical_power(self, delta_lambda_shift: float):
        # dynamic integration base (push-pull)
        delta_w = self.delta_w_0 + 2 * delta_lambda_shift

        if delta_w <= 0:
            return 0.0

        r_max = self.r_max
        dl_r = self.delta_lambda_r
        a = r_max / dl_r

        if delta_w <= dl_r:
            # Regime 1: Intersecção pura nas rampas (Equação Cúbica da Tese)
            # Válido para pequenas sobreposições
            integral = (a**2 / 6) * (delta_w**3)

        elif delta_w <= 2 * dl_r:
            # Regime 2: Transição (As rampas cruzam os topos planos)
            # A área é a soma de duas extremidades (flat * ramp) e um centro (ramp * ramp)
            lim_inf = delta_w - dl_r
            lim_sup = dl_r

            # Integral das bordas onde uma rede é plana e a outra é rampa
            area_flats = 2 * (r_max * a * (lim_inf**2) / 2)

            # Integral do miolo onde as duas redes ainda são rampas cruzadas
            def int_ramps(x):
                return a**2 * (delta_w * (x**2) / 2 - (x**3) / 3)

            area_ramps = int_ramps(lim_sup) - int_ramps(lim_inf)
            integral = area_flats + area_ramps

        else:
            # Regime 3: Saturação profunda (Topos planos sobrepostos)
            # Comportamento estritamente linear
            integral = (r_max**2) * (delta_w - dl_r)

        return self.amplitude_w_by_m * integral

    def plot_linear_approximation(self):
        # define the search region between the two peaks to find the exact intersection
        peak_e_idx = np.argmax(self.fbg_e_interp)
        peak_d_idx = np.argmax(self.fbg_d_interp)

        idx_min = min(peak_e_idx, peak_d_idx)
        idx_max = max(peak_e_idx, peak_d_idx)

        # locate the crossing point at rest
        diff_array = np.abs(
            self.fbg_e_interp[idx_min:idx_max] - self.fbg_d_interp[idx_min:idx_max]
        )
        cross_idx_relative = np.argmin(diff_array)
        cross_idx_global = idx_min + cross_idx_relative

        w_cross = self.w_fbg_d_interp[cross_idx_global]
        r_cross = self.fbg_d_interp[cross_idx_global]

        # calculate the geometric slope
        slope_magnitude = self.r_max / self.delta_lambda_r

        # left fbg (falling edge) linear equation
        r_approx_e = -slope_magnitude * (self.w_fbg_d_interp - w_cross) + r_cross
        r_approx_e = np.clip(r_approx_e, 0, self.r_max)

        # right fbg (rising edge) linear equation
        r_approx_d = slope_magnitude * (self.w_fbg_d_interp - w_cross) + r_cross
        r_approx_d = np.clip(r_approx_d, 0, self.r_max)

        # absolute coordinates extraction for geometric annotations [nm]
        w_cross_nm = w_cross * 1e9
        lam_i2_nm = w_cross_nm - (r_cross / slope_magnitude) * 1e9
        lam_f1_nm = w_cross_nm + (r_cross / slope_magnitude) * 1e9
        lam_top_e_nm = lam_f1_nm - self.delta_lambda_r * 1e9

        # plot generation
        fig, ax = plt.subplots(figsize=(FIG_L, FIG_A), dpi=288)

        ax.plot(
            self.w_fbg_d_interp * 1e9,
            self.fbg_e_interp,
            color=my_colors[0],
            label="FBG 1",
        )
        ax.plot(
            self.w_fbg_d_interp * 1e9,
            self.fbg_d_interp,
            color=my_colors[2],
            label="FBG 2",
        )

        ax.plot(
            self.w_fbg_d_interp * 1e9,
            r_approx_e,
            color=my_colors[0],
            alpha=0.5,
            linewidth=2,
            label="Aprox. FBG 1",
        )
        ax.plot(
            self.w_fbg_d_interp * 1e9,
            r_approx_d,
            color=my_colors[2],
            alpha=0.5,
            linewidth=2,
            label="Aprox. FBG 2",
        )

        ax.plot(w_cross_nm, r_cross, my_colors[1], marker="o", markersize=4, label="Ponto de operação")

        # --- geometric annotations ---
        # 2. Delta_lambda_r annotation (width of the falling ramp)
        ax.annotate(
            "",
            xy=(lam_top_e_nm, self.r_max + 0.02),
            xytext=(lam_f1_nm, self.r_max + 0.02),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color=my_colors[1]),
        )
        ax.text(
            (lam_f1_nm + lam_top_e_nm) / 2,
            self.r_max + 0.04,
            r"$\Delta\lambda_r$",
            color=my_colors[1],
            ha="center",
        )

        # 3. Delta_w_0 annotation (initial overlap at y=0)
        ax.annotate(
            "",
            xy=(lam_i2_nm, -0.015),
            xytext=(lam_f1_nm, -0.015),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color=my_colors[1]),
        )
        ax.text(
            w_cross_nm,
            -0.03,
            r"$\Delta w_0$",
            color=my_colors[1],
            ha="center",
            va="top",
        )

        # 4. Lambda intersection boundaries
        ax.plot(lam_i2_nm, 0, marker="o", color=my_colors[2], markersize=6)
        ax.plot(lam_f1_nm, 0, marker="o", color=my_colors[0], markersize=6)

        ax.text(
            lam_i2_nm,
            -0.06,
            r"$\lambda_{i_2}$",
            color=my_colors[2],
            ha="right",
        )
        ax.text(
            lam_f1_nm, -0.06, r"$\lambda_{f_1}$", color=my_colors[0], ha="left"
        )

        # axes formatting
        ax.set_ylim(-0.1, self.r_max + 0.05)
        ax.set_xlim(w_cross_nm - 4, w_cross_nm + 4)
        ax.set_xlabel(r"Comprimento de onda")
        ax.set_ylabel(r"Refletividade $r(\lambda)$")

        # hide top and right spines
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_position("zero")

        # remove x ticks inside the bounding box
        ax.set_xticks([])
        ax.set_yticks([0, self.r_max])
        ax.set_yticklabels(["", r"$R_{max}$"])

        ax.legend(loc="upper right")

        plt.savefig(IMAGE_FOLDER+"aproximacao_linear_ideal_da_borda_fbg.png", format="png")
        plt.close(fig=1)


def run_deformation_topology_comparison():
    sim = fbg_simulation("20240207/fbg9", "20231130/fbg7", use_ideal_model=True)

    shifts = np.linspace(-1e-9, 1e-9, 50)
    accel_sensitivity_per_fbg = (83e-12) / 2.0

    # arrays for laser
    num_laser_single, num_laser_dr, num_laser_tr, num_laser_dt = [], [], [], []
    ana_laser_single, ana_laser_dr, ana_laser_tr, ana_laser_dt = [], [], [], []

    # arrays for sld
    num_sld_dr, num_sld_tr, num_sld_dt = [], [], []
    ana_sld_cross = []

    for shift_val in shifts:
        sim.translate_fbgs(shift_val)

        num_laser_single.append(sim.p_single_laser_num)
        num_laser_dr.append(sim.p_dr_laser_num)
        num_laser_tr.append(sim.p_tr_laser_num)
        num_laser_dt.append(sim.p_dt_laser_num)

        p_s, p_dr, p_tr, p_dt = sim.compute_analytical_laser_power(shift_val)
        ana_laser_single.append(p_s)
        ana_laser_dr.append(p_dr)
        ana_laser_tr.append(p_tr)
        ana_laser_dt.append(p_dt)

        num_sld_dr.append(sim.p_dr_num)
        num_sld_tr.append(sim.p_tr_num)
        num_sld_dt.append(sim.p_dt_num)

        ana_sld_cross.append(sim.compute_analytical_cross_integral(shift_val))

    acceleration_g = shifts / accel_sensitivity_per_fbg
    ana_sld_cross = np.array(ana_sld_cross)

    # absolute analytical formulation for sld
    area_fbg = sim.amplitude_w_by_m * np.trapezoid(
        sim.fbg_e_interp, dx=sim.step_of_w_fbg
    )
    span_total = sim.amplitude_w_by_m * np.trapezoid(
        np.ones_like(sim.fbg_e_interp), dx=sim.step_of_w_fbg
    )

    ana_sld_dr = (1 / 16) * ana_sld_cross
    ana_sld_tr = (1 / 4) * (area_fbg - ana_sld_cross)
    ana_sld_dt = span_total - 2 * area_fbg + ana_sld_cross

    # figure 1: laser
    fig_laser, ax_laser = plt.subplots(figsize=(FIG_L, FIG_A), dpi=144)
    ax_laser_dr = ax_laser.twinx()
    ax_laser_tr = ax_laser.twinx()
    ax_laser_dt = ax_laser.twinx()

    # Offset the right spines of ax_laser_tr and ax_laser_dt
    ax_laser_tr.spines["right"].set_position(("outward", 40))
    ax_laser_dt.spines["right"].set_position(("outward", 80))

    # FBG Única on ax_laser (left)
    ax_laser.plot(
        acceleration_g,
        np.array(num_laser_single) * 1e3,
        color=my_colors[0],
        linewidth=4,
        alpha=0.5,
        label="Num: FBG Única",
    )
    ax_laser.plot(
        acceleration_g,
        np.array(ana_laser_single) * 1e3,
        color=my_colors[0],
        linestyle="--",
        linewidth=2,
        label="Ana: FBG Única",
    )
    ax_laser.set_ylabel(r"FBG Única [\unit{\milli\watt}]", color=my_colors[0])
    ax_laser.tick_params(axis='y', labelcolor=my_colors[0])

    # Dupla Reflexão on ax_laser_dr
    ax_laser_dr.plot(
        acceleration_g,
        np.array(num_laser_dr) * 1e3,
        color=my_colors[1],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Reflexão",
    )
    ax_laser_dr.plot(
        acceleration_g,
        np.array(ana_laser_dr) * 1e3,
        color=my_colors[1],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Reflexão",
    )
    ax_laser_dr.set_ylabel(r"Dupla Reflexão [\unit{\milli\watt}]", color=my_colors[1])
    ax_laser_dr.tick_params(axis='y', labelcolor=my_colors[1])

    # Transmissão-Reflexão on ax_laser_tr
    ax_laser_tr.plot(
        acceleration_g,
        np.array(num_laser_tr) * 1e3,
        color=my_colors[2],
        linewidth=4,
        alpha=0.5,
        label="Num: Transmissão-Reflexão",
    )
    ax_laser_tr.plot(
        acceleration_g,
        np.array(ana_laser_tr) * 1e3,
        color=my_colors[2],
        linestyle="--",
        linewidth=2,
        label="Ana: Transmissão-Reflexão",
    )
    ax_laser_tr.set_ylabel(r"Transmissão-Reflexão [\unit{\milli\watt}]", color=my_colors[2])
    ax_laser_tr.tick_params(axis='y', labelcolor=my_colors[2])

    # Dupla Transmissão on ax_laser_dt
    ax_laser_dt.plot(
        acceleration_g,
        np.array(num_laser_dt) * 1e3,
        color=my_colors[3],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Transmissão",
    )
    ax_laser_dt.plot(
        acceleration_g,
        np.array(ana_laser_dt) * 1e3,
        color=my_colors[3],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Transmissão",
    )
    ax_laser_dt.set_ylabel(r"Dupla Transmissão [\unit{\milli\watt}]", color=my_colors[3])
    ax_laser_dt.tick_params(axis='y', labelcolor=my_colors[3])

    ax_laser.set_xlabel(r"Aceleração Inercial [$g$]")

    # Legend consolidation
    h1, l1 = ax_laser.get_legend_handles_labels()
    h2, l2 = ax_laser_dr.get_legend_handles_labels()
    h3, l3 = ax_laser_tr.get_legend_handles_labels()
    h4, l4 = ax_laser_dt.get_legend_handles_labels()
    # ax_laser.legend(h1 + h2 + h3 + h4, l1 + l2 + l3 + l4, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=8)
    ax_laser_dr.spines["right"].set_visible(True)
    ax_laser_tr.spines["right"].set_visible(True)
    ax_laser_dt.spines["right"].set_visible(True)

    ax_laser.set_xlim(-10, 10)
    fig_laser.savefig(IMAGE_FOLDER+"transfer_function_laser_deformation.png",format="png")

    # figure 2: sld
    fig_sld, ax_sld = plt.subplots(figsize=(FIG_L, FIG_A), dpi=144)
    ax_sld_tr = ax_sld.twinx()
    ax_sld_dt = ax_sld.twinx()

    # Offset the right spine of ax_sld_dt
    ax_sld_dt.spines["right"].set_position(("outward", 40))

    # Dupla Reflexão on ax_sld (left)
    ax_sld.plot(
        acceleration_g, np.array(num_sld_dr) * 1e3,
        color=my_colors[0], linewidth=4, alpha=0.5, label="Num: Dupla Reflexão",
    )
    ax_sld.plot(
        acceleration_g, ana_sld_dr * 1e3,
        color=my_colors[0], linestyle="--", linewidth=2, label="Ana: Dupla Reflexão",
    )
    ax_sld.set_ylabel(r"Dupla Reflexão [\unit{\milli\watt}]", color=my_colors[0])
    ax_sld.tick_params(axis='y', labelcolor=my_colors[0])

    # Transmissão-Reflexão on ax_sld_tr
    ax_sld_tr.plot(
        acceleration_g, np.array(num_sld_tr) * 1e3,
        color=my_colors[1], linewidth=4, alpha=0.5, label="Num: Transmissão-Reflexão",
    )
    ax_sld_tr.plot(
        acceleration_g, ana_sld_tr * 1e3,
        color=my_colors[1], linestyle="--", linewidth=2, label="Ana: Transmissão-Reflexão",
    )
    ax_sld_tr.set_ylabel(r"Transmissão-Reflexão [\unit{\milli\watt}]", color=my_colors[1])
    ax_sld_tr.tick_params(axis='y', labelcolor=my_colors[1])

    # Dupla Transmissão on ax_sld_dt
    ax_sld_dt.plot(
        acceleration_g, np.array(num_sld_dt) * 1e3,
        color=my_colors[2], linewidth=4, alpha=0.5, label="Num: Dupla Transmissão",
    )
    ax_sld_dt.plot(
        acceleration_g, ana_sld_dt * 1e3,
        color=my_colors[2], linestyle="--", linewidth=2, label="Ana: Dupla Transmissão",
    )
    ax_sld_dt.set_ylabel(r"Dupla Transmissão [\unit{\milli\watt}]", color=my_colors[2])
    ax_sld_dt.tick_params(axis='y', labelcolor=my_colors[2])

    ax_sld.set_xlabel(r"Aceleração Inercial [$g$]")
    ax_sld.set_ylim(bottom=0)
    # Legend consolidation
    h1, l1 = ax_sld.get_legend_handles_labels()
    h2, l2 = ax_sld_tr.get_legend_handles_labels()
    h3, l3 = ax_sld_dt.get_legend_handles_labels()
    # ax_sld.legend(h1 + h2 + h3, l1 + l2 + l3, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=8)
    ax_sld_tr.spines["right"].set_visible(True)
    ax_sld_dt.spines["right"].set_visible(True)
    ax_sld.set_xlim(-10,10)
    fig_sld.savefig(IMAGE_FOLDER+"transfer_function_sld_deformation.png",format="png")
    # Sensitivity calculation and CSV generation
    # We evaluate exactly at -10g, 0g, and +10g for precision
    def get_topology_powers(g_val):
        s = g_val * accel_sensitivity_per_fbg
        sim.translate_fbgs(s)
        return {
            "laser_dr": sim.p_dr_laser_num,
            "laser_tr": sim.p_tr_laser_num,
            "laser_dt": sim.p_dt_laser_num,
            "sld_dr": sim.p_dr_num,
            "sld_tr": sim.p_tr_num,
            "sld_dt": sim.p_dt_num,
        }

    p_m10 = get_topology_powers(-10.0)
    p_0g = get_topology_powers(0.0)
    p_p10 = get_topology_powers(10.0)

    for source in ["laser", "sld"]:
        print(source.upper())
        print("topo\t source\t delta_p\t p_dc\t  p_dc/p_s\t delta_p/p_dc")
        p_s = sim.a_l if source == "laser" else sim.sld_total_power
        for topo in ["dr", "tr", "dt"]:
            key = f"{source}_{topo}"
            p_dc = p_0g[key]
            delta_p = np.abs(p_p10[key] - p_m10[key])

            # p_dc_over_dp = p_dc / delta_p if delta_p != 0 else np.inf
            p_dc_over_ps = p_dc / p_s
            dp_over_pdc = delta_p / p_dc

            print(
                f"{topo.upper()}\t {source}\t {1e6*delta_p:.6e}\t {1e6*p_dc:.6e}\t {100*p_dc_over_ps:.6f}\t {100*dp_over_pdc:.6f}"
            )


def run_temperature_sweep_comparison():
    # initialize simulation with ideal trapezoidal model
    sim = fbg_simulation("20240207/fbg9", "20231130/fbg7", use_ideal_model=True)

    # temperature variation from -20 to 60 degrees celsius
    temperatures = np.linspace(-20.0, 40.0, 20)

    # constant strain (zero acceleration)
    fixed_shift_val = 0.0

    # arrays for laser
    num_laser_single, num_laser_dr, num_laser_tr, num_laser_dt = [], [], [], []
    ana_laser_single, ana_laser_dr, ana_laser_tr, ana_laser_dt = [], [], [], []

    # arrays for sld
    num_sld_dr, num_sld_tr, num_sld_dt = [], [], []
    ana_sld_cross = []

    for temp in temperatures:
        sim.translate_fbgs(fixed_shift_val, shift_t=temp)

        num_laser_single.append(sim.p_single_laser_num)
        num_laser_dr.append(sim.p_dr_laser_num)
        num_laser_tr.append(sim.p_tr_laser_num)
        num_laser_dt.append(sim.p_dt_laser_num)

        p_s, p_dr, p_tr, p_dt = sim.compute_analytical_laser_power(
            fixed_shift_val, delta_t=temp
        )
        ana_laser_single.append(p_s)
        ana_laser_dr.append(p_dr)
        ana_laser_tr.append(p_tr)
        ana_laser_dt.append(p_dt)

        num_sld_dr.append(sim.p_dr_num)
        num_sld_tr.append(sim.p_tr_num)
        num_sld_dt.append(sim.p_dt_num)

        # analytical area integral is independent of temperature shift
        ana_sld_cross.append(sim.compute_analytical_cross_integral(fixed_shift_val))

    ana_sld_cross = np.array(ana_sld_cross)

    # absolute analytical formulation for sld
    area_fbg = sim.amplitude_w_by_m * np.trapezoid(
        sim.fbg_e_interp, dx=sim.step_of_w_fbg
    )
    span_total = sim.amplitude_w_by_m * np.trapezoid(
        np.ones_like(sim.fbg_e_interp), dx=sim.step_of_w_fbg
    )

    ana_sld_dr = (1 / 16) * ana_sld_cross
    ana_sld_tr = (1 / 4) * (area_fbg - ana_sld_cross)
    ana_sld_dt = span_total - 2 * area_fbg + ana_sld_cross

    # figure 1: laser
    fig_laser, ax_laser = plt.subplots(figsize=(FIG_L, FIG_A), dpi=144)
    ax_laser_dr = ax_laser.twinx()
    ax_laser_tr = ax_laser.twinx()
    ax_laser_dt = ax_laser.twinx()

    # Offset the right spines of ax_laser_tr and ax_laser_dt
    ax_laser_tr.spines["right"].set_position(("outward", 40))
    ax_laser_dt.spines["right"].set_position(("outward", 80))

    # FBG Única on ax_laser (left)
    ax_laser.plot(
        temperatures,
        np.array(num_laser_single) * 1e3,
        color=my_colors[0],
        linewidth=4,
        alpha=0.5,
        label="Num: FBG Única",
    )
    ax_laser.plot(
        temperatures,
        np.array(ana_laser_single) * 1e3,
        color=my_colors[0],
        linestyle="--",
        linewidth=2,
        label="Ana: FBG Única",
    )
    ax_laser.set_ylabel(r"FBG Única [\unit{\milli\watt}]", color=my_colors[0])
    ax_laser.tick_params(axis="y", labelcolor=my_colors[0])

    # Dupla Reflexão on ax_laser_dr
    ax_laser_dr.plot(
        temperatures,
        np.array(num_laser_dr) * 1e3,
        color=my_colors[1],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Reflexão",
    )
    ax_laser_dr.plot(
        temperatures,
        np.array(ana_laser_dr) * 1e3,
        color=my_colors[1],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Reflexão",
    )
    ax_laser_dr.set_ylabel(r"Dupla Reflexão [\unit{\milli\watt}]", color=my_colors[1])
    ax_laser_dr.tick_params(axis="y", labelcolor=my_colors[1])

    # Transmissão-Reflexão on ax_laser_tr
    ax_laser_tr.plot(
        temperatures,
        np.array(num_laser_tr) * 1e3,
        color=my_colors[2],
        linewidth=4,
        alpha=0.5,
        label="Num: Transmissão-Reflexão",
    )
    ax_laser_tr.plot(
        temperatures,
        np.array(ana_laser_tr) * 1e3,
        color=my_colors[2],
        linestyle="--",
        linewidth=2,
        label="Ana: Transmissão-Reflexão",
    )
    ax_laser_tr.set_ylabel(
        r"Transmissão-Reflexão [\unit{\milli\watt}]", color=my_colors[2]
    )
    ax_laser_tr.tick_params(axis="y", labelcolor=my_colors[2])

    # Dupla Transmissão on ax_laser_dt
    ax_laser_dt.plot(
        temperatures,
        np.array(num_laser_dt) * 1e3,
        color=my_colors[3],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Transmissão",
    )
    ax_laser_dt.plot(
        temperatures,
        np.array(ana_laser_dt) * 1e3,
        color=my_colors[3],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Transmissão",
    )
    ax_laser_dt.set_ylabel(
        r"Dupla Transmissão [\unit{\milli\watt}]", color=my_colors[3]
    )
    ax_laser_dt.tick_params(axis="y", labelcolor=my_colors[3])

    ax_laser.set_xlabel(r"Variação de temperatura $\Delta T$ [\unit{\degreeCelsius}]")

    ax_laser_dr.spines["right"].set_visible(True)
    ax_laser_tr.spines["right"].set_visible(True)
    ax_laser_dt.spines["right"].set_visible(True)
    fig_laser.savefig(
        IMAGE_FOLDER + "laser_vs_temperature_interrogation_analysis.png", format="png"
    )

    # figure 2: sld
    fig_sld, ax_sld = plt.subplots(figsize=(FIG_L, FIG_A), dpi=144)
    ax_sld_tr = ax_sld.twinx()
    ax_sld_dt = ax_sld.twinx()

    # Offset the right spine of ax_sld_dt
    ax_sld_dt.spines["right"].set_position(("outward", 40))

    # Dupla Reflexão on ax_sld (left)
    ax_sld.plot(
        temperatures,
        np.array(num_sld_dr) * 1e3,
        color=my_colors[0],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Reflexão",
    )
    ax_sld.plot(
        temperatures,
        ana_sld_dr * 1e3,
        color=my_colors[0],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Reflexão",
    )
    ax_sld.set_ylabel(r"Dupla Reflexão [\unit{\milli\watt}]", color=my_colors[0])
    ax_sld.tick_params(axis="y", labelcolor=my_colors[0])

    # Transmissão-Reflexão on ax_sld_tr
    ax_sld_tr.plot(
        temperatures,
        np.array(num_sld_tr) * 1e3,
        color=my_colors[1],
        linewidth=4,
        alpha=0.5,
        label="Num: Transmissão-Reflexão",
    )
    ax_sld_tr.plot(
        temperatures,
        ana_sld_tr * 1e3,
        color=my_colors[1],
        linestyle="--",
        linewidth=2,
        label="Ana: Transmissão-Reflexão",
    )
    ax_sld_tr.set_ylabel(
        r"Transmissão-Reflexão [\unit{\milli\watt}]", color=my_colors[1]
    )
    ax_sld_tr.tick_params(axis="y", labelcolor=my_colors[1])

    # Dupla Transmissão on ax_sld_dt
    ax_sld_dt.plot(
        temperatures,
        np.array(num_sld_dt) * 1e3,
        color=my_colors[2],
        linewidth=4,
        alpha=0.5,
        label="Num: Dupla Transmissão",
    )
    ax_sld_dt.plot(
        temperatures,
        ana_sld_dt * 1e3,
        color=my_colors[2],
        linestyle="--",
        linewidth=2,
        label="Ana: Dupla Transmissão",
    )
    ax_sld_dt.set_ylabel(r"Dupla Transmissão [\unit{\milli\watt}]", color=my_colors[2])
    ax_sld_dt.tick_params(axis="y", labelcolor=my_colors[2])

    ax_sld.set_xlabel(r"Variação de temperatura $\Delta T$ [\unit{\degreeCelsius}]")
    ax_sld_tr.spines["right"].set_visible(True)
    ax_sld_dt.spines["right"].set_visible(True)

    fig_sld.savefig(
        IMAGE_FOLDER + "sld_vs_temerature_interrogation_analysis.png", format="png"
    )


def plot_linear_approximation():
    sim = fbg_simulation("20240207/fbg9", "20231130/fbg7", use_ideal_model=True)
    sim.plot_linear_approximation()

if __name__ == "__main__":
    # run_temperature_sweep_comparison()
    run_deformation_topology_comparison()
    # plot_linear_approximation()
