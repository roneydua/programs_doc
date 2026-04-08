import numpy as np
import pandas as pd
from modeling.accel_model import accel_model_euler_poincare
from matplotlib import pyplot as plt


def apply_thermokinematic_perturbation(
    case: str,
    apply_perturbation: bool = True,
    input_file: str = "./modeling/data/modeling.h5",
    output_file: str = "./modeling/data/modeling.h5",
    noise_std:float=1e-9,  # standard deviation of the optical interrogator noise (e.g., 1 nm)
):
    """
    reads the clean kinematic simulation data, applies a time-varying thermal gradient,
    injects white gaussian noise, and exports the corrupted data to be read by the estimator.
    """
    print(f"loading clean simulation data from {input_file}...")
    df_sim = pd.read_hdf(input_file, key=f"{case}/simulation")
    time = df_sim["time"].values
    num_steps = len(time)

    # instantiate the geometric model to get nominal lengths (l_0)
    model = accel_model_euler_poincare()
    # get initial length assuming initial alli
    l_0 = np.zeros(12)
    for j in range(12):
        d_j = model.m_m[j] - model.b_b[j]
        l_0[j] = np.linalg.norm(d_j)
    # optical and thermal properties of silica
    p_e = 0.22                  # effective photoelastic coefficient
    alpha = 0.55e-6             # thermal expansion coefficient (1/°C)
    zeta = 8.60e-6              # thermo-optic coefficient (1/°C)

    # thermal-kinematic sensitivity constant (K_T)
    k_t = (alpha + zeta) / (1.0 - p_e)

    # define the temperature profile (delta_t) over time.
    # example: a linear ramp from 0 C to 30C
    # delta_t_profile = np.linspace(0.0, 30.0, num_steps)

    if apply_perturbation:
        delta_t_profile =50/time[-1] * time  * np.sin(2.0 * np.pi * 2.0 * time)
    else:
        delta_t_profile = np.zeros(num_steps)
    df_perturbed = df_sim.copy()

    if plt.fignum_exists(3):
        plt.close(3)
    # fig, ax = plt.subplots(nrows=4, ncols=3, num=3 ,sharey=True,sharex=True)
    # _ax = ax.flatten()
    print("injecting thermal expansion and white noise...")
    for j in range(12):
        col_name = f"fiber_{j+1}_length"
        clean_lengths = df_sim[col_name].values

        # calculate the apparent elongation caused by temperature
        thermal_elongation = l_0[j] * k_t * delta_t_profile

        # generate white gaussian noise for this specific fiber
        white_noise = np.random.normal(loc=0.0, scale=noise_std, size=num_steps)
        if apply_perturbation:
            # superimpose the true kinematics with thermal error and noise
            corrupted_lengths = clean_lengths + thermal_elongation + white_noise
        else:
            corrupted_lengths = clean_lengths
        # overwrite the column with the corrupted data
        df_perturbed[col_name] = corrupted_lengths
        # _ax[j].plot(time, df_sim[col_name].values-l_0[j], label="clean")
        # _ax[j].plot(time, df_perturbed[col_name].values - l_0[j], label="perturbed")
        # _ax[j].set_ylabel(col_name)
        # _ax[j].legend()
    # save the true temperature profile in the h5 so we can plot and compare later!
    df_perturbed["dT_true"] = delta_t_profile
    # plt.show()
    df_perturbed.to_hdf(
        output_file, key=f"{case}/simulation_output_perturbed", mode="a"
    )
    print(f"corrupted data successfully saved to {output_file}.")
    print(f"applied K_T = {k_t:.4e} 1/C with noise std = {noise_std:.1e} m.")


if __name__ == "__main__":
    # execute the perturbation script
    # you can adjust the noise level here. 1e-9 meters = 1 nanometer resolution.
    apply_thermokinematic_perturbation(case="perturbed", noise_std=2e-9)
