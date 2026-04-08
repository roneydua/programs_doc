"""
main_modeling.py
Author: Roney Silva
Contact: roneyddasilva@gmail
Description: main file to run the modeling pipeline
"""

from modeling.main_dynamics import simulate_dynamics
from modeling.inverse_problem_closed_form import solve_inverse_problem_closed_form
from modeling.optical_push_pull import solve_inverse_problem_optical_push_pull
from modeling.add_thermal_noise import apply_thermokinematic_perturbation
from modeling.graphics_new_approach import plot_graphics
import argparse

APPLY_PERTURBATION = True
if APPLY_PERTURBATION:
    CASE_NAME = "sinusoidal_with_temp_perturbation"
else:
    CASE_NAME = "sinusoidal_without_temp_perturbation"

def run_full_pipeline():
    # Bloco 1: Dinâmica
    print("integrate dynamics")
    simulate_dynamics(case=CASE_NAME, t_end=3, dt=1e-2)

    print("add pertubation on data")
    add_perturbations()
    # Bloco 2: Estimação (Inverse Problem)


def add_perturbations():
    apply_thermokinematic_perturbation(apply_perturbation=APPLY_PERTURBATION,case=CASE_NAME, noise_std=10e-12)
    run_estimation()


def run_estimation():
    print("solve inverse problem")
    # solve_inverse_problem_closed_form(case=CASE_NAME,mmq_mode=1)
    solve_inverse_problem_closed_form(
        case=CASE_NAME, mmq_mode=2, active_fibers=[0, 1, 4, 5, 8, 9, 11]
    )
    solve_inverse_problem_closed_form(
        case=CASE_NAME, mmq_mode=3, active_fibers=[0, 1, 4, 5, 8, 9, 11]
    )

    print("done")
    print("Solving optical push-pull")
    solve_inverse_problem_optical_push_pull(
        case=CASE_NAME,
        push_pull_pairs=[[0, 3], [4, 7], [8, 11]],
        name_save="inverse_output_optical_push_pull_cruzed",
    )
    solve_inverse_problem_optical_push_pull(
        case=CASE_NAME,
        push_pull_pairs=[[0, 2], [4, 6], [8, 10]],
        name_save="inverse_output_optical_push_pull_aligned",
    )
    plot()


def plot():
    plot_graphics(case=CASE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de simulação e inversão.")

    # Criamos um grupo mutuamente exclusivo: ou roda um, ou roda outro
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Roda a pipeline completa")
    group.add_argument(
        "--perturbation", action="store_true", help="add perturbation on fiber lengths"
    )
    group.add_argument(
        "--estimation", action="store_true", help="Run all estimation inverso"
    )
    group.add_argument("--graphics", action="store_true", help="Graphics only")

    args = parser.parse_args()

    if args.all:
        run_full_pipeline()
    elif args.perturbation:
        add_perturbations()
    elif args.estimation:
        run_estimation()
    elif args.graphics:
        plot()
