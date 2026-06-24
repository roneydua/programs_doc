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

def run_dynamics():
    case_1 = "all_faces_with_equal_temp"
    print(f"--- integrate dynamics for {case_1} ---")
    simulate_dynamics(case=case_1, t_end=3, dt=1e-2)

    case_2 = "temp_gradient_analisys"
    print(f"--- integrate dynamics for {case_2} ---")
    simulate_dynamics(case=case_2, t_end=3, dt=1e-2)

def add_perturbations():
    case_1 = "all_faces_with_equal_temp"
    print(f"--- add perturbation on data for {case_1} ---")
    apply_thermokinematic_perturbation(
        apply_perturbation=True, case=case_1, noise_std=10e-12, mode='7dof'
    )

    case_2 = "temp_gradient_analisys"
    print(f"--- add perturbation on data for {case_2} ---")
    apply_thermokinematic_perturbation(
        apply_perturbation=True, case=case_2, noise_std=10e-12, mode='12dof'
    )

def run_estimation():
    # Frente 1: all_faces_with_equal_temp
    case_1 = "all_faces_with_equal_temp"
    print(f"--- Running Estimation for Frente 1: {case_1} ---")
    solve_inverse_problem_closed_form(
        case=case_1, mmq_mode=2, active_fibers=[0, 1, 4, 5, 8, 9, 11]
    )
    solve_inverse_problem_closed_form(
        case=case_1, mmq_mode=3, active_fibers=[0, 1, 4, 5, 8, 9, 11]
    )
    solve_inverse_problem_optical_push_pull(
        case=case_1,
        push_pull_pairs=[[0, 3], [4, 7], [8, 11]],
        name_save="inverse_output_optical_push_pull_cruzed",
    )
    solve_inverse_problem_optical_push_pull(
        case=case_1,
        push_pull_pairs=[[0, 2], [4, 6], [8, 10]],
        name_save="inverse_output_optical_push_pull_aligned",
    )

    # Frente 2: temp_gradient_analisys
    case_2 = "temp_gradient_analisys"
    print(f"--- Running Estimation for Frente 2: {case_2} ---")
    solve_inverse_problem_closed_form(
        case=case_2, mmq_mode=3, active_fibers=[0, 1, 4, 5, 8, 9, 11]
    )
    solve_inverse_problem_closed_form(
        case=case_2, mmq_mode=4, active_fibers=list(range(12))
    )

def plot():
    print("--- Plotting Graphics ---")
    plot_graphics(case="all_faces_with_equal_temp", is_second_approach=False)
    plot_graphics(case="temp_gradient_analisys", is_second_approach=True)

def run_full_pipeline():
    run_dynamics()
    add_perturbations()
    run_estimation()
    plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de simulação e inversão.")
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
