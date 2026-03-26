from modeling.main_dynamics import simulate_dynamics
from modeling.inverse_problem_closed_form import solve_inverse_problem_closed_form
from modeling.optical_push_pull import solve_inverse_problem_optical_push_pull
from modeling.add_thermal_noise import apply_thermokinematic_perturbation
from modeling.graphics_new_approach import plot_graphics
import argparse


def run_full_pipeline():
    # Bloco 1: Dinâmica
    print("integrate dynamics")
    simulate_dynamics(t_end=3, dt=5e-3)

    print("add pertubation on data")
    add_perturbations()
    # Bloco 2: Estimação (Inverse Problem)

def add_perturbations():
    apply_thermokinematic_perturbation(noise_std=10e-12)
    run_estimation()


def run_estimation():
    print("solve inverse problem")
    solve_inverse_problem_closed_form(mmq_mode=1)
    solve_inverse_problem_closed_form(mmq_mode=2)
    solve_inverse_problem_closed_form(mmq_mode=3)

    print("done")
    print("Solving optical push-pull")
    solve_inverse_problem_optical_push_pull(
        push_pull_pairs=[[0, 3], [4, 7], [8, 11]],
        name_save="modeling/data/inverse_output_optical_push_pull_cruzed.csv",
    )
    solve_inverse_problem_optical_push_pull(
        push_pull_pairs=[[0, 2], [4, 6], [8, 10]],
        name_save="modeling/data/inverse_output_optical_push_pull_aligned.csv",
    )
    plot()


def plot():
    plot_graphics()

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
    group.add_argument(
        "--graphics", action="store_true", help="Graphics only"
    )
    

    args = parser.parse_args()

    if args.all:
        run_full_pipeline()
    elif args.perturbation:
        add_perturbations()
    elif args.estimation:
        run_estimation()
    elif args.graphics:
        plot()
