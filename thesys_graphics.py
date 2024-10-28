import bragg.optical_simulation
from bragg.pure_reflection_and_pure_transmission_analysis import (
    interrogation_laser_one_fbg,
    interrogation_two_fbgs_reflection_of_reflection,
)
from interrogation_analysis.push_pull_interrogation_analysis import (
    power_vs_delta_lambda_animation_one_fiber,
    power_vs_delta_lambda_animation_two_fibers,
    
)
from interrogation_analysis.interrogation_analysis import interrogation_two_fbgs_reflection_of_reflection


if __name__ == "__main__":
    # grafico 1
    """Cap. Introducao"""

    # bragg.optical_simulation.plot_drawFig6_spectres()
    """Cap. interrogacao"""

    bragg.optical_simulation.plot_bragg_spectrum()
    interrogation_laser_one_fbg()
    power_vs_delta_lambda_animation_two_fibers()
    power_vs_delta_lambda_animation_one_fiber()
    interrogation_two_fbgs_reflection_of_reflection()
    pass
