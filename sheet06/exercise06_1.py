import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from enum import Enum
from kernel import kernel_e_1d_gaussian
from lists import CellList1D, VerletList
from numpy import ndarray
from typing import Tuple, List
from sim_impl import simulate_1d, rw_predict_u_1d, pse_predict_u_1d


class Main(Enum):
    CONCENTRATION_PLOT = 0
    CONVERGENCE_PLOT = 1


#######################################
# Select your main method:
main = Main.CONCENTRATION_PLOT
#######################################

#######################################
# Select your number of particles:
# (only takes effect in concentration plot)
# normal_particle_number = 50
# normal_particle_number = 100
# normal_particle_number = 200
# normal_particle_number = 400
normal_particle_number = 800
#######################################


def u0(x: float) -> float:
    return 0 if x < 0 else x * math.exp(-x ** 2)


def u_exact(x: float, t: float) -> float:
    return x / ((1 + 4 * sim.D * t) ** (3 / 2)) * math.exp(-x ** 2 / 1 + 4 * sim.D * t)


# create normal particles (0,4] and mirrored particles [-4,0)
def initial_particles() -> Tuple[ndarray, VerletList]:
    _particles = np.zeros((sim.particle_number_per_dim, 2))

    for i in range(0, sim.particle_number_per_dim):
        x = (i - normal_particle_number + 0.5) * sim.h
        mass = u0(x) * sim.volume_p
        _particles[i][:] = x, mass

    _cells = CellList1D(_particles[:, 0:1])
    _verlet = VerletList(_particles[:, 0:1], _cells)
    return _particles, _verlet


def apply_rw(_particles: ndarray, _verlet: VerletList = None) -> ndarray:
    d_mean = 0
    d_variance = (2 * sim.D * sim.dt)
    updated_particles = np.zeros((sim.particle_number_per_dim, 2))

    for i in range(0, sim.particle_number_per_dim):
        x, mass = _particles[i][:]
        dx = np.random.normal(loc=d_mean, scale=d_variance)
        updated_particles[i][:] = x + dx, mass

    return updated_particles


def apply_pse(_particles: ndarray, _verlet: VerletList) -> ndarray:
    updated_particles = np.zeros((sim.particle_number_per_dim, 2))

    for i in range(0, sim.particle_number_per_dim):
        p = _particles[i]
        summed_mass_interaction = 0

        for j in _verlet[i]:
            q = _particles[j]

            kernel_value = kernel_e_1d_gaussian(p, q)
            mass_difference = q[1] - p[1]
            summed_mass_interaction += mass_difference * kernel_value

        d_mass = sim.volume_p * sim.D / (sim.epsilon ** 2) * summed_mass_interaction
        updated_mass = p[1] + d_mass * sim.dt
        updated_particles[i][:] = p[0], updated_mass

    return updated_particles


def evaluate(x_coords: List[float], u_coords: List[float], t) -> Tuple[float, float]:
    error = []

    for i in range(0, len(x_coords)):
        exact_u = u_exact(x_coords[i], t)
        error.append(abs(u_coords[i] - exact_u))

    l2_norm = math.sqrt(sum(map(lambda e: e ** 2, error)) / len(x_coords))
    linf_norm = max(error)
    return l2_norm, linf_norm


def concentration_plot():
    particles, verlet = initial_particles()
    rw_particles = simulate_1d(particles, None, apply_rw)
    pse_particles = simulate_1d(particles, verlet, apply_pse)

    rw_x_coords, rw_u = rw_predict_u_1d(rw_particles, 0, sim.domain_upper_bound)
    pse_x_coords, pse_u = pse_predict_u_1d(pse_particles, 0, sim.domain_upper_bound)
    fine_x = np.arange(0, 4, 0.01)
    exact_u = [u_exact(x, sim.t_max) for x in fine_x]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fine_x, exact_u, color="red", label="Exact concentration")
    ax.scatter(pse_x_coords, pse_u, marker="o", label="PSE: Predicted concentration")
    ax.scatter(rw_x_coords, rw_u, marker="+", s=200, label="RW: Predicted concentration")

    fig.legend()
    plt.show()


def convergence_plot():
    n_coords = []
    rw_l2_coords = []
    rw_linf_coords = []
    pse_l2_coords = []
    pse_linf_coords = []

    for n in [50, 100, 200, 400, 800]:
        global normal_particle_number
        normal_particle_number = n
        sim.particle_number_per_dim = normal_particle_number * 2
        sim.h = (sim.domain_upper_bound - sim.domain_lower_bound) / (sim.particle_number_per_dim - 1)

        particles, verlet = initial_particles()
        rw_particles = simulate_1d(particles, None, apply_rw)
        pse_particles = simulate_1d(particles, verlet, apply_pse)

        rw_x_coords, rw_u = rw_predict_u_1d(rw_particles, 0, sim.domain_upper_bound)
        pse_x_coords, pse_u = pse_predict_u_1d(pse_particles, 0, sim.domain_upper_bound)

        rw_l2_norm, rw_linf_norm = evaluate(rw_x_coords, rw_u, sim.t_max)
        pse_l2_norm, pse_linf_norm = evaluate(pse_x_coords, pse_u, sim.t_max)

        n_coords.append(n)
        rw_l2_coords.append(rw_l2_norm)
        rw_linf_coords.append(rw_linf_norm)
        pse_l2_coords.append(pse_l2_norm)
        pse_linf_coords.append(pse_linf_norm)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(n_coords, rw_l2_coords, label="RW: L2 norm")
    ax.plot(n_coords, rw_linf_coords, label="RW: Linf norm")
    ax.plot(n_coords, pse_l2_coords, label="PSE: L2 norm")
    ax.plot(n_coords, pse_linf_coords, label="PSE: Linf norm")

    fig.legend()
    plt.show()


if __name__ == "__main__":
    sim.D = 0.0001
    sim.domain_lower_bound = -4
    sim.domain_upper_bound = 4
    sim.particle_number_per_dim = normal_particle_number * 2  # normal particles + mirrored particles
    sim.h = (sim.domain_upper_bound - sim.domain_lower_bound) / (sim.particle_number_per_dim - 1)
    sim.epsilon = sim.h
    sim.volume_p = sim.h ** 1
    sim.cutoff = 3 * sim.epsilon
    sim.cell_side = sim.cutoff
    sim.t_max = 10
    sim.dt = 0.1

    if main == Main.CONCENTRATION_PLOT:
        concentration_plot()
    elif main == Main.CONVERGENCE_PLOT:
        convergence_plot()
    else:
        sys.exit(1)
