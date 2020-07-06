import math
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from evaluation import plot_nxm
from kernel import kernel_e_2d_gaussian
from lists import CellList2D, VerletList
from typing import Tuple
from numpy import ndarray
from sim_impl import simulate_2d, pse_predict_u_2d


def delta(a: float, x: float):
    return 1 / (a * math.sqrt(math.pi)) * math.exp(-(x / a) ** 2)


def u0(x: float, y: float):
    a = 1 / 16
    x_ = x - 1 / 4
    y_ = y - 1 / 2

    return delta(a, x_) * delta(a, y_)


def initial_particles() -> Tuple[ndarray, VerletList]:
    _particles = np.zeros((sim.particle_number_per_dim ** 2, 3))

    for i in range(0, sim.particle_number_per_dim):
        for j in range(0, sim.particle_number_per_dim):
            x = i * sim.h
            y = j * sim.h
            mass = u0(x, y)

            _particles[i * sim.particle_number_per_dim + j][:] = x, y, mass

    _cells = CellList2D(_particles[:, 0:2], sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side)
    _verlet = VerletList(_particles[:, 0:2], _cells, sim.cutoff)
    return _particles, _verlet


def apply_diffusion(_particles: ndarray, _verlet: VerletList) -> ndarray:
    updated_particles = np.zeros((sim.particle_number_per_dim ** 2, 3))

    for i in range(0, sim.particle_number_per_dim ** 2):
        p = _particles[i]
        summed_mass_interaction = 0

        for j in _verlet[i]:
            q = _particles[j]

            kernel_value = kernel_e_2d_gaussian(p, q)
            mass_difference = q[2] - p[2]
            summed_mass_interaction += mass_difference * kernel_value

        d_mass = sim.volume_p * sim.D / (sim.epsilon ** 2) * summed_mass_interaction
        updated_mass = p[2] + d_mass * sim.dt
        updated_particles[i][:] = p[0], p[1], updated_mass

    return updated_particles


if __name__ == "__main__":
    sim.D = 2
    sim.domain_lower_bound = 0
    sim.domain_upper_bound = 1
    sim.particle_number_per_dim = 26
    sim.h = (sim.domain_upper_bound - sim.domain_lower_bound) / (sim.particle_number_per_dim - 1)
    sim.epsilon = sim.h
    sim.volume_p = sim.h ** 2
    sim.cutoff = 3 * sim.epsilon
    sim.cell_side = sim.cutoff
    sim.t_max = 0.3
    sim.dt = sim.h ** 2 / (3 * sim.D)

    particles, verlet = initial_particles()
    particle_evolution = simulate_2d(particles, verlet, 4, apply_diffusion)

    #######################################
    # 4-in-1 plot
    #######################################
    xy_concentration = []
    t_coords = []
    for t in range(0, 4):
        x_coords, y_coords, concentration = pse_predict_u_2d(particle_evolution[t][1], 0)
        xy_concentration.append((x_coords, y_coords, concentration))
        t_coords.append(round(particle_evolution[t][0], 2))

    fig = plot_nxm(xy_concentration, 2, 2,
                   zlabels=("u", "u", "u", "u"),
                   titles=("t={}".format(t_coords[0]), "t={}".format(t_coords[1]),
                           "t={}".format(t_coords[2]), "t={}".format(t_coords[3])))
    fig.show()
