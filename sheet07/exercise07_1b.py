import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from evaluation import plot_nxm
from kernel import kernel_e_2d_gaussian
from lists import CellList2D, VerletList
from random import uniform
from typing import Tuple
from sim_impl import simulate_2d, pse_predict_u_2d
from numpy import ndarray


def initial_particles() -> Tuple[ndarray, VerletList]:
    _particles = np.zeros((sim.particle_number_per_dim ** 2, 4))

    for i in range(0, sim.particle_number_per_dim):
        for j in range(0, sim.particle_number_per_dim):
            x = i * sim.h
            y = j * sim.h
            u = uniform(0, 1)
            v = uniform(0, 1) + 7

            _particles[i * sim.particle_number_per_dim + j][:] = x, y, u * sim.volume_p, v * sim.volume_p

    _cells = CellList2D(_particles[:, 0:2])
    _verlet = VerletList(_particles[:, 0:2], _cells)
    return _particles, _verlet


def apply_brusselator(_uv_strengths: ndarray) -> Tuple[float, float]:
    _u = _uv_strengths[0] / sim.volume_p
    _v = _uv_strengths[1] / sim.volume_p
    du = a + k * _u ** 2 * _v - (b + 1) * _u
    dv = b * _u - k * _u ** 2 * _v
    return du * sim.volume_p, dv * sim.volume_p


def apply_diffusion_reaction(_particles: ndarray, _verlet: VerletList) -> ndarray:
    updated_particles = np.zeros((sim.particle_number_per_dim ** 2, 4))

    for i in range(0, sim.particle_number_per_dim ** 2):
        p = _particles[i]
        summed_strength_interaction = [0, 0]

        # Diffusive part
        for j in _verlet[i]:
            q = _particles[j]
            kernel_value = kernel_e_2d_gaussian(p, q)
            strength_difference = q[2:] - p[2:]
            summed_strength_interaction += strength_difference * kernel_value

        d_strength_diff = sim.volume_p * sim.D / (sim.epsilon ** 2) * summed_strength_interaction

        # Reactive part
        d_strength_reac = apply_brusselator(p[2:])

        # Total update
        updated_strength = p[2:] + (d_strength_diff + d_strength_reac) * sim.dt
        updated_particles[i][:] = p[0], p[1], updated_strength[0], updated_strength[1]

    return updated_particles


if __name__ == "__main__":
    sim.D = 10
    sim.domain_lower_bound = 0
    sim.domain_upper_bound = 81
    sim.particle_number_per_dim = 51
    sim.h = (sim.domain_upper_bound - sim.domain_lower_bound) / (sim.particle_number_per_dim - 1)
    sim.epsilon = sim.h
    sim.volume_p = sim.h ** 2
    sim.cutoff = 3 * sim.epsilon
    sim.cell_side = sim.cutoff
    sim.t_max = 10
    sim.dt = 0.01

    a, b, k = 2, 6, 1
    particles, verlet = initial_particles()
    particle_evolution = simulate_2d(particles, verlet, 4, apply_diffusion_reaction)

    #######################################
    # xy-u and xy-v 4x4 plot
    #######################################
    xyu_coords = []

    for strength_i in range(0, 2):
        for t in range(0, 4):
            x_coords, y_coords, concentration_i = pse_predict_u_2d(particle_evolution[t][1], strength_i)
            xyu_coords.append((x_coords, y_coords, concentration_i))

    fig = plot_nxm(xyu_coords, 4, 2,
                   zlabels=("u", "u", "u", "u", "v", "v", "v", "v"),
                   titles=("t=0", "t=1/3t_max", "t=2/3t_max", "t=t_max",
                           "t=0", "t=1/3t_max", "t=2/3t_max", "t=t_max"))
    fig.show()
