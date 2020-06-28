import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from evaluation import *
from pse import *
from random import uniform

D = 10
domain_lower_bound = 0
domain_upper_bound = 81
particle_number_per_dim = 51
h = (domain_upper_bound - domain_lower_bound) / (particle_number_per_dim - 1)
epsilon = h
volume_p = h ** 2
cutoff = 3 * epsilon
cell_side = cutoff
t_max = 20
dt = 0.01
env = Environment(D, domain_lower_bound, domain_upper_bound, particle_number_per_dim, h, epsilon, volume_p, cutoff,
                  cell_side, t_max, dt)


def apply_brusselator(_uv_strengths: Tuple[float, float]) -> Tuple[float, float]:
    _u = _uv_strengths[0] / volume_p
    _v = _uv_strengths[1] / volume_p
    du = a + k * _u ** 2 * _v - (b + 1) * _u
    dv = b * _u - k * _u ** 2 * _v
    return du * volume_p, dv * volume_p


def kernel_e(p: Particle2D2, q: Particle2D2) -> float:
    factor = 4 / (math.pi * epsilon ** 2)
    squared_norm = (q.x - p.x) ** 2 + (q.y - p.y) ** 2
    exponent = -squared_norm / epsilon ** 2
    return factor * math.exp(exponent)


def initial_particles() -> Tuple[List[Particle2D], List[Particle2D]]:
    _particles = []
    _particle_pos = []

    for i in range(0, particle_number_per_dim):
        for j in range(0, particle_number_per_dim):
            x = i * h
            y = j * h
            u = uniform(0, 1)
            v = uniform(0, 1) + 7

            _particles.append(Particle2D2(x, y, u * volume_p, v * volume_p))
            _particle_pos.append((x, y))

    return _particles, _particle_pos


if __name__ == "__main__":
    a, b, k = 2, 6, 1
    particles, particle_pos = initial_particles()
    cells = CellList2D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)

    particle_evolution = pse_operator_2d(particles, verlet, env, 4, kernel_e, apply_brusselator)

    #######################################
    # xy-u and xy-v 4x4 plot
    #######################################
    xy_concentration = []

    for strength_i in range(0, 2):
        for t in range(0, 4):
            x_coords, y_coords, concentration_i = pse_predict_u_2d(particle_evolution[t][1], strength_i, env)
            xy_concentration.append((x_coords, y_coords, concentration_i))

    fig = plot_nxm(xy_concentration, 4, 2,
                   zlabels=("u", "u", "u", "u", "v", "v", "v", "v"),
                   titles=("t=0", "t=1/3t_max", "t=2/3t_max", "t=t_max", "t=0", "t=1/3t_max", "t=2/3t_max", "t=t_max"))
    plt.show()
