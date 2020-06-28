import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from evaluation import *
from pse import *

D = 2
domain_lower_bound = 0
domain_upper_bound = 1
particle_number_per_dim = 26
h = (domain_upper_bound - domain_lower_bound) / (particle_number_per_dim - 1)
epsilon = h
volume_p = h ** 2
cutoff = 3 * epsilon
cell_side = cutoff
t_max = 0.3
dt = h ** 2 / (3 * D)
env = Environment(D, domain_lower_bound, domain_upper_bound, particle_number_per_dim, h, epsilon, volume_p, cutoff,
                  cell_side, t_max, dt)


def delta(a: float, x: float):
    return 1 / (a * math.sqrt(math.pi)) * math.exp(-(x / a) ** 2)


def u0(x: float, y: float):
    a = 1 / 16
    x_ = x - 1 / 4
    y_ = y - 1 / 2

    return delta(a, x_) * delta(a, y_)


def kernel_e(p: Particle2D, q: Particle2D):
    factor = 4 / (math.pi * epsilon ** 2)
    squared_norm = (q.x - p.x) ** 2 + (q.y - p.y) ** 2
    exponent = -squared_norm / epsilon ** 2
    return factor * math.exp(exponent)


def initial_particles():
    _particles = []
    _particle_pos = []

    for i in range(0, particle_number_per_dim):
        for j in range(0, particle_number_per_dim):
            x = i * h
            y = j * h
            mass = volume_p * u0(x, y)
            _particles.append(Particle2D1(x, y, mass))
            _particle_pos.append((x, y))

    return _particles, _particle_pos


if __name__ == "__main__":
    particles, particle_pos = initial_particles()
    cells = CellList2D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)

    particle_evolution = pse_operator_2d(particles, verlet, env, 4, kernel_e)

    #######################################
    # Single plot
    #######################################
    # fig = plt.figure()
    # x_coords, y_coords, concentration = predict_u_pse(particle_evolution[-1])

    # ax = plt.axes(projection='3d')
    # surf = ax.plot_trisurf(x_coords, y_coords, concentration, cmap="jet", linewidth=0.1)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    #######################################
    # 4-in-1 plot
    #######################################
    xy_concentration = []
    t_coords = []
    for t in range(0, 4):
        x_coords, y_coords, concentration = pse_predict_u_2d(particle_evolution[t][1], 0, env)
        xy_concentration.append((x_coords, y_coords, concentration))
        t_coords.append(round(particle_evolution[t][0], 2))

    fig = plot_nxm(xy_concentration, 2, 2, zlabels=("u", "u", "u", "u"),
                   titles=("t={}".format(t_coords[0]), "t={}".format(t_coords[1]), "t={}".format(t_coords[2]),
                           "t={}".format(t_coords[3])))
    plt.show()
