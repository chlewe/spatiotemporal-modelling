import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

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

    particle_evolution = pse_operator_2d(particles, verlet, kernel_e, env)

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
    fig = plt.figure()
    x_evo, y_evo, u_evo = [], [], []
    for step in particle_evolution:
        x_coords, y_coords, concentration = pse_predict_u_2d(step, 0, env)
        x_evo.append(x_coords)
        y_evo.append(y_coords)
        u_evo.append(concentration)

    # First subplot
    ax0 = fig.add_subplot(2, 2, 1, projection='3d')
    surf0 = ax0.plot_trisurf(x_evo[0], y_evo[0], u_evo[0], cmap="jet", linewidth=0.1)
    fig.colorbar(surf0, shrink=0.5, aspect=5)

    # Second subplot
    ax1 = fig.add_subplot(2, 2, 2, projection='3d', sharex=ax0, sharey=ax0)
    surf1 = ax1.plot_trisurf(x_evo[1], y_evo[1], u_evo[1], cmap="jet", linewidth=0.1)
    fig.colorbar(surf1, shrink=0.5, aspect=5)

    # Third subplot
    ax2 = fig.add_subplot(2, 2, 3, projection='3d', sharex=ax0, sharey=ax0)
    surf2 = ax2.plot_trisurf(x_evo[2], y_evo[2], u_evo[2], cmap="jet", linewidth=0.1)
    fig.colorbar(surf2, shrink=0.5, aspect=5)

    # Fourth subplot
    ax3 = fig.add_subplot(2, 2, 4, projection='3d', sharex=ax0, sharey=ax0)
    surf3 = ax3.plot_trisurf(x_evo[3], y_evo[3], u_evo[3], cmap="jet", linewidth=0.1)
    fig.colorbar(surf3, shrink=0.5, aspect=5)

    plt.show()
