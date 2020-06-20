import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from matplotlib import cm
from utils import *
from typing import List

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


def delta(a: float, x: float):
    return 1 / (a * math.sqrt(math.pi)) * math.exp(-(x / a) ** 2)


def u0(x: float, y: float):
    a = 1 / 16
    x_ = x - 1 / 4
    y_ = y - 1 / 2

    return delta(a, x_) * delta(a, y_)


def kernel_e(x: float, y: float):
    factor = 4 / (math.pi * epsilon ** 2)
    squared_norm = x ** 2 + y ** 2
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
            _particles.append(Particle2D(x, y, mass))
            _particle_pos.append((x, y))

    return _particles, _particle_pos


def inner_square_outer_boundary():
    inner_indices = []
    outer_index_pairs = dict()
    inner_square_left_index = int(cutoff / h)
    inner_square_after_right_index = particle_number_per_dim - inner_square_left_index
    inner_square_width = inner_square_after_right_index - inner_square_left_index

    def to_index(_i, _j):
        return _i * particle_number_per_dim + _j

    # Inner square
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            inner_indices.append(to_index(i, j))
    # Upper border
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j)
    # Lower border
    for i in range(inner_square_after_right_index, particle_number_per_dim):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j)
    # Left border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i, j + inner_square_width)
    # Right border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_after_right_index, particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i, j - inner_square_width)
    # Corners
    for i in range(0, inner_square_left_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j + inner_square_width)
    for i in range(inner_square_after_right_index, particle_number_per_dim):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j + inner_square_width)
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_after_right_index, particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j - inner_square_width)
    for i in range(inner_square_after_right_index, particle_number_per_dim):
        for j in range(inner_square_after_right_index, particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j - inner_square_width)

    return inner_indices, outer_index_pairs


def update_strength_2d(p: Particle2D, neighbours: List[int], _particles: List[Particle2D]):
    summed_interaction = 0
    for j in neighbours:
        q = _particles[j]
        strength_difference = q.strength - p.strength
        kernel_value = kernel_e(q.x - p.x, q.y - p.y)
        summed_interaction += strength_difference * kernel_value

    delta_strength_p = volume_p * D / (epsilon ** 2) * summed_interaction
    return p.strength + delta_strength_p * dt


def pse_operator_2d(_particles: List[Particle2D], _verlet: VerletList):
    inner_square, outer_index_pairs = inner_square_outer_boundary()
    _particle_evolution = [_particles]

    for t in np.arange(0, t_max, dt):
        updated_particles: List[Particle2D] = [None for p in _particles]

        # Inner square interaction (normal PSE)
        for i in inner_square:
            p = _particles[i]
            updated_mass = update_strength_2d(p, _verlet.cells[i], _particles)
            updated_particles[i] = Particle2D(p.x, p.y, updated_mass)

        # Outer boundary interaction (copying from inner square)
        for i in outer_index_pairs:
            p = _particles[i]
            copy_index = outer_index_pairs[i]
            updated_particles[i] = Particle2D(p.x, p.y, updated_particles[copy_index].strength)

        _particles = updated_particles
        if (t_max * 1 / 3) <= t < (t_max * 1 / 3) + dt or \
           (t_max * 2 / 3) <= t < (t_max * 2 / 3) + dt:
            _particle_evolution.append(_particles)

    _particle_evolution.append(_particles)
    return _particle_evolution


def pse_predict_u_2d(_particles: List[Particle2D]):
    _x_coords = []
    _y_coords = []
    _concentration = []

    for p in _particles:
        _x_coords.append(p.x)
        _y_coords.append(p.y)
        _concentration.append(p.strength / volume_p)

    return _x_coords, _y_coords, _concentration


if __name__ == "__main__":
    particles, particle_pos = initial_particles()
    cells = CellList2D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)

    particle_evolution = pse_operator_2d(particles, verlet)
    fig = plt.figure()

    #######################################
    # Single plot
    #######################################
    # x_coords, y_coords, concentration = predict_u_pse(particle_evolution[-1])

    # ax = plt.axes(projection='3d')
    # surf = ax.plot_trisurf(x_coords, y_coords, concentration, cmap="jet", linewidth=0.1)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    #######################################
    # 4-in-1 plot
    #######################################
    x_evo, y_evo, u_evo = [], [], []
    for step in particle_evolution:
        x_coords, y_coords, concentration = pse_predict_u_2d(step)
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
