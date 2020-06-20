import numpy as np

from lists import *
from typing import List


def update_strength(p, neighbours, _particles, _kernel_e, env: Environment):
    summed_interaction = 0
    for j in neighbours:
        q = _particles[j]
        strength_difference = q.strength0 - p.strength0
        kernel_value = _kernel_e(p, q)
        summed_interaction += strength_difference * kernel_value

    delta_strength_p = env.volume_p * env.D / (env.epsilon ** 2) * summed_interaction
    return p.strength0 + delta_strength_p * env.dt


def pse_operator_1d(_particles: List[Particle1D1], _verlet: VerletList, _kernel_e, env: Environment):
    for _ in np.arange(0, env.t_max, env.dt):
        updated_particles = []
        for (i, p) in enumerate(_particles):
            updated_mass = update_strength(p, _verlet.cells[i], _particles, _kernel_e, env)
            updated_particles.append(Particle1D1(p.x, updated_mass))

        _particles = updated_particles

    return _particles


def inner_square_outer_boundary(env: Environment):
    inner_indices = []
    outer_index_pairs = dict()
    inner_square_left_index = int(env.cutoff / env.h)
    inner_square_after_right_index = env.particle_number_per_dim - inner_square_left_index
    inner_square_width = inner_square_after_right_index - inner_square_left_index

    def to_index(_i, _j):
        return _i * env.particle_number_per_dim + _j

    # Inner square
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            inner_indices.append(to_index(i, j))
    # Upper border
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j)
    # Lower border
    for i in range(inner_square_after_right_index, env.particle_number_per_dim):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j)
    # Left border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i, j + inner_square_width)
    # Right border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_after_right_index, env.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i, j - inner_square_width)
    # Corners
    for i in range(0, inner_square_left_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j + inner_square_width)
    for i in range(inner_square_after_right_index, env.particle_number_per_dim):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j + inner_square_width)
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_after_right_index, env.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j - inner_square_width)
    for i in range(inner_square_after_right_index, env.particle_number_per_dim):
        for j in range(inner_square_after_right_index, env.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j - inner_square_width)

    return inner_indices, outer_index_pairs


def pse_operator_2d(_particles: List[Particle2D1], _verlet: VerletList, _kernel_e, env: Environment):
    inner_square, outer_index_pairs = inner_square_outer_boundary(env)
    _particle_evolution = [_particles]

    for t in np.arange(0, env.t_max, env.dt):
        updated_particles: List[Particle2D1] = [None for _ in _particles]

        # Inner square interaction (normal PSE)
        for i in inner_square:
            p = _particles[i]
            updated_mass = update_strength(p, _verlet.cells[i], _particles, _kernel_e, env)
            updated_particles[i] = Particle2D1(p.x, p.y, updated_mass)

        # Outer boundary interaction (copying from inner square)
        for i in outer_index_pairs:
            p = _particles[i]
            copy_index = outer_index_pairs[i]
            updated_particles[i] = Particle2D1(p.x, p.y, updated_particles[copy_index].strength0)

        _particles = updated_particles
        if (env.t_max * 1 / 3) <= t < (env.t_max * 1 / 3) + env.dt or \
           (env.t_max * 2 / 3) <= t < (env.t_max * 2 / 3) + env.dt:
            _particle_evolution.append(_particles)

    _particle_evolution.append(_particles)
    return _particle_evolution


def pse_predict_u_1d(_particles: List[Particle1D1], start_x, end_x, env: Environment):
    _x_coords = []
    _concentration = []

    for p in filter(lambda _p: start_x <= _p.x <= end_x, _particles):
        _x_coords.append(p.x)
        _concentration.append(p.strength0 / env.volume_p)

    return _x_coords, _concentration


def pse_predict_u_2d(_particles: List[Particle2D], dimension: int, env: Environment):
    _x_coords = []
    _y_coords = []
    _concentration = []
    dimension_index = 2 + dimension

    for p in _particles:
        _x_coords.append(p.x)
        _y_coords.append(p.y)
        _concentration.append(p[dimension_index] / env.volume_p)

    return _x_coords, _y_coords, _concentration
