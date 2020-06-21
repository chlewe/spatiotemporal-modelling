import numpy as np

from lists import *
from typing import List, Dict


def inner_square_outer_boundary(env: Environment)\
        -> Tuple[List[int], Dict[int, int]]:
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


def update_strengths(p, neighbours, _particles, env: Environment, _kernel_e, _reaction=None)\
        -> List[float]:
    num_strengths = len(p) - 2
    summed_interaction = [0 for _ in range(0, num_strengths)]

    for j in neighbours:
        q = _particles[j]
        kernel_value = _kernel_e(p, q)

        # Reactive part
        if _reaction:
            d_strengths_reac = _reaction(p[2:])
        else:
            d_strengths_reac = [0 for _ in p[2:]]

        # Diffusive part
        for strength_i in range(0, num_strengths):
            strength_i_difference = q[strength_i + 2] - p[strength_i + 2]
            summed_interaction[strength_i] += strength_i_difference * kernel_value

    updated_strengths = []
    for strength_i in range(0, num_strengths):
        d_strength_i_diff = env.volume_p * env.D / (env.epsilon ** 2) * summed_interaction[strength_i]
        d_strength_i = d_strength_i_diff + d_strengths_reac[strength_i]
        updated_strengths.append(p[strength_i + 2] + d_strength_i * env.dt)

    return updated_strengths


def pse_operator_1d(_particles: List[Particle1D1], _verlet: VerletList, _kernel_e, env: Environment)\
        -> List[Particle1D]:
    for _ in np.arange(0, env.t_max, env.dt):
        updated_particles = []
        for (i, p) in enumerate(_particles):
            updated_strengths = update_strengths(p, _verlet.cells[i], _particles, env, _kernel_e)
            updated_particles.append(create_particle_2d(p.x, p.y, updated_strengths))

        _particles = updated_particles

    return _particles


def pse_operator_2d(_particles: List[Particle2D], _verlet: VerletList, env: Environment, _kernel_e, _reaction=None)\
        -> List[List[Particle2D]]:
    inner_square, outer_index_pairs = inner_square_outer_boundary(env)
    _particle_evolution = [_particles]

    for t in np.arange(0, env.t_max, env.dt):
        updated_particles: List[Particle2D1] = [None for _ in _particles]

        # Inner square interaction (normal PSE)
        for i in inner_square:
            p = _particles[i]
            updated_strengths = update_strengths(p, _verlet.cells[i], _particles, env, _kernel_e, _reaction)
            updated_particles[i] = create_particle_2d(p.x, p.y, updated_strengths)

        # Outer boundary interaction (copying from inner square)
        for i in outer_index_pairs:
            p = _particles[i]
            copy_index = outer_index_pairs[i]
            updated_particles[i] = create_particle_2d(p.x, p.y, updated_particles[copy_index][2:])

        _particles = updated_particles
        if (env.t_max * 1 / 3) <= t < (env.t_max * 1 / 3) + env.dt or \
           (env.t_max * 2 / 3) <= t < (env.t_max * 2 / 3) + env.dt:
            _particle_evolution.append(_particles)

    _particle_evolution.append(_particles)
    return _particle_evolution


def pse_predict_u_1d(_particles: List[Particle1D1], start_x, end_x, env: Environment)\
        -> Tuple[List[float], List[float]]:
    _x_coords = []
    _concentration = []

    for p in filter(lambda _p: start_x <= _p.x <= end_x, _particles):
        _x_coords.append(p.x)
        _concentration.append(p.strength0 / env.volume_p)

    return _x_coords, _concentration


def pse_predict_u_2d(_particles: List[Particle2D], strength_i: int, env: Environment)\
        -> Tuple[List[float], List[float], List[float]]:
    _x_coords = []
    _y_coords = []
    _concentration = []
    strength_index = strength_i + 2

    for p in _particles:
        _x_coords.append(p.x)
        _y_coords.append(p.y)
        _concentration.append(p[strength_index] / env.volume_p)

    return _x_coords, _y_coords, _concentration
