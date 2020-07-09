import math
import numpy as np
import sim

from lists import VerletList
from typing import Tuple, List, Dict
from numpy import ndarray


def inner_square_outer_boundary_2d() -> Tuple[List[int], Dict[int, int]]:
    inner_indices = []
    outer_index_pairs = dict()
    inner_square_left_index = int(sim.cutoff / sim.h)
    inner_square_after_right_index = sim.particle_number_per_dim - inner_square_left_index
    inner_square_width = inner_square_after_right_index - inner_square_left_index

    def to_index(_i, _j):
        return _i * sim.particle_number_per_dim + _j

    # Inner square
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            inner_indices.append(to_index(i, j))
    # Upper border
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j)
    # Lower border
    for i in range(inner_square_after_right_index, sim.particle_number_per_dim):
        for j in range(inner_square_left_index, inner_square_after_right_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j)
    # Left border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i, j + inner_square_width)
    # Right border
    for i in range(inner_square_left_index, inner_square_after_right_index):
        for j in range(inner_square_after_right_index, sim.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i, j - inner_square_width)
    # Corners
    for i in range(0, inner_square_left_index):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j + inner_square_width)
    for i in range(inner_square_after_right_index, sim.particle_number_per_dim):
        for j in range(0, inner_square_left_index):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j + inner_square_width)
    for i in range(0, inner_square_left_index):
        for j in range(inner_square_after_right_index, sim.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i + inner_square_width, j - inner_square_width)
    for i in range(inner_square_after_right_index, sim.particle_number_per_dim):
        for j in range(inner_square_after_right_index, sim.particle_number_per_dim):
            outer_index_pairs[to_index(i, j)] = to_index(i - inner_square_width, j - inner_square_width)

    return inner_indices, outer_index_pairs


def apply_periodic_boundary_conditions(_particles: ndarray, outer_index_pairs):
    for i in outer_index_pairs:
        copy_index = outer_index_pairs[i]
        _particles[i][2:] = _particles[copy_index][2:]


def simulate_1d(_particles: ndarray, _verlet: VerletList, _apply_diffusion_reaction)\
        -> ndarray:
    _particle_evolution = [(0, _particles)]

    for t in np.arange(sim.dt, sim.t_max + sim.dt, sim.dt):
        print("{:6.2f}%".format(t / (sim.t_max + sim.dt) * 100))
        updated_particles = _apply_diffusion_reaction(_particles, _verlet)
        _particles = updated_particles

    return updated_particles


def simulate_2d(_particles: ndarray, _verlet: VerletList, n_evolutions: int, _apply_diffusion_reaction)\
        -> List[Tuple[float, ndarray]]:
    inner_square, outer_index_pairs = inner_square_outer_boundary_2d()
    _particle_evolution = [(0, _particles)]
    dt_evolution = sim.t_max if n_evolutions < 1 else sim.t_max / (n_evolutions - 1)

    for t in np.arange(sim.dt, sim.t_max + sim.dt, sim.dt):
        print("{:6.2f}%".format(t / (sim.t_max + sim.dt) * 100))

        updated_particles = _apply_diffusion_reaction(_particles, _verlet)
        apply_periodic_boundary_conditions(updated_particles, outer_index_pairs)

        _particles = updated_particles
        if t % dt_evolution < sim.dt:
            _particle_evolution.append((t, _particles))

    # There are cases where the last evolution step is missed
    if sim.t_max % dt_evolution != 0:
        _particle_evolution.append((sim.t_max, _particles))
    return _particle_evolution


def rw_predict_u_1d(_particles: ndarray, start_x: float, end_x: float) -> Tuple[List[float], List[float]]:
    bins = 16
    bin_width = (end_x - start_x) / bins
    bin_concentrations = []
    bin_edges = [start_x + i * bin_width for i in range(0, bins + 1)]

    for i in range(0, bins):
        bin_masses = [_particles[j][1] for j in range(0, sim.particle_number_per_dim)
                      if bin_edges[i] < _particles[j][0] <= bin_edges[i + 1]]
        if not bin_masses:
            bin_concentrations.append(0)
        else:
            bin_concentrations.append(sum(bin_masses) / bin_width)

    bin_centroids = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(0, bins)]
    return bin_centroids, bin_concentrations


def pse_predict_u_1d(_particles: ndarray, start_x: float, end_x: float) -> Tuple[List[float], List[float]]:
    _x_coords = []
    _concentrations = []

    for i in range(0, sim.particle_number_per_dim):
        if start_x <= _particles[i][0] <= end_x:
            _x_coords.append(_particles[i][0])
            _concentrations.append(_particles[i][1] / sim.volume_p)

    return _x_coords, _concentrations


def pse_predict_u_2d(_particles: ndarray, strength_i: int) -> Tuple[ndarray, ndarray, ndarray]:
    _x_coords = []
    _y_coords = []
    _concentrations = []

    _x_coords = _particles[:, 0]
    _y_coords = _particles[:, 1]
    _concentrations = _particles[:, 2 + strength_i] / sim.volume_p

    return _x_coords, _y_coords, _concentrations
