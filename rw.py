import numpy as np

from lists import *
from typing import List


def rw_operator_1d(_particles: List[Particle1D1], env: Environment):
    d_mean = 0
    d_variance = (2 * env.D * env.dt)

    for _ in np.arange(0, env.t_max, env.dt):
        updated_particles = []
        for p in _particles:
            dx = np.random.normal(loc=d_mean, scale=d_variance)
            updated_x = p.x + dx
            updated_particles.append(Particle1D1(updated_x, p.strength0))

        _particles = updated_particles

    return _particles


def rw_predict_u_1d(_particles: List[Particle1D1], start_x: float, end_x: float):
    bins = 16
    bin_width = (end_x - start_x) / bins
    bin_concentration = []
    bin_edges = [start_x + i * bin_width for i in range(0, bins + 1)]

    for i in range(0, bins):
        bin_particles = list(filter(lambda p: bin_edges[i] < p[0] <= bin_edges[i + 1], _particles))
        if not bin_particles:
            bin_concentration.append(0)
        else:
            bin_concentration.append(sum(map(lambda p: p[1], bin_particles)) / bin_width)

    bin_centroids = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(0, bins)]
    return bin_centroids, bin_concentration
