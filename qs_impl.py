import math
import matplotlib.pyplot as plt
import numpy as np
import qs

from typing import List, Tuple
from lists import *

"""
Implementation of quorum sensing (QS)
Order of particle strengths: u_e, u_c
"""


def f(u_c):
    return qs.alpha + (qs.beta * u_c ** qs.n) / (qs.u_thresh ** qs.n + u_c ** qs.n) - qs.gamma_c * u_c


def reaction_with_bacterium(_ue_uc_strengths: Tuple[float, float]) -> Tuple[float, float]:
    u_e = _ue_uc_strengths[0] / qs.volume_p
    u_c = _ue_uc_strengths[1] / qs.volume_p
    flow_inside = qs.d1 * u_e - qs.d2 * u_c  # use intracellular decay rate
    du_e = -flow_inside - qs.gamma_c * u_e
    du_c = f(u_c) + flow_inside
    return du_e * qs.volume_p, du_c * qs.volume_p


def reaction_without_bacterium(_ue_uc_strengths: Tuple[float, float]) -> Tuple[float, float]:
    u_e = _ue_uc_strengths[0] / qs.volume_p
    # u_c = _ue_uc_strengths[1] / volume_p
    du_e = -qs.gamma_e * u_e  # use extracellular decay rate
    du_c = 0
    return du_e * qs.volume_p, du_c * qs.volume_p


def update_particle_qs(_i: int, _neighbours, _particles, _env: Environment, _kernel_e, _reaction=None)\
        -> Tuple[float, float, List[float]]:
    p = _particles[_i]
    summed_u_e_interaction = 0

    if _i in qs.bacteria_particles:
        _reaction = reaction_with_bacterium
    else:
        _reaction = reaction_without_bacterium

    for j in _neighbours:
        q = _particles[j]
        kernel_value = _kernel_e(p, q)

        # Reactive part
        d_strengths_reac = _reaction(p[2:])

        # Diffusive part (only u_e)
        strength_u_e_difference = q.strength0 - p.strength0
        summed_u_e_interaction += strength_u_e_difference * kernel_value

    d_strength_u_e_diff = qs.volume_p * qs.D / (qs.epsilon ** 2) * summed_u_e_interaction
    updated_strength_u_e = p.strength0 + (d_strength_u_e_diff + d_strengths_reac[0]) * qs.dt
    updated_strength_u_c = p.strength1 + d_strengths_reac[1] * qs.dt

    return p.x, p.y, [updated_strength_u_e, updated_strength_u_c]


def initial_particles() -> Tuple[List[Particle2D2], List[Particle2D2]]:
    _particles = []
    _particle_pos = []

    for i in range(0, qs.particle_number_per_dim):
        for j in range(0, qs.particle_number_per_dim):
            x = i * qs.h
            y = j * qs.h
            u_e = 0
            u_c = 0

            _particles.append(Particle2D2(x, y, u_e * qs.volume_p, u_c * qs.volume_p))
            _particle_pos.append((x, y))

    return _particles, _particle_pos


def plot_summed_ahl(_particle_evolution):
    t_coords = []
    u_e_coords = []
    u_c_coords = []
    for t in range(0, len(_particle_evolution)):
        total_u_e = sum(map(lambda q: q.strength0, _particle_evolution[t][1]))
        total_u_c = sum(map(lambda q: q.strength1, _particle_evolution[t][1]))
        t_coords.append(_particle_evolution[t][0])
        u_e_coords.append(total_u_e)
        u_c_coords.append(total_u_c)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t_coords, u_e_coords, label="total ∫u_e")
    ax.plot(t_coords, u_c_coords, label="total ∫u_c")
    plt.legend()
    return fig


def print_activated_bacteria(_particle_evolution):
    for i in qs.bacteria_particles:
        p = _particle_evolution[-1][1][i]
        print("{} bacterium at x={} y={}"
              .format("  ACTIVE" if p.strength1 / qs.volume_p >= qs.u_thresh else "inactive", p.x, p.y))
