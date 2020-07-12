import matplotlib.pyplot as plt
import numpy as np
import qs
import sim

from kernel import kernel_e_2d_gaussian
from numpy import ndarray
from lists import CellList2D, VerletList
from typing import List, Tuple

"""
Implementation of quorum sensing (QS)
Order of particle strengths: u_e, u_c
"""


def f(u_c: float):
    return qs.alpha + (qs.beta * u_c ** qs.n) / (qs.u_thresh ** qs.n + u_c ** qs.n) - qs.gamma_c * u_c


def reaction_with_bacterium(p: ndarray) -> Tuple[float, float]:
    u_e = p[2] / sim.volume_p
    u_c = p[3] / sim.volume_p
    flow_inside = qs.d1 * u_e - qs.d2 * u_c  # use intracellular decay rate
    du_e = -flow_inside - qs.gamma_c * u_e
    du_c = f(u_c) + flow_inside
    return du_e * sim.volume_p, du_c * sim.volume_p


def reaction_without_bacterium(p: ndarray) -> Tuple[float, float]:
    u_e = p[2] / sim.volume_p
    # u_c = p[3] / sim.volume_p
    du_e = -qs.gamma_e * u_e  # use extracellular decay rate
    du_c = 0
    return du_e * sim.volume_p, du_c * sim.volume_p


def apply_diffusion_reaction(_particles: ndarray, _verlet: VerletList) -> ndarray:
    updated_particles = np.zeros((sim.particle_number_per_dim ** 2, 4))

    for i in range(0, sim.particle_number_per_dim ** 2):
        p = _particles[i]
        summed_u_e_interaction = 0

        if i in qs.bacteria_particles:
            _reaction = reaction_with_bacterium
        else:
            _reaction = reaction_without_bacterium

        # Diffusive part (only u_e)
        for j in _verlet[i]:
            q = _particles[j]
            kernel_value = kernel_e_2d_gaussian(p, q)
            strength_u_e_difference = q[2] - p[2]
            summed_u_e_interaction += strength_u_e_difference * kernel_value

        d_strength_u_e_diff = sim.volume_p * sim.D / (sim.epsilon ** 2) * summed_u_e_interaction

        # Reactive part
        d_strength_u_e_reac, d_strength_u_c_reac = _reaction(p)

        # Total update
        updated_strength_u_e = p[2] + (d_strength_u_e_diff + d_strength_u_e_reac) * sim.dt
        updated_strength_u_c = p[3] + d_strength_u_c_reac * sim.dt
        updated_particles[i][:] = p[0], p[1], updated_strength_u_e, updated_strength_u_c

    return updated_particles


def xy_to_index(x: int, y: int) -> int:
    return int((x - sim.domain_lower_bound) / sim.h * sim.particle_number_per_dim + (y - sim.domain_lower_bound) / sim.h)


def initial_particles() -> Tuple[ndarray, VerletList]:
    _particles = np.zeros((sim.particle_number_per_dim ** 2, 4))

    for i in range(0, sim.particle_number_per_dim):
        for j in range(0, sim.particle_number_per_dim):
            x = sim.domain_lower_bound + i * sim.h
            y = sim.domain_lower_bound + j * sim.h
            strength_u_e = 0
            strength_u_c = 0

            _particles[i * sim.particle_number_per_dim + j][:] = x, y, strength_u_e, strength_u_c

    cells = CellList2D(_particles[:, 0:2])
    verlet = VerletList(_particles[:, 0:2], cells)
    return _particles, verlet


def load_bacteria(bacteria_file, _particles: ndarray):
    with open(bacteria_file, "r") as file:
        for line_i, line in enumerate(file):
            x, y = map(lambda s: float(s), line.split("   ")[1:3])
            best_j = xy_to_index(int(x), int(y))
            best_x = _particles[best_j][0]
            best_y = _particles[best_j][1]

            # First 7 bacteria have u_c = u_thresh, rest u_c = 0
            if line_i < 7:
                _particles[best_j][:] = best_x, best_y, 0, qs.u_thresh * sim.volume_p
            else:
                _particles[best_j][:] = best_x, best_y, 0, 0
            qs.bacteria_particles.append(best_j)


def print_summed_ahl(_particle_evolution: List[Tuple[float, ndarray]]):
    total_u_e = sum(_particle_evolution[-1][1][:, 2])
    total_u_c = sum(_particle_evolution[-1][1][:, 3])
    print("{} (∫u_e) + {} (∫u_c) ≈ {}"
          .format(round(total_u_e, 2), round(total_u_c, 2), round(total_u_e + total_u_c, 2)))


def plot_summed_ahl(_particle_evolution: List[Tuple[float, ndarray]]):
    t_coords = []
    u_e_coords = []
    u_c_coords = []
    for t in range(0, len(_particle_evolution)):
        total_u_e = sum(_particle_evolution[t][1][:, 2])
        total_u_c = sum(_particle_evolution[t][1][:, 3])
        t_coords.append(_particle_evolution[t][0])
        u_e_coords.append(total_u_e)
        u_c_coords.append(total_u_c)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t_coords, u_e_coords, label="total ∫u_e")
    ax.plot(t_coords, u_c_coords, label="total ∫u_c")
    plt.legend()
    return fig


def print_activated_bacteria(_particle_evolution: List[Tuple[float, ndarray]]):
    for i in qs.bacteria_particles:
        p = _particle_evolution[-1][1][i]
        print("{} bacterium at x={} y={}"
              .format("  ACTIVE" if p[3] / sim.volume_p >= qs.u_thresh else "inactive", p[0], p[1]))
