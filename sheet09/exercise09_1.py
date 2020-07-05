import math
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import qs

from evaluation import plot_colormap
from lists import VerletList
from pse import inner_square_outer_boundary
from qs_impl import initial_particles, print_activated_bacteria, apply_diffusion_reaction, print_summed_ahl,\
    load_bacteria, qs_predict_u_2d, apply_periodic_boundary_conditions
from numpy import ndarray


def apply_advection(_particles: ndarray):
    for i in range(0, qs.particle_number_per_dim ** 2):
        x = _particles[i][0]
        y = _particles[i][1]

        updated_x = x + qs.dt * 4 * (1 - math.cos(2 * math.pi * y / 50))
        updated_y = y + qs.dt * 3
        _particles[i][0] = updated_x
        _particles[i][1] = updated_y


def a(s: float) -> float:
    _s = abs(s)
    if 0 <= _s < 1:
        return 1 - 0.5 * (5 * _s ** 2 - 3 * _s ** 3)
    elif 1 <= _s < 2:
        return 0.5 * (2 - _s) ** 2 * (1 - _s)
    else:
        return 0


def remesh_range(x: float):
    _x = math.floor(x)
    return filter(lambda y: 0 <= y < qs.particle_number_per_dim, [_x - 2, _x - 1, _x, _x + 1, _x + 2, _x + 3])


def apply_remeshing(unmoved_particles: ndarray, moved_particles: ndarray) -> ndarray:
    remeshed_particles = np.zeros(moved_particles.shape)
    remeshed_particles[:, 0] = unmoved_particles[:, 0]  # copy x
    remeshed_particles[:, 1] = unmoved_particles[:, 1]  # copy y
    remeshed_particles[:, 3] = moved_particles[:, 3]  # copy u_c (bacteria stay fixed)

    for i in range(0, qs.particle_number_per_dim ** 2):
        x = moved_particles[i][0]
        y = moved_particles[i][1]
        strength_u_e = moved_particles[i][2]

        for grid_x in remesh_range(x):
            for grid_y in remesh_range(y):
                s_x = (x - grid_x)  # / qs.h = 1
                s_y = (y - grid_y)  # / qs.h = 1
                j = grid_x * qs.particle_number_per_dim + grid_y
                remeshed_particles[j][2] += a(s_x) * a(s_y) * strength_u_e

    return remeshed_particles


def simulate(_particles: ndarray, _verlet: VerletList, n_evolutions: int):
    inner_square, outer_index_pairs = inner_square_outer_boundary(qs.particle_number_per_dim, qs.cutoff, qs.h)
    _particle_evolution = [(0, _particles)]
    dt_evolution = qs.t_max if n_evolutions < 1 else qs.t_max / (n_evolutions - 1)

    for t in np.arange(qs.dt, qs.t_max + qs.dt, qs.dt):
        print(t)

        updated_particles = apply_diffusion_reaction(_particles, _verlet)
        apply_periodic_boundary_conditions(updated_particles, outer_index_pairs)

        apply_advection(updated_particles)
        updated_particles = apply_remeshing(_particle_evolution[0][1], updated_particles)
        apply_periodic_boundary_conditions(updated_particles, outer_index_pairs)

        _particles = updated_particles
        if t % dt_evolution < qs.dt:
            _particle_evolution.append((t, _particles))

    # There are cases where the last evolution step is missed
    if qs.t_max % dt_evolution != 0:
        _particle_evolution.append((qs.t_max, _particles))
    return _particle_evolution


if __name__ == "__main__":
    qs.gamma_e = 0.1
    qs.gamma_c = 0.1
    qs.t_max = 10

    particles, verlet = initial_particles()
    load_bacteria("bacterialPos.dat", particles)

    particle_evolution = simulate(particles, verlet, 2)
    print_summed_ahl(particle_evolution)
    print_activated_bacteria(particle_evolution)

    x_coords, y_coords, u_e_coords = qs_predict_u_2d(particle_evolution[-1][1], 0)
    fig = plot_colormap(u_e_coords, qs.particle_number_per_dim, qs.particle_number_per_dim,
                        title="$u_e$ concentration field at t={}".format(particle_evolution[-1][0]))
    fig.show()
