import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import qs

from enum import Enum
from evaluation import plot_colormap
from pse import inner_square_outer_boundary
from lists import VerletList
from qs_impl import f, initial_particles, plot_summed_ahl, print_activated_bacteria, apply_diffusion_reaction,\
    print_summed_ahl, load_bacteria, apply_periodic_boundary_conditions, qs_predict_u_2d


class Main(Enum):
    TEST_F_QS = 0
    FIRST_BENCHMARK = 1
    SECOND_BENCHMARK = 2
    THIRD_BENCHMARK = 3


#######################################
# Select your main method
main = Main.THIRD_BENCHMARK
#######################################


def test_f_qs():
    x_coords = np.arange(0, 5, 0.1)
    y_coords = [f(u_c) for u_c in x_coords]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_coords, y_coords)
    fig.show()


def simulate(_particles, _verlet: VerletList, n_evolutions: int):
    inner_square, outer_index_pairs = inner_square_outer_boundary(qs.particle_number_per_dim, qs.cutoff, qs.h)
    _particle_evolution = [(0, _particles)]
    dt_evolution = qs.t_max if n_evolutions < 1 else qs.t_max / (n_evolutions - 1)

    for t in np.arange(qs.dt, qs.t_max + qs.dt, qs.dt):
        print(t)

        updated_particles = apply_diffusion_reaction(_particles, _verlet)
        apply_periodic_boundary_conditions(updated_particles, outer_index_pairs)

        _particles = updated_particles
        if t % dt_evolution < qs.dt:
            _particle_evolution.append((t, _particles))

    # There are cases where the last evolution step is missed
    if qs.t_max % dt_evolution != 0:
        _particle_evolution.append((qs.t_max, _particles))
    return _particle_evolution


def first_benchmark():
    particles, verlet = initial_particles()
    i = math.floor(len(particles) / 2)
    particles[i][2:] = 0, qs.u_thresh
    qs.bacteria_particles = [i]

    particle_evolution = simulate(particles, verlet, 2)
    print_summed_ahl(particle_evolution)


def second_benchmark():
    particles, verlet = initial_particles()
    i = math.floor(len(particles) / 2)
    particles[i][2:] = 0, qs.u_thresh
    qs.bacteria_particles = [i]

    for j in range(0, qs.particle_number_per_dim ** 2):
        if particles[j][0] == particles[i][0] + 4 and particles[j][1] == particles[i][1]:
            qs.bacteria_particles.append(j)
            break

    particle_evolution = simulate(particles, verlet, 100)
    print_activated_bacteria(particle_evolution)

    fig = plot_summed_ahl(particle_evolution)
    fig.show()


def third_benchmark():
    particles, verlet = initial_particles()
    load_bacteria("bacterialPos.dat", particles)

    particle_evolution = simulate(particles, verlet, 2)
    print_activated_bacteria(particle_evolution)

    x_coords, y_coords, u_e_coords = qs_predict_u_2d(particle_evolution[-1][1], 0)
    fig = plot_colormap(u_e_coords, qs.particle_number_per_dim, qs.particle_number_per_dim,
                        xlabel="x",
                        ylabel="y",
                        title="$u_e$ concentration field at t={}".format(particle_evolution[-1][0]))
    fig.show()


if __name__ == "__main__":
    if main == Main.TEST_F_QS:
        test_f_qs()
    elif main == Main.FIRST_BENCHMARK:
        first_benchmark()
    elif main == Main.SECOND_BENCHMARK:
        qs.gamma_e = 0.05
        qs.gamma_c = 0.05
        qs.t_max = 50
        second_benchmark()
    elif main == Main.THIRD_BENCHMARK:
        qs.gamma_e = 0.01
        qs.gamma_c = 0.01
        qs.t_max = 200
        third_benchmark()
