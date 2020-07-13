import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import qs
import sim

from enum import Enum
from evaluation import plot_colormap
from lists import VerletList
from qs_impl import f, initial_particles, plot_summed_ahl, print_activated_bacteria, apply_diffusion_reaction,\
    print_summed_ahl, load_bacteria
from sim_impl import apply_periodic_boundary_conditions, inner_square_outer_boundary_2d, pse_predict_u_2d


class Main(Enum):
    TEST_F_QS = 0
    FIRST_BENCHMARK = 1
    SECOND_BENCHMARK = 2
    THIRD_BENCHMARK = 3


#######################################
# Select your main method:
main = Main.THIRD_BENCHMARK
#######################################


def test_f_qs():
    x_coords = np.arange(0, 5, 0.1)
    y_coords = [f(u_c) for u_c in x_coords]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_coords, y_coords)
    fig.show()


def simulate(_particles, _verlet: VerletList, n_evolutions: int):
    inner_square, outer_index_pairs = inner_square_outer_boundary_2d()
    _particle_evolution = [(0, _particles)]
    dt_evolution = sim.t_max if n_evolutions < 1 else sim.t_max / (n_evolutions - 1)

    for t in np.arange(sim.dt, sim.t_max + sim.dt, sim.dt):
        print("{:6.2f}%".format(t / (sim.t_max + sim.dt) * 100))

        updated_particles = apply_diffusion_reaction(_particles, _verlet)
        apply_periodic_boundary_conditions(updated_particles, outer_index_pairs)

        _particles = updated_particles
        if t % dt_evolution < sim.dt:
            _particle_evolution.append((t, _particles))

    # There are cases where the last evolution step is missed
    if sim.t_max % dt_evolution != 0:
        _particle_evolution.append((sim.t_max, _particles))
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

    for j in range(0, sim.particle_number_per_dim ** 2):
        if particles[j][0] == particles[i][0] + 4 and particles[j][1] == particles[i][1]:
            qs.bacteria_particles.append(j)
            break

    particle_evolution = simulate(particles, verlet, 100)
    print_activated_bacteria(particle_evolution)

    fig = plot_summed_ahl(particle_evolution)
    plt.show()


def third_benchmark():
    particles, verlet = initial_particles()
    load_bacteria("bacterialPos.dat", particles)

    particle_evolution = simulate(particles, verlet, 2)
    print_activated_bacteria(particle_evolution)

    x_coords, y_coords, u_e_coords = pse_predict_u_2d(particle_evolution[-1][1], 0)
    fig = plot_colormap(u_e_coords, sim.particle_number_per_dim, sim.particle_number_per_dim,
                        xlabel="x",
                        ylabel="y",
                        title="$u_e$ concentration field at t={}".format(particle_evolution[-1][0]))
    plt.show()


if __name__ == "__main__":
    sim.D = 1
    sim.domain_lower_bound = 0
    sim.domain_upper_bound = 50
    sim.particle_number_per_dim = 51
    sim.h = (sim.domain_upper_bound - sim.domain_lower_bound) / (sim.particle_number_per_dim - 1)
    sim.epsilon = sim.h
    sim.volume_p = sim.h ** 2
    sim.cutoff = 3 * sim.epsilon
    sim.cell_side = sim.cutoff
    sim.t_max = 20
    sim.dt = sim.h ** 2 / (4 * sim.D)

    if main == Main.TEST_F_QS:
        test_f_qs()
    elif main == Main.FIRST_BENCHMARK:
        first_benchmark()
    elif main == Main.SECOND_BENCHMARK:
        qs.gamma_e = 0.05
        qs.gamma_c = 0.05
        sim.t_max = 50
        second_benchmark()
    elif main == Main.THIRD_BENCHMARK:
        qs.gamma_e = 0.01
        qs.gamma_c = 0.01
        sim.t_max = 200
        third_benchmark()
    else:
        sys.exit(1)
