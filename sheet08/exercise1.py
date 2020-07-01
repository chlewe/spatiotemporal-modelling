import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import qs

from enum import Enum
from evaluation import *
from functools import partial
from kernel import *
from pse import *
from qs_impl import f, initial_particles, plot_summed_ahl, print_activated_bacteria, update_particle_qs

kernel_e = partial(kernel_e_2d_gaussian, epsilon=qs.epsilon)


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


def first_benchmark():
    particles, particle_pos = initial_particles()
    i = math.floor(len(particles) / 2)
    middle_p = particles[i]
    particles[i] = Particle2D2(middle_p.x, middle_p.y, 0, qs.u_thresh * qs.volume_p)
    qs.bacteria_particles = [i]

    cells = CellList2D(particle_pos, qs.domain_lower_bound, qs.domain_upper_bound, qs.cell_side)
    verlet = VerletList(particle_pos, cells, qs.cutoff)
    particle_evolution = pse_operator_2d(particles, verlet, qs.env, 2, kernel_e, _update_particle=update_particle_qs)

    total_u_e = sum(map(lambda q: q.strength0, particle_evolution[-1][1]))
    total_u_c = sum(map(lambda q: q.strength1, particle_evolution[-1][1]))
    print("{} (∫u_e) + {} (∫u_c) ≈ {}"
          .format(round(total_u_e, 2), round(total_u_c, 2), round(total_u_e + total_u_c, 2)))


def second_benchmark():
    particles, particle_pos = initial_particles()
    i = math.floor(len(particles) / 2)
    middle_p = particles[i]
    particles[i] = Particle2D2(middle_p.x, middle_p.y, 0, qs.u_thresh * qs.volume_p)
    qs.bacteria_particles = [i]

    j = next(_j for _j in range(0, len(particles)) if
             particles[_j].x == middle_p.x + 4 and particles[_j].y == middle_p.y)
    particles[j] = Particle2D2(middle_p.x + 4, middle_p.y, 0, 0)
    qs.bacteria_particles.append(j)

    cells = CellList2D(particle_pos, qs.domain_lower_bound, qs.domain_upper_bound, qs.cell_side)
    verlet = VerletList(particle_pos, cells, qs.cutoff)
    particle_evolution = pse_operator_2d(particles, verlet, qs.env, 100, kernel_e, _update_particle=update_particle_qs)

    print_activated_bacteria(particle_evolution)

    fig = plot_summed_ahl(particle_evolution)
    fig.show()


def third_benchmark():
    particles, particle_pos = initial_particles()

    with open("bacterialPos.dat", "r") as file:
        for line_i, line in enumerate(file):
            x, y = map(lambda s: float(s), line.split("   ")[1:3])
            best_j = next(_j for _j in range(0, len(particles))
                          if particles[_j].x <= x < particles[_j].x + qs.h
                          and particles[_j].y <= y < particles[_j].y + qs.h)

            # First 7 bacteria have u_c = u_thresh, rest u_c = 0
            if line_i < 7:
                particles[best_j] = Particle2D2(x, y, 0, qs.u_thresh * qs.volume_p)
            else:
                particles[best_j] = Particle2D2(x, y, 0, 0)
            qs.bacteria_particles.append(best_j)

    cells = CellList2D(particle_pos, qs.domain_lower_bound, qs.domain_upper_bound, qs.cell_side)
    verlet = VerletList(particle_pos, cells, qs.cutoff)
    particle_evolution = pse_operator_2d(particles, verlet, qs.env, 2, kernel_e, _update_particle=update_particle_qs)

    print_activated_bacteria(particle_evolution)

    x_coords, y_coords, u_e_coords = pse_predict_u_2d(particle_evolution[-1][1], 0, qs.env)
    fig = plot_colormap(u_e_coords, qs.particle_number_per_dim, qs.particle_number_per_dim,
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
        qs.env = Environment(qs.D, qs.domain_lower_bound, qs.domain_upper_bound, qs.particle_number_per_dim, qs.h,
                             qs.epsilon, qs.volume_p, qs.cutoff, qs.cell_side, qs.t_max, qs.dt)
        second_benchmark()
    elif main == Main.THIRD_BENCHMARK:
        qs.gamma_e = 0.01
        qs.gamma_c = 0.01
        qs.t_max = 200
        qs.env = Environment(qs.D, qs.domain_lower_bound, qs.domain_upper_bound, qs.particle_number_per_dim, qs.h,
                             qs.epsilon, qs.volume_p, qs.cutoff, qs.cell_side, qs.t_max, qs.dt)
        third_benchmark()
