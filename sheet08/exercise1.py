import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from evaluation import *
from pse import *
from enum import Enum


class Main(Enum):
    TEST_F_QS = 0
    FIRST_BENCHMARK = 1
    SECOND_BENCHMARK = 2
    THIRD_BENCHMARK = 3


#######################################
# Select your main method
main = Main.SECOND_BENCHMARK
#######################################

D = 1
domain_lower_bound = 0
domain_upper_bound = 50
particle_number_per_dim = 51
h = (domain_upper_bound - domain_lower_bound) / (particle_number_per_dim - 1)
epsilon = h
volume_p = h ** 2
cutoff = 3 * epsilon
cell_side = cutoff
t_max = 20
dt = h ** 2 / (4 * D)
env = Environment(D, domain_lower_bound, domain_upper_bound, particle_number_per_dim, h, epsilon, volume_p, cutoff,
                  cell_side, t_max, dt)

d1 = 0.25      # diffusion rate of AHL from outside to inside the cell
d2 = 2.5       # diffusion rate of AHL from inside to outside the cell
alpha = 1      # low production rate of AHL
beta = 100     # increase of production rate of AHL
u_thresh = 2   # threshold AHL concentration between low and increased activity
n = 10         # polymerisation degree
gamma_e = 0.5  # decay rate of extracellular AHL
gamma_c = 0.5  # decay rate of intracellular AHL


def f(u_c):
    return alpha + (beta * u_c ** n) / (u_thresh ** n + u_c ** n) - gamma_c * u_c


def test_f_qs():
    x_coords = np.arange(0, 5, 0.1)
    y_coords = [f(u_c) for u_c in x_coords]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_coords, y_coords)
    fig.show()


# Strength order: u_e, u_c
def reaction_with_bacterium(_ue_uc_strengths: Tuple[float, float]) -> Tuple[float, float]:
    u_e = _ue_uc_strengths[0] / volume_p
    u_c = _ue_uc_strengths[1] / volume_p
    flow_inside = d1 * u_e - d2 * u_c  # use intracellular decay rate
    du_e = -flow_inside - gamma_c * u_e
    du_c = f(u_c) + flow_inside
    return du_e, du_c


def reaction_without_bacterium(_ue_uc_strengths: Tuple[float, float]) -> Tuple[float, float]:
    u_e = _ue_uc_strengths[0] / volume_p
    # u_c = _ue_uc_strengths[1] / volume_p
    du_e = -gamma_e * u_e  # use extracellular decay rate
    du_c = 0
    return du_e, du_c


def update_particle_qs(_i: int, _neighbours, _particles, _env: Environment, _kernel_e, _reaction=None)\
        -> Tuple[float, float, List[float]]:
    p = _particles[_i]
    summed_u_e_interaction = 0

    if _i in bacteria_particles:
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

    d_strength_u_e_diff = _env.volume_p * _env.D / (_env.epsilon ** 2) * summed_u_e_interaction
    updated_strength_u_e = p.strength0 + (d_strength_u_e_diff + d_strengths_reac[0]) * _env.dt
    updated_strength_u_c = p.strength1 + d_strengths_reac[1] * _env.dt

    return p.x, p.y, [updated_strength_u_e, updated_strength_u_c]


def kernel_e(p: Particle2D2, q: Particle2D2) -> float:
    factor = 4 / (math.pi * epsilon ** 2)
    squared_norm = (q.x - p.x) ** 2 + (q.y - p.y) ** 2
    exponent = -squared_norm / epsilon ** 2
    return factor * math.exp(exponent)


def initial_particles() -> Tuple[List[Particle2D2], List[Particle2D2]]:
    _particles = []
    _particle_pos = []

    for i in range(0, particle_number_per_dim):
        for j in range(0, particle_number_per_dim):
            x = i * h
            y = j * h
            u_e = 0
            u_c = 0

            _particles.append(Particle2D2(x, y, u_e * volume_p, u_c * volume_p))
            _particle_pos.append((x, y))

    return _particles, _particle_pos


def first_benchmark():
    global bacteria_particles
    particles, particle_pos = initial_particles()
    i = math.floor(len(particles) / 2)
    middle_p = particles[i]
    particles[i] = Particle2D2(middle_p.x, middle_p.y, 0, u_thresh * volume_p)
    bacteria_particles = [i]

    cells = CellList2D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)
    particle_evolution = pse_operator_2d(particles, verlet, env, 2, kernel_e, _update_particle=update_particle_qs)

    total_u_e = sum(map(lambda q: q.strength0, particle_evolution[-1][1]))
    total_u_c = sum(map(lambda q: q.strength1, particle_evolution[-1][1]))
    print("{} (u_e) + {} (u_c) ≈ {}".format(round(total_u_e, 2), round(total_u_c, 2), round(total_u_e + total_u_c, 2)))


def second_benchmark():
    global bacteria_particles
    particles, particle_pos = initial_particles()
    i = math.floor(len(particles) / 2)
    middle_p = particles[i]
    particles[i] = Particle2D2(middle_p.x, middle_p.y, 0, u_thresh * volume_p)
    bacteria_particles = [i]
    j = next(_j for _j in range(0, len(particles)) if
             particles[_j].x == middle_p.x + 4 and particles[_j].y == middle_p.y)
    bacteria_particles.append(j)

    cells = CellList2D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)
    particle_evolution = pse_operator_2d(particles, verlet, env, 100, kernel_e, _update_particle=update_particle_qs)

    t_coords = []
    u_e_coords = []
    u_c_coords = []
    for t in range(0, 100):
        total_u_e = sum(map(lambda q: q.strength0, particle_evolution[t][1]))
        total_u_c = sum(map(lambda q: q.strength1, particle_evolution[t][1]))
        t_coords.append(particle_evolution[t][0])
        u_e_coords.append(total_u_e)
        u_c_coords.append(total_u_c)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t_coords, u_e_coords, label="total u_e")
    ax.plot(t_coords, u_c_coords, label="total u_c")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if main == Main.TEST_F_QS:
        test_f_qs()
    elif main == Main.FIRST_BENCHMARK:
        first_benchmark()
    elif main == Main.SECOND_BENCHMARK:
        gamma_e = 0.05
        gamma_c = 0.05
        t_max = 50
        env = Environment(D, domain_lower_bound, domain_upper_bound, particle_number_per_dim, h, epsilon, volume_p,
                          cutoff, cell_side, t_max, dt)
        second_benchmark()
    elif main == Main.THIRD_BENCHMARK:
        pass
