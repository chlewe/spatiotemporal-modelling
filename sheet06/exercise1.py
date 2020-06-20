import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from utils import *

D = 0.0001
domain_lower_bound = -4
domain_upper_bound = 4
normal_particle_number = 50  # plus the same number of mirrored particles
h = (domain_upper_bound - domain_lower_bound) / (normal_particle_number * 2 - 1)
epsilon = h
volume_p = h
cutoff = 3 * epsilon
cell_side = cutoff
t_max = 10
dt = 0.1


def u0(x):
    return 0 if x < 0 else x * math.exp(-x ** 2)


def u_exact(x, t):
    return x / ((1 + 4 * D * t) ** (3/2)) * math.exp(-x**2 / 1 + 4 * D * t)


# Gaussian kernel
def kernel_e(p: Particle1D, q: Particle1D):
    dist_x = q.x - p.x
    return 1 / (2 * epsilon * math.sqrt(math.pi)) * math.exp(-dist_x ** 2 / (4 * epsilon ** 2))


# Alternative, polynomial kernel
def kernel_e_poly(p: Particle1D, q: Particle1D):
    dist_x = q.x - p.x
    return 1 / h * 15 / math.pi ** 2 / (abs(dist_x / h) ** 10 + 1)


# create normal particles (0,4] and mirrored particles [-4,0)
def initial_particles():
    _particles = []
    _particle_pos = []

    for i in range(0, normal_particle_number * 2):
        x = (i - normal_particle_number + 0.5) * h
        mass = volume_p * u0(x)
        _particles.append(Particle1D(x, mass))
        _particle_pos.append((x, ))

    return _particles, _particle_pos


#######################################
# Random walk (RW)
#######################################
def rw_operator_1d(_particles: List[Particle1D]):
    d_mean = 0
    d_variance = (2 * D * dt)

    for _ in np.arange(0, t_max, dt):
        updated_particles = []
        for p in _particles:
            dx = np.random.normal(loc=d_mean, scale=d_variance)
            updated_x = p.x + dx
            updated_particles.append(Particle1D(updated_x, p.strength))

        _particles = updated_particles

    return _particles


def rw_predict_u_1d(_particles: List[Particle1D], start_x: float, end_x: float):
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


#######################################
# Particle Strength Exchange (PSE)
#######################################
def update_strength(p, neighbours, _particles, _kernel_e):
    summed_interaction = 0
    for j in neighbours:
        q = _particles[j]
        strength_difference = q.strength - p.strength
        kernel_value = _kernel_e(p, q)
        summed_interaction += strength_difference * kernel_value

    delta_strength_p = volume_p * D / (epsilon ** 2) * summed_interaction
    return p.strength + delta_strength_p * dt


def pse_operator_1d(_particles: List[Particle1D], _verlet: VerletList):
    for _ in np.arange(0, t_max, dt):
        updated_particles = []
        for (i, p) in enumerate(_particles):
            updated_mass = update_strength(p, _verlet.cells[i], _particles, kernel_e)
            updated_particles.append(Particle1D(p.x, updated_mass))

        _particles = updated_particles

    return particles


def pse_predict_u_1d(_particles: List[Particle1D], start_x, end_x):
    _x_coords = []
    _concentration = []

    for p in filter(lambda _p: start_x <= _p.x <= end_x, _particles):
        _x_coords.append(p.x)
        _concentration.append(p.strength / volume_p)

    return _x_coords, _concentration


# Returns predicted concentrations, exact concentrations, l2 norm and linf norm
# def evaluate(particles):
#    error = []
#    positions = []
#    prediction = []
#    exact = []
#
#    for p in particles:
#        x = p[0]
#        u_x = predict_u(particles, x)
#        u_x_exact = u_exact(x, integration_time)
#
#        positions.append(x)
#        prediction.append(u_x)
#        exact.append(u_x_exact)
#        error.append(abs(u_x - u_x_exact))
#        #print("x: {:5.2f} | RW: {:7.4f} | Exact: {:7.4f} | Error: {:7.4f}".format(x, u_x_RW, u_x_exact, error))
#
#    l2_norm = math.sqrt(sum(map(lambda e: e**2, error)) / len(particles))
#    linf_norm = max(error)
#    return positions, prediction, exact, l2_norm, linf_norm


if __name__ == "__main__":
    particles, particle_pos = initial_particles()
    cells = CellList1D(particle_pos, domain_lower_bound, domain_upper_bound, cell_side)
    verlet = VerletList(particle_pos, cells, cutoff)

    rw_particles = rw_operator_1d(particles)
    pse_particles = pse_operator_1d(particles, verlet)
    rw_x, rw_u = rw_predict_u_1d(rw_particles, 0, domain_upper_bound)
    pse_x, pse_u = pse_predict_u_1d(pse_particles, 0, domain_upper_bound)
    fine_x = np.arange(0, 4, 0.01)
    exact_u = [u_exact(x, t_max) for x in fine_x]

    #######################################
    # Concentration plot
    #######################################
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(rw_x, rw_u, label="RW: Predicted concentration")
    ax.scatter(pse_x, pse_u, label="PSE: Predicted concentration")
    ax.plot(fine_x, exact_u, label="Exact concentration")
    plt.legend()
    plt.show()

    #######################################
    # Convergence plot
    #######################################
    # N = []
    # l2_RW = []
    # l2_PSE = []
    # linf_RW = []
    # linf_PSE = []

    # for particle_number in [50, 100, 200, 400, 800]:
    #    N.append(particle_number)
    #
    #    particles = random_walk(particle_number)
    #    _, _, _, l2_norm, linf_norm = evaluate(particles)
    #    l2_RW.append(l2_norm)
    #    linf_RW.append(linf_norm)
    #
    #    #particles = pse(particle_number)
    #    #_, _, _, l2_norm, linf_norm = evaluate(particles)
    #    #l2_PSE.append(l2_norm)
    #    #linf_PSE.append(linf_norm)
    # ax.plot(N, l2_RW, label="RW: L2 norm")
    # ax.plot(N, linf_RW, label="RW: Linf norm")
    # ax.plot(N, l2_PSE, label="PSE: L2 norm")
    # ax.plot(N, linf_PSE, label="PSE: Linf norm")
