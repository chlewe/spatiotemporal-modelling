import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from pse import *
from rw import *

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
env = Environment(D, domain_lower_bound, domain_upper_bound, normal_particle_number * 2, h, epsilon, volume_p, cutoff,
                  cell_side, t_max, dt)


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
        _particles.append(Particle1D1(x, mass))
        _particle_pos.append((x, ))

    return _particles, _particle_pos


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

    rw_particles = rw_operator_1d(particles, env)
    pse_particles = pse_operator_1d(particles, verlet, kernel_e, env)
    rw_x, rw_u = rw_predict_u_1d(rw_particles, 0, domain_upper_bound)
    pse_x, pse_u = pse_predict_u_1d(pse_particles, 0, domain_upper_bound, env)
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
