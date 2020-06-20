import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from utils import CellList, VerletList

domain_bound = 4
diffusion_coefficient = 0.0001
time_step = 0.1
integration_time = 10

def u0(x):
    return 0 if x < 0 else x * math.exp(-x**2)

def u_exact(x, t):
    return x / ((1 + 4 * diffusion_coefficient * t) ** (3/2)) * math.exp(-x**2 / 1 + 4 * diffusion_coefficient * t)

def initial_particles(particle_number):
    interparticle_spacing = 2 * domain_bound / (2 * particle_number - 1)
    volume_p = interparticle_spacing
    print(interparticle_spacing)

    # create normal particles (0,4] and mirrored particles [-4,0)
    particles = []
    for i in range(0, particle_number * 2):
        x_p = (i - particle_number + 0.5) * interparticle_spacing
        value_p = volume_p * u0(x_p)
        particles.append((x_p, value_p))

    return particles


#
# Random walk
#
def random_walk(particle_number):
    particles = initial_particles(particle_number)

    d_mean = 0
    d_variance = (2 * diffusion_coefficient * time_step)

    for _ in np.arange(0, integration_time, time_step):
        updated_particles = []
        for p in particles:
            dx = np.random.normal(loc=d_mean, scale=d_variance)
            updated_x_p = p[0] + dx
            constant_value_p = p[1]
            updated_particles.append((updated_x_p, constant_value_p))

        particles = updated_particles

    return particles

#
# Particle Strength Exchange
#
def pse(particle_number):
    interparticle_spacing = 2 * domain_bound / (2 * particle_number - 1)
    kernel_size = interparticle_spacing
    volume_p = interparticle_spacing

    particles = initial_particles(particle_number)
    particle_pos = list(map(lambda x: (x,), np.array(particles)[:,0]))
    cells = CellList(particle_pos, -domain_bound, domain_bound, kernel_size)
    verlet = VerletList(particle_pos, cells, kernel_size)

    def kernel_e(x):
        return 1 / (2 * kernel_size * math.sqrt(math.pi)) * math.exp(-x**2 / (4 * kernel_size**2))

    #def kernel_e(x):
    #    return 1 / kernel_size * 15 / math.pi**2 / (abs(x / kernel_size)**10 + 1)

    for _ in np.arange(0, integration_time, time_step):
        updated_particles = []
        for (i, p) in enumerate(particles):
        #for p in particles:
            summed_exchange = 0
            for j in verlet.cells[i]:
            #for q in particles:
                q = particles[j]
                summed_exchange += (q[1] - p[1]) * kernel_e(q[0] - p[0])

            constant_x_p = p[0]
            delta_value_p = volume_p * diffusion_coefficient / (kernel_size**2) * summed_exchange
            updated_particles.append((constant_x_p, p[1] + delta_value_p))

        particles = updated_particles

    return particles


#
# Evaluation
#
def predict_u_rw(particles):
    bins = 16
    bin_width = domain_bound / bins
    bin_u = []
    bin_edges = [i * bin_width for i in range(0, bins + 1)]
    current_bin = 0
    current_bin_values = []

    for i in range(0, bins):
        bin_particles = list(filter(lambda p: bin_edges[i] < p[0] and p[0] <= bin_edges[i + 1], particles))
        if not bin_particles:
            print(i)
            bin_u.append(0)
        else:
            bin_u.append(sum(map(lambda p: p[1], bin_particles)) / bin_width)

    bin_centroids = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(0, bins)]

    return bin_centroids, bin_u

def predict_u_pse(particles):
    particle_number = len(particles) // 2
    interparticle_spacing = 2 * domain_bound / (2 * particle_number - 1)
    volume_p = interparticle_spacing
    positions = []
    u = []

    for p in filter(lambda p: 0 <= p[0], particles):
        positions.append(p[0])
        u.append(p[1] / volume_p)

    return positions, u

## Returns predicted concentrations, exact concentrations, l2 norm and linf norm
#def evaluate(particles):
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


N = []
l2_RW = []
l2_PSE = []
linf_RW = []
linf_PSE = []
fig, ax = plt.subplots(figsize=(10,10))

#
# Concentration plot
#
particles = random_walk(400)
positions, prediction = predict_u_rw(particles)
ax.scatter(positions, prediction, label="RW: Predicted concentration")

particles = pse(400)
positions, prediction = predict_u_pse(particles)
ax.scatter(positions, prediction, label="PSE: Predicted concentration")

fine_positions = np.arange(0, 4, 0.01)
exact = [u_exact(x, integration_time) for x in fine_positions]
ax.plot(fine_positions, exact, label="Exact concentration")

##
## Convergence plot
##
#for particle_number in [50, 100, 200, 400, 800]:
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
#ax.plot(N, l2_RW, label="RW: L2 norm")
#ax.plot(N, linf_RW, label="RW: Linf norm")
##ax.plot(N, l2_PSE, label="PSE: L2 norm")
##ax.plot(N, linf_PSE, label="PSE: Linf norm")

plt.legend()
plt.show()
