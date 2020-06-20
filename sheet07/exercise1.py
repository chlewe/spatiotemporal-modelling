import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from utils import *

end_time = 20
time_step = 0.01


def apply_brusselator(particles_u, particles_v, a, b, k):
    particles_du = []
    particles_dv = []

    for i in range(0, len(particles_u)):
        u = particles_u[i]
        v = particles_v[i]

        du = a + k * u**2 * v - (b + 1) * u
        dv = b * u - k * u**2 * v
        particles_du.append(du * time_step)
        particles_dv.append(dv * time_step)

    return particles_du, particles_dv


if __name__ == "__main__":
    a, b, k = 2, 6, 1
    U, V, T = [], [], []

    # oscillating setting
    particles_u, particles_v = [0.7], [0.04]
    # fixed-point setting
    # particles_u, particles_v = [a], [b / a]

    for t in np.arange(0, end_time, time_step):
        U.append(particles_u[0])
        V.append(particles_v[0])
        T.append(t)

        particles_du, particles_dv = apply_brusselator(particles_u, particles_v, a, b, k)
        particles_u = [particles_u[0] + particles_du[0]]
        particles_v = [particles_v[0] + particles_dv[0]]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(T, U, label="u")
    ax.plot(T, V, label="v")
    ax.set_xlabel("t")
    ax.set_ylabel("concentration")
    plt.legend()
    plt.show()
