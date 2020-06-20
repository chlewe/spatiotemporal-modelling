import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from lists import *

t_max = 20
dt = 0.01


def apply_brusselator(_particles_u: List[int], _particles_v: List[int], _a: int, _b: int, _k: int):
    _particles_du = []
    _particles_dv = []

    for i in range(0, len(_particles_u)):
        u = _particles_u[i]
        v = _particles_v[i]

        du = _a + _k * u ** 2 * v - (_b + 1) * u
        dv = _b * u - _k * u ** 2 * v
        _particles_du.append(du * dt)
        _particles_dv.append(dv * dt)

    return _particles_du, _particles_dv


if __name__ == "__main__":
    a, b, k = 2, 6, 1
    u_values, v_values, t_values = [], [], []

    # oscillating setting
    particles_u, particles_v = [0.7], [0.04]
    # fixed-point setting
    # particles_u, particles_v = [a], [b / a]

    for t in np.arange(0, t_max, dt):
        u_values.append(particles_u[0])
        v_values.append(particles_v[0])
        t_values.append(t)

        particles_du, particles_dv = apply_brusselator(particles_u, particles_v, a, b, k)
        particles_u = [particles_u[0] + particles_du[0]]
        particles_v = [particles_v[0] + particles_dv[0]]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t_values, u_values, label="u")
    ax.plot(t_values, v_values, label="v")
    ax.set_xlabel("t")
    ax.set_ylabel("concentration")
    plt.legend()
    plt.show()
