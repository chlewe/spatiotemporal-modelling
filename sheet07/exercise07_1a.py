import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from typing import Tuple


def apply_brusselator(_u: float, _v: float, _a: int, _b: int, _k: int) -> Tuple[float, float]:
    du = _a + _k * _u ** 2 * _v - (_b + 1) * _u
    dv = _b * _u - _k * _u ** 2 * _v
    return du, dv


if __name__ == "__main__":
    sim.dt = 0.01
    sim.t_max = 20

    a, b, k = 2, 6, 1
    u_values, v_values, t_values = [], [], []

    # oscillating setting
    particles_u, particles_v = [0.7], [0.04]
    # fixed-point setting
    # particles_u, particles_v = [a], [b / a]

    for t in np.arange(0, sim.t_max, sim.dt):
        u_values.append(particles_u[0])
        v_values.append(particles_v[0])
        t_values.append(t)

        du, dv = apply_brusselator(particles_u[0], particles_v[0], a, b, k)
        particles_u = [particles_u[0] + du * sim.dt]
        particles_v = [particles_v[0] + dv * sim.dt]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t_values, u_values, label="u")
    ax.plot(t_values, v_values, label="v")
    ax.set_xlabel("t")
    ax.set_ylabel("concentration")
    fig.legend()
    fig.show()
