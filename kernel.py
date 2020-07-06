import math
import sim

from numpy import ndarray


def kernel_e_1d_gaussian(p: ndarray, q: ndarray) -> float:
    dist_x = q[0] - p[0]
    return 1 / (2 * sim.epsilon * math.sqrt(math.pi)) * math.exp(-dist_x ** 2 / (4 * sim.epsilon ** 2))


def kernel_e_1d_poly(p: ndarray, q: ndarray) -> float:
    dist_x = q[0] - p[0]
    return 1 / sim.h * 15 / math.pi ** 2 / (abs(dist_x / sim.h) ** 10 + 1)


def kernel_e_2d_gaussian(p: ndarray, q: ndarray) -> float:
    factor = 4 / (math.pi * sim.epsilon ** 2)
    squared_norm = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
    exponent = -squared_norm / sim.epsilon ** 2
    return factor * math.exp(exponent)
