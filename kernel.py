import math

from lists import *


def kernel_e_1d_gaussian(p: Particle1D, q: Particle1D, epsilon: float) -> float:
    dist_x = q.x - p.x
    return 1 / (2 * epsilon * math.sqrt(math.pi)) * math.exp(-dist_x ** 2 / (4 * epsilon ** 2))


def kernel_e_1d_poly(p: Particle1D, q: Particle1D, h: float) -> float:
    dist_x = q.x - p.x
    return 1 / h * 15 / math.pi ** 2 / (abs(dist_x / h) ** 10 + 1)


def kernel_e_2d_gaussian(p: Particle2D2, q: Particle2D2, epsilon: float) -> float:
    factor = 4 / (math.pi * epsilon ** 2)
    squared_norm = (q.x - p.x) ** 2 + (q.y - p.y) ** 2
    exponent = -squared_norm / epsilon ** 2
    return factor * math.exp(exponent)
