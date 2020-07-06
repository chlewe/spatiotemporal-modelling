import numpy as np

from lists import *
from typing import List, Dict


# FIXME: Currently broken due to `update_particle_2d`
def pse_operator_1d(_particles: List[Particle1D1], _verlet: VerletList, _kernel_e, env: Environment)\
        -> List[Particle1D]:
    for _ in np.arange(0, env.t_max, env.dt):
        updated_particles = []
        for (i, p) in enumerate(_particles):
            x, y, updated_strengths = update_particle_2d(i, _verlet.cells[i], _particles, env, _kernel_e)
            updated_particles.append(create_particle_1d(p.x, updated_strengths))

        _particles = updated_particles

    return _particles


def pse_predict_u_1d(_particles: List[Particle1D1], start_x, end_x, env: Environment)\
        -> Tuple[List[float], List[float]]:
    _x_coords = []
    _concentration = []

    for p in filter(lambda _p: start_x <= _p.x <= end_x, _particles):
        _x_coords.append(p.x)
        _concentration.append(p.strength0 / env.volume_p)

    return _x_coords, _concentration
