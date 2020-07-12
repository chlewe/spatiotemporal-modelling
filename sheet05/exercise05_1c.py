import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from lists import CellList2D, VerletList


def get_pos_of_cell(_cell, _positions):
    _cell_pos = []
    for _pos_index, pos in enumerate(_positions):
        if _pos_index in _cell:
            _cell_pos.append(pos)
    return _cell_pos


if __name__ == "__main__":
    sim.domain_lower_bound = 0
    sim.domain_upper_bound = 1
    sim.cell_side = 0.05
    sim.cutoff = 0.05

    particle_pos = []
    random.seed()

    for _ in range(0, 10000):
        particle_pos.append([random.random(), random.random()])
    particle_pos = np.array(particle_pos)

    cells = CellList2D(particle_pos)
    verlet = VerletList(particle_pos, cells)
    pos_index = random.randint(0, len(particle_pos))

    #######################################
    # Plot particles
    #######################################
    x_coords = particle_pos[:, 0]
    y_coords = particle_pos[:, 1]

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax.set_yticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax.scatter(x_coords, y_coords, s=10, c="b")
    ax.grid()
    ax.set_title("Neighbours of one particle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    #######################################
    # Plot cell
    #######################################
    cell_index = cells.pos_get_cell_index(particle_pos[pos_index])
    cell_pos = np.array(get_pos_of_cell(cells[cell_index], particle_pos))

    x_coords1 = cell_pos[:, 0]
    y_coords1 = cell_pos[:, 1]
    ax.scatter(x_coords1, y_coords1, s=10, c="g")

    #######################################
    # Plot cell neighbours
    #######################################
    cell_neighbour_pos = []
    for neighbour_index in cells.get_neighbour_cell_indices(cell_index):
        cell_neighbour_pos += get_pos_of_cell(cells[neighbour_index], particle_pos)
    cell_neighbour_pos = np.array(cell_neighbour_pos)

    x_coords2 = cell_neighbour_pos[:, 0]
    y_coords2 = cell_neighbour_pos[:, 1]
    ax.scatter(x_coords2, y_coords2, s=10, c="y", label="Cell neighbours")

    #######################################
    # Plot verlet neighbours
    #######################################
    verlet_neighbour_pos = []
    for verlet_neighbour_index in verlet.cells[pos_index]:
        verlet_neighbour = particle_pos[verlet_neighbour_index]
        verlet_neighbour_pos.append(verlet_neighbour)
    verlet_neighbour_pos = np.array(verlet_neighbour_pos)

    x_coords3 = verlet_neighbour_pos[:, 0]
    y_coords3 = verlet_neighbour_pos[:, 1]
    ax.scatter(x_coords3, y_coords3, s=10, c="r", label="Verlet neighbours")

    fig.legend()
    fig.show()
