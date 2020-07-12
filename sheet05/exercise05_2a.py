import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
import sim

from lists import CellList2D, VerletList


def load_bacteria(bacteria_file):
    _bacteria_pos = []
    with open(bacteria_file, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            x, y = line.split("   ")[1:3]
            _bacteria_pos.append((float(x), float(y)))
    return np.array(_bacteria_pos)


def get_pos_of_cell(_cell, _positions):
    _cell_pos = []
    for _pos_index, pos in enumerate(_positions):
        if _pos_index in _cell:
            _cell_pos.append(pos)
    return _cell_pos


if __name__ == "__main__":
    sim.domain_lower_bound = 0
    sim.domain_upper_bound = 10
    sim.cell_side = 0.5
    sim.cutoff = 0.5

    bacteria_pos = load_bacteria("QSBacterialPos.dat")
    cells = CellList2D(bacteria_pos)
    verlet = VerletList(bacteria_pos, cells)

    x_coords = bacteria_pos[:, 0]
    y_coords = bacteria_pos[:, 1]

    #######################################
    # Figure 1: Cells with high density
    #######################################
    high_density_pos = []
    for cell_row in cells.cells:
        for cell in cell_row:
            if len(cell) >= 20:
                high_density_pos += get_pos_of_cell(cell, bacteria_pos)
    high_density_pos = np.array(high_density_pos)
    x_coords1 = high_density_pos[:, 0]
    y_coords1 = high_density_pos[:, 1]

    fig1, ax1 = plt.subplots()
    ax1.set_title("Cells with high density")
    ax1.set_xticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax1.set_yticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax1.scatter(x_coords, y_coords, s=10, c="b")
    ax1.grid()

    ax1.scatter(x_coords1, y_coords1, s=10, c="r")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    #######################################
    # Figure 2: Cells with many neighbours
    #######################################
    many_neighbour_pos = []
    most_neighbours = 0
    pos_index_most_neighbours = 0

    for (pos_index, cell) in enumerate(verlet.cells):
        neighbours = len(cell)
        if neighbours >= 50:
            many_neighbour_pos.append(bacteria_pos[pos_index])
        if neighbours > most_neighbours:
            most_neighbours = neighbours
            pos_index_most_neighbours = pos_index

    many_neighbour_pos = np.array(many_neighbour_pos)
    x_coords2 = many_neighbour_pos[:, 0]
    y_coords2 = many_neighbour_pos[:, 1]

    fig2, ax2 = plt.subplots()
    ax2.set_title("Bacteria with $\geq 50$ neighbours")
    ax2.set_xticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax2.set_yticks(np.arange(sim.domain_lower_bound, sim.domain_upper_bound, sim.cell_side))
    ax2.scatter(x_coords, y_coords, s=10, c="b")
    ax2.grid()

    ax2.scatter(x_coords2, y_coords2, s=10, c="r")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.show()

    #######################################
    # Extra: Bacterium with most neighbours
    #######################################

    x_most_neighbours, y_most_neighbours = bacteria_pos[pos_index_most_neighbours]

    print("The bacterium with most neighbours is at x={}, y={} with {} neighbours"
          .format(x_most_neighbours, y_most_neighbours, most_neighbours))
