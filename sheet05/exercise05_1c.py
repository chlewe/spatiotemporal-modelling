import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from lists import *

# Exercise 1c)
if __name__ == "__main__":
    positions = []
    random.seed()

    for _ in range(0, 10000):
        positions.append([random.random(), random.random()])
    positions = np.array(positions)

    cells = CellList(positions, 0, 1, 0.05)
    verlet = VerletList(positions, cells, 0.05)
    position_index = random.randint(0, len(positions))

    ## Plotting
    x = positions[:, 0]
    y = positions[:, 1]

    fig, ax = plt.subplots(figsize=(10,10))
    ## Plot particles and cell grid
    ax.set_xticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.set_yticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.scatter(x, y, s=10, c="b")

    ## Plot cell
    cell_index = random.randint(0, cells.dim_width ** cells.dim_number)
    print(cell_index)
    cell_positions = np.array(cells.get_pos_of_cell(positions, cell_index))
    x = cell_positions[:,0]
    y = cell_positions[:,1]
    ax.scatter(x, y, s=10, c="g")

    ## Plot neighbouring cells
    neighbour_positions = []
    for neighbour_index in cells.get_neighbour_cell_indices(cell_index):
        neighbour_positions += cells.get_pos_of_cell(positions, neighbour_index)
    neighbour_positions = np.array(neighbour_positions)
    x = neighbour_positions[:,0]
    y = neighbour_positions[:,1]
    ax.scatter(x, y, s=10, c="y")

    ## Plot verlet neighbours
    verlet_neighbours = []
    for verlet_neighbour_index in verlet.cells[position_index]:
        verlet_neighbour = positions[verlet_neighbour_index]
        verlet_neighbours.append(verlet_neighbour)
    verlet_neighbours = np.array(verlet_neighbours)
    x = verlet_neighbours[:,0]
    y = verlet_neighbours[:,1]
    ax.scatter(x, y, s=10, c="r")


    plt.grid()
    plt.show()
