import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from utils import *

# Exercise 2a)
if __name__ == "__main__":
    bacteria_pos = []
    with open("QSBacterialPos.dat", "r") as f:
        for line in f:
            line = line.replace("\n", "")
            pos = list(map(lambda x: float(x), line.split("   ")[1:3]))
            bacteria_pos.append(pos)
    bacteria_pos = np.array(bacteria_pos)

    cells = CellList(bacteria_pos, 0, 10, 0.5)
    verlet = VerletList(bacteria_pos, cells, 0.5)

    x = bacteria_pos[:,0]
    y = bacteria_pos[:,1]

    # Left figure (Cell List)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.set_yticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.scatter(x, y, s=10, c="b")
    plt.grid()

    # Cells with high density
    high_density_pos = []
    for (cell_index, cell) in enumerate(cells.cells):
        if len(cell) >= 20:
            high_density_pos += cells.get_pos_of_cell(bacteria_pos, cell_index)
    high_density_pos = np.array(high_density_pos)
    x1 = high_density_pos[:,0]
    y1 = high_density_pos[:,1]
    ax.scatter(x1, y1, s=10, c="r")


    # Right Figure (Verlet)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.set_yticks(np.arange(0, cells.upper_bound, cells.cell_width))
    ax.scatter(x, y, s=10, c="b")
    plt.grid()

    # Cells with many Vetlet neighbours
    # Cell with most neighbours
    high_density_pos = []
    most_neighbours = 0
    pos_index_most_neighbours = 0
    for (pos_index, cell) in enumerate(verlet.cells):
        neighbours = len(cell)
        if neighbours >= 50:
            high_density_pos.append(bacteria_pos[pos_index])
        if neighbours > most_neighbours:
            most_neighbours = neighbours
            pos_index_most_neighbours = pos_index

    high_density_pos = np.array(high_density_pos)
    x1 = high_density_pos[:,0]
    y1 = high_density_pos[:,1]
    ax.scatter(x1, y1, s=10, c="r")

    print("The cell with most neighbours is at {} with {} neighbours"\
            .format(bacteria_pos[pos_index_most_neighbours], most_neighbours))
    plt.show()
