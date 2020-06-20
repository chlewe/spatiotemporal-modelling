import collections
import math

from typing import List, Tuple


def distance(pos1, pos2):
    squared_distance = 0
    for i in range(0, len(pos1)):
        squared_distance += (pos1[i] - pos2[i]) ** 2
    return math.sqrt(squared_distance)


Particle = collections.namedtuple("Particle", ["x", "y", "strength"])
CellIndex2D = collections.namedtuple("CellIndex2D", ["x", "y"])


class CellList2D:

    def __init__(self, positions: List[Tuple[float, float]], lower_bound: int, upper_bound: int, cell_side: float):
        # self.lower_bound = lower_bound
        # self.upper_bound = upper_bound
        self.cell_side = cell_side
        self.dim_width = math.floor((upper_bound - lower_bound) / cell_side) + 1
        self.cells: List[List[List[int]]] = [[[] for _ in range(0, self.dim_width)] for _ in range(0, self.dim_width)]

        for pos_index, pos in enumerate(positions):
            cell_index = self.xy_get_cell_index(pos[0], pos[1])
            self[cell_index].append(pos_index)

    def xy_get_cell_index(self, x: float, y: float):
        x_index = math.floor(x / self.cell_side)
        y_index = math.floor(y / self.cell_side)
        return CellIndex2D(x_index, y_index)

    def get_neighbour_cell_indices(self, cell_index: CellIndex2D):
        neighbour_indices = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = cell_index.x + dx
                y = cell_index.y + dy

                if 0 <= x < self.dim_width and 0 <= y < self.dim_width:
                    neighbour_indices.append(CellIndex2D(x, y))
        return neighbour_indices

    def __getitem__(self, item: CellIndex2D):
        return self.cells[item.x][item.y]


class VerletList2D:

    def __init__(self, positions: List[Tuple[float, float]], cell_list: CellList2D, cutoff: float):
        num_particles = len(positions)
        self.cells = [[] for _ in range(0, num_particles)]

        for pos_index, pos in enumerate(positions):
            cell_index = cell_list.xy_get_cell_index(pos[0], pos[1])
            neighbour_indices = cell_list.get_neighbour_cell_indices(cell_index)

            for neighbour_index in neighbour_indices:
                for other_pos_index in cell_list[neighbour_index]:
                    other_pos = positions[other_pos_index]
                    if distance(pos, other_pos) <= cutoff:
                        self.cells[pos_index].append(other_pos_index)

    def __getitem__(self, item: int):
        return self.cells[item]
