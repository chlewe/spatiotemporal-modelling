import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List


def plot_nxm(data: List[Tuple[List, List, List]], n: int, m: int, zlabels=None, titles=None):
    fig = plt.figure()

    for i in range(0, n * m):
        ax = fig.add_subplot(n, m, i + 1, projection='3d')
        surf = ax.plot_trisurf(data[i][0], data[i][1], data[i][2], cmap="jet", linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if zlabels:
            ax.set_zlabel(zlabels[i])
        if titles:
            ax.title.set_text(titles[i])

    return fig


def plot_colormap(z_data: List, n: int, m: int, xlabel=None, ylabel=None, title=None):
    z_grid = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            z_grid[i][j] = z_data[i + j * n]

    fig, ax = plt.subplots()
    im = ax.pcolormesh(z_grid, cmap="viridis")
    fig.colorbar(im, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.title.set_text(title)

    return fig
