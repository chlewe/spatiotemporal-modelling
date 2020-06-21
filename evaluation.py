import matplotlib.pyplot as plt

from typing import Tuple, List


def plot_nxm(data: List[Tuple[List, List, List]], n: int, m: int, zlabels=None, titles=None):
    fig = plt.figure()

    for i in range(0, n * m):
        ax = fig.add_subplot(n, m, i + 1, projection='3d')
        surf = ax.plot_trisurf(data[i][0], data[i][1], data[i][2], cmap="jet", linewidth=0.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if zlabels:
            ax.set_zlabel(zlabels[i])
        if titles:
            ax.title.set_text(titles[i])
        fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig
