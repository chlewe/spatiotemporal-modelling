import numpy as np
import matplotlib.pyplot as plt
import sys

from enum import Enum

c_1 = 1
c_2 = 1
x_T = 2 * 10 ** 5
x_0 = 0.8 * 10 ** 5


class Main(Enum):
    APPROXIMATE_ISOMER_FUNCTION = 0
    APPROXIMATE_EULER_FUNCTION = 1


#######################################
# Select your main method:
main = Main.APPROXIMATE_ISOMER_FUNCTION
#######################################


def explicit_euler(_x_0, _dt, _t_max, _f):
    _x_coords = [_x_0]
    _t_coords = [0]

    for _ in np.arange(0, _t_max, _dt):
        _t_coords.append(_t_coords[-1] + dt)
        _x_coords.append(_x_coords[-1] + dt * f(_t_coords[-1], _x_coords[-1]))

    return np.array(_x_coords), np.array(_t_coords)


def f_isomer(t, x_t):
    return -c_1 * x_t + c_2 * (x_T - x_t)


def f_euler(t, x_t):
    return x_t


if __name__ == "__main__":
    t_max = 10
    fig, ax = plt.subplots()

    if main == Main.APPROXIMATE_ISOMER_FUNCTION:
        f = f_isomer
        ax.set_title("Approximation of given isomer function")
    elif main == Main.APPROXIMATE_EULER_FUNCTION:
        f = f_euler
        ax.set_title("Approximation of Euler function")
    else:
        sys.exit(1)

    for dt in np.arange(0.25, 1.5, 0.25):
        x_coords, t_coords = explicit_euler(x_0, dt, t_max, f)
        ax.plot(t_coords, x_coords, label="{}".format(dt))

    ax.set_xlabel("t")
    ax.set_ylabel("x")
    fig.legend()
    plt.show()
