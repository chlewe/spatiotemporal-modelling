import numpy as np
import matplotlib.pyplot as plt
import math

c_1 = 1
c_2 = 1
x_T = 2 * 10 ** 5
x_0 = 0.8 * 10 ** 5

def euler(x_0, step_size, total_steps, f):
    x = [x_0]
    t = [0]

    for n in range(0, total_steps):
        t.append(t[-1] + step_size)
        x.append(x[-1] + step_size * f(t[-1], x[-1]))

    return np.array(x), np.array(t)


def f_isomer(t, x_t):
    global c_1, c_2, x_T, x_0
    return -c_1 * x_t + c_2 * (x_T - x_t)

def f_euler(t, x):
    return x

#x, t = euler(x_0, 1.0, 4, f_euler)
#plt.plot(t, x)

t_max = 10

for step_size in np.arange(0.25, 1.5, 0.25):
    x, t = euler(x_0, step_size, int(t_max / step_size), f_isomer)
    plt.plot(t, x, label="{}".format(step_size))

plt.legend()
plt.show()
