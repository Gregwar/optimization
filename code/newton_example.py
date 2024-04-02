import numpy as np
import os
import matplotlib.pyplot as plt

def f(x):
    return x**2 + x - 2

def df(x):
    return 2*x + 1

ts = np.linspace(-7, 7, 100)
vs = [f(t) for t in ts]
i = 0

for x_0 in -5, 5:
    i += 1
    polygon = [[x_0, 0]]

    for k in range(5):
        # Computing f
        y_0 = f(x_0)
        polygon.append([x_0, y_0])

        # Computing f'
        dy_0 = df(x_0)

        # Solving y_0 + dy_0 dx = 0
        dx = -y_0 / dy_0
        x_1 = x_0 + dx

        poly = np.array(polygon)
        plt.clf()
        plt.plot(ts, vs)
        plt.plot(poly[:, 0], poly[:, 1], color='orange')
        plt.scatter(poly[:, 0], poly[:, 1], color='orange')
        plt.title(f"Newton method, x={x_0:.2f}, y={y_0:.2f}")
        plt.grid()
        # plt.show()
        plt.savefig(f"imgs/newton_{i}_{k}.png")

        x_0 = x_1
        polygon.append([x_0, 0])

    os.system(f"convert -delay 100 imgs/newton_{i}_*.png imgs/newton_{i}.gif")

