import numpy as np
import matplotlib.pyplot as plt


def draw_rr(alpha, beta, l1, l2, pause=None):

    x1 = l1 * np.cos(alpha)
    y1 = l1 * np.sin(alpha)
    x2 = x1 + l2 * np.cos(alpha + beta)
    y2 = y1 + l2 * np.sin(alpha + beta)
    plt.clf()
    plt.plot([0, x1], [0, y1], "o-")
    plt.plot([x1, x2], [y1, y2], "o-")
    plt.plot(x2, y2, "o")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid()
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)
