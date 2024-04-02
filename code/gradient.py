import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

def model(X, Y):
    return X**2 + 1.5*Y**2 - 2.*X + 10*np.exp(-(X**2 + 2*Y**2))

def grad(X, Y):
    return np.array([2*X - 20*X*np.exp(-(X**2 + 3*Y**2)) -2., 3*Y - 40*Y*np.exp(-(X**2 + 2*Y**2))])

X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = model(X, Y)
X0 = [-1.5, 1.75]

Xs = []
Ys = []
Zs = []
learning_rate = 1e-2
G = np.zeros(2)

for k in range(512):
    Z0 = model(X0[0], X0[1]) + 0.05

    Xs.append(X0[0])
    Ys.append(X0[1])
    Zs.append(Z0)

    G = grad(X0[0], X0[1])
    X1 = X0 - learning_rate*G


    plt.clf()
    ax = plt.axes(projection='3d', computed_zorder=False)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d", "compute_zorder": True})

    ax.view_init(elev=30, azim=0 + 30*np.cos(k/20.), roll=0)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                        facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))

    ax.scatter(Xs, Ys, Zs, color='red')

    # Customize the z axis.
    ax.set_zlim(0, 10)

    plt.title(f"Step {k}")
    # plt.show()
    plt.savefig(f"imgs/gradient_step_{k}.png")

    X0 = X1

# Build the palette
# ffmpeg -i gradient_step_%d.png -vf palettegen palette.png
# Make the GIF
# ffmpeg -i gradient_step_%d.png -i palette.png -lavfi paletteuse gradient_steps.gif