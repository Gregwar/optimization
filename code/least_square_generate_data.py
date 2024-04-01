import numpy as np
import matplotlib.pyplot as plt

models = {
    'f(x) = w_1 x + w_2': 
        lambda x: 1.4*x - 5.4,
    'f(x) = w_1 x^2 + w_2 x + w_3':
        lambda x: 0.5*x**2 - 2.1*x + 1.2,
    'f(x) = w_1 cos(x) + w_2 x + w_3':
        lambda x: 3.1*np.cos(x*2) + 1.2*x - 2.3,
    'f(x) = w_1 log(x) + w_2 sin(x)':
        lambda x: 3.2*np.log(x) + 3.*np.sin(x),
}

for k, model in enumerate(models):
    xs = np.random.uniform(0, 10, 100)
    ys = [models[model](x) + np.random.normal(0, 1) for x in xs]
    np.savetxt("data/data_" + str(k) + ".csv", np.array([xs, ys]).T)

    plt.clf()
    plt.scatter(xs, ys, label="Data")
    plt.legend()
    plt.grid()
    # plt.title("Model: " + model)

    plt.savefig("imgs/data_" + str(k) + ".png")
    # plt.show()