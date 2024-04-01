import numpy as np
import matplotlib.pyplot as plt


# Data (x_i, y_i)
data = [
    [1.0, 2.0],
    [2.0, 3.1],
    [3.0, 3.8],
    [4.0, 5.2],
    [5.0, 5.9],
    [6.0, 7.1],
]
data = np.array(data)

sum_x = np.sum(data[:, 0])
sum_y = np.sum(data[:, 1])
sum_x2 = np.sum(data[:, 0] ** 2)
sum_xy = np.sum(data[:, 0] * data[:, 1])
n = data.shape[0]

w2 = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - (sum_x)**2)
w1 = (sum_y - w2*sum_x) / n

xs = np.linspace(0, 7, 100)
ys = w1 + w2*xs

# Plot data
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.plot(xs, ys, label='Linear regression', color="orange")
plt.legend()
plt.grid()
plt.show()
