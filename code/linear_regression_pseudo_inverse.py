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

# Building A and b
A = [
    [1.0, x]
    for x in data[:, 0]
]
b = data[:, 1]

# Using pseudo-inverse
w = np.linalg.pinv(A) @ b
    
# Plot linear regression
xs = np.linspace(0, 7, 100)
ys = [w[0] + w[1] * x for x in xs]

# Plot data
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.plot(xs, ys, label='Linear regression', color="orange")
plt.legend()
plt.grid()
plt.show()
