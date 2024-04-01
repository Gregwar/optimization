import numpy as np
import matplotlib.pyplot as plt

# Data (x_i, y_i)
data = [
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 1.0],
    [4.0, 3.0],
    [5.0, 1.0],
    [6.0, 1.1],
]

data = np.array(data)

# Defining A and b
A = np.array([
    [x**5, x**4, x**3, x**2, x, 1]
    for x in data[:, 0]
])
b = data[:, 1]

# Solving the system
w = np.linalg.solve(A, b)

# Plotting the data and the prediction
ts = np.linspace(0, 7, 100)
prediction = np.polyval(w, ts)

plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.plot(ts, prediction, label='Prediction', color='orange')
plt.title('Quintic polynomial fit to data')
plt.grid()
plt.ylim(-5, 10)
plt.legend()
plt.show()
