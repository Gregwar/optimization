import numpy as np

# Parameters to identify
x_b = 1.25
y_b = 2.53
l1 = 0.56
l2 = 0.78


def effector_position(alpha, beta):
    return [
        x_b + l1 * np.cos(alpha) + l2 * np.cos(alpha + beta),
        y_b + l1 * np.sin(alpha) + l2 * np.sin(alpha + beta),
    ]


data = []
for k in range(128):
    alpha = np.random.uniform(0, 2 * np.pi)
    beta = np.random.uniform(0, 2 * np.pi)

    effector = effector_position(alpha, beta) + np.random.normal(0, 0.25, 2)
    data.append([alpha, beta, effector[0], effector[1]])

np.savetxt("data/rr_data.csv", np.array(data))
