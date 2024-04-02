import matplotlib.pyplot as plt
import numpy as np
import torch as th
from mlp import MLP

# Create some (noisy) data
xs = np.linspace(0, 6, 1000)
ys = np.sin(xs) * 2.5 + 1.5 + np.random.randn(1000) * 0.1

# Converting the data to tensors
xs = th.tensor(xs, dtype=th.float).unsqueeze(1)
ys = th.tensor(ys, dtype=th.float).unsqueeze(1)

# Creating the network and optimizer
net = MLP(1, 1)
optimizer = th.optim.Adam(net.parameters(), 1e-3)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.9)
losses = []

for epoch in range(512):
    # Showing
    plt.clf()
    plt.plot(xs.numpy(), ys.numpy(), label="sin(x)*2.5 + 1.5")
    with th.no_grad():
        plt.scatter(xs, net(xs).numpy(), label="Neural network", color="orange")
    plt.legend()
    plt.ylim(-2, 5)
    plt.grid()
    plt.title(f"Epoch #{epoch}")

    plt.pause(1e-2)
    # plt.savefig(f"imgs/learn_step_{epoch}.png")

    # Training
    loss = th.nn.functional.mse_loss(net(xs), ys)

    # Optimizing
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    print(f"Epoch {epoch}, loss={loss.item()}, lr={scheduler.get_last_lr()}")
