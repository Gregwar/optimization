import torch as th

x = th.zeros(1, requires_grad=True)
y = th.zeros(1, requires_grad=True)

optimizer = th.optim.Adam([x, y], 1e-3)

while True:
    optimizer.zero_grad()
    e = x**2 + y**2 + 3*x + 2*y -5.
    e.backward()
    optimizer.step()

    print(f"x={x.item()}, y={y.item()}, e={e.item()}")