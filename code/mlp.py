import torch as th


class MLP(th.nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()

        self.net = th.nn.Sequential(
            th.nn.Linear(input_dimension, 256),
            th.nn.ELU(),
            th.nn.Linear(256, 256),
            th.nn.ELU(),
            th.nn.Linear(256, 256),
            th.nn.ELU(),
            th.nn.Linear(256, output_dimension),
        )

    def forward(self, x):
        return self.net(x)
