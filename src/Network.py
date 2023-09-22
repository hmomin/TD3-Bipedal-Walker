from typing import Callable
import torch as T
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    def __init__(
        self,
        shape: list,
        outputActivation: Callable,
        learningRate: float,
        device: T.device,
    ):
        super().__init__()
        # initialize the network
        layers = []
        for i in range(1, len(shape)):
            dim1 = shape[i - 1]
            dim2 = shape[i]
            layers.append(nn.Linear(dim1, dim2))
            if i < len(shape) - 1:
                layers.append(nn.ReLU())
        layers.append(outputActivation())
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.device = device
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        return self.network(state)

    def gradientDescentStep(self, loss: T.Tensor, retainGraph: bool = False) -> None:
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retainGraph)
        self.optimizer.step()
