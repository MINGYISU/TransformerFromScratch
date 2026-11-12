"""The smallest interesting neural network: learn XOR (see lessons/07_first_neural_network.md).

    x1  x2 | XOR
     0   0 |  0
     0   1 |  1
     1   0 |  1
     1   1 |  0

No straight line separates the 1s from the 0s, so a single linear layer can't do it.
One hidden layer with an activation can.
"""
import torch
from modules.module import Module, Linear
from modules.function import relu
from modules.loss import mse
from modules.optim import SGD

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])   # [4, 2]
Y = torch.tensor([[0.], [1.], [1.], [0.]])                   # [4, 1]

class XorNet(Module):
    """2 inputs -> hidden layer (bend with ReLU) -> 1 output"""
    def __init__(self, hidden: int=8):
        self.fc1 = Linear(2, hidden)
        self.fc2 = Linear(hidden, 1)

    def _param_keys(self): return 'fc1', 'fc2'
    def _config_keys(self): return self._param_keys()

    def forward(self, x):
        return self.fc2(relu(self.fc1(x)))

def train(steps: int=500, lr: float=0.1, seed: int=0, verbose: bool=True) -> XorNet:
    torch.manual_seed(seed)
    model = XorNet()
    optimizer = SGD(model.parameters(), lr=lr)
    for step in range(steps):
        loss = mse(model(X), Y)      # 1. predict + measure how wrong
        optimizer.zero_grad()
        loss.backward()              # 2. slopes
        optimizer.step()             # 3. step downhill
        if verbose and step % 100 == 0:
            print(f'step {step:4d}  loss {loss.item():.4f}')
    return model

if __name__ == '__main__':
    model = train()
    print(model(X).detach().round(decimals=2).squeeze().tolist())   # ≈ [0, 1, 1, 0]
