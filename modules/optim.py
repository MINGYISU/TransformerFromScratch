import torch

class SGD:
    """Gradient descent: move every parameter a small step downhill (see lessons/04_gradient_descent.md).
        new value = old value - lr * gradient
    """
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """Forget the old gradients (PyTorch adds new gradients on top of old ones)."""
        for p in self.params:
            p.grad = None

    def step(self):
        # no_grad: the update itself is not part of the model's math, so don't track it
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad
