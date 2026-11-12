import math
import torch

class Optimizer:
    """Holds the parameters to train and the learning rate. Subclasses decide how to step."""
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """Forget the old gradients (PyTorch adds new gradients on top of old ones)."""
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    """Gradient descent: move every parameter a small step downhill (see lessons/04_gradient_descent.md).
        new value = old value - lr * gradient
    With momentum (lessons/11_momentum_and_adam.md), we step along a running sum of past gradients instead:
        velocity  = momentum * velocity + gradient
        new value = old value - lr * velocity
    """
    def __init__(self, params, lr: float, momentum: float=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        # no_grad: the update itself is not part of the model's math, so don't track it
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                p -= self.lr * self.velocity[i]

class Adam(Optimizer):
    """Adam = momentum + a separate step size for every parameter (see lessons/11_momentum_and_adam.md).
        m = running average of gradients          (which way to go)
        v = running average of squared gradients  (how big the gradients usually are)
        new value = old value - lr * m / (sqrt(v) + eps)
    """
    def __init__(self, params, lr: float=1e-3, betas=(0.9, 0.999), eps: float=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0  # number of steps taken

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
                # m and v start at 0, so early on they are too small: divide by (1 - beta^t) to correct
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class AdamW(Adam):
    """Adam + weight decay: every step, also shrink every weight a little towards 0.
        new value = old value - lr * weight_decay * old value   (then the normal Adam step)
    """
    def __init__(self, params, lr: float=1e-3, betas=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * self.weight_decay * p
        super().step()

def warmup_cosine_lr(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float=0.0) -> float:
    """Learning rate schedule (see lessons/12_training_tricks.md):
    climb linearly from ~0 to max_lr during warmup, then glide down to min_lr along a cosine curve.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))  # 0 -> 1
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def clip_grad_norm(params, max_norm: float) -> float:
    """If the gradient of all parameters together is longer than max_norm, shrink it to max_norm.
    The direction stays the same; only the step size is limited. Returns the length before clipping.
    """
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = math.sqrt(sum((g ** 2).sum().item() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale
    return total_norm
