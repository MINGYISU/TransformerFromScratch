# 03 · Linear layer

> In one sentence: a linear layer turns a vector of `in` numbers into a vector of `out` numbers,
> where every output is a weighted sum of all inputs plus a bias.

## Intuition

Each output number asks: *"how much of each input do I want?"* The answers are the weights
`W` (one row per output) and the bias `b` is a constant offset. Written as math:
`y = x Wᵀ + b`. Almost every learnable part of a transformer is built from this.

## Code

The math, in [`modules/function.py`](../modules/function.py):

```python
import torch

def linear(x, weight, bias=None):
    # x: [..., in], weight: [out, in], bias: [out]  ->  [..., out]
    out = x @ weight.T
    return out + bias if bias is not None else out
```

The class that **owns** the weights, in [`modules/module.py`](../modules/module.py)
(simplified — the real one also tracks its parameters so we can save and train it):

```python
import math
from modules.module import Module, Parameter

class MyLinear(Module):
    def __init__(self, in_features, out_features):
        self.W = Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.b = Parameter(torch.zeros(out_features))

    def _param_keys(self): return 'W', 'b'
    def _config_keys(self): return self._param_keys()
    def forward(self, x): return linear(x, self.W, self.b)
```

`Parameter` is just a tensor that says "I'm a weight, please train me" (it has `requires_grad=True`).

## Try it

```python
layer = MyLinear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])     # [1, 3]
print(layer(x).shape)                    # [1, 2]
print([name for name, _ in layer.named_parameters()])   # ['W', 'b']
```

## Check yourself

1. How many learnable numbers does `Linear(512, 2048)` have?

<details><summary>Answer</summary>

`512 * 2048` weights `+ 2048` biases `= 1,050,624`.
</details>

<details><summary>Going deeper: why divide by <code>√in_features</code>?</summary>

Each output is a sum of `in_features` random products. Adding up `n` random numbers makes
the result about `√n` times bigger. Stack a few layers and the numbers explode — then
softmax saturates and training stalls. Dividing the weights by `√n` keeps outputs at about the same size
as the inputs:

```python
x = torch.randn(1000, 512)
print((x @ torch.randn(512, 512)).std().item())                     # ≈ 22.6 (= √512)
print((x @ (torch.randn(512, 512) / math.sqrt(512))).std().item())  # ≈ 1.0
```

This repo used plain `torch.randn` at first, and training was much worse. This is called
"LeCun" initialization; GPT-2 instead uses a fixed small std (0.02).
</details>

---
[Course map](../README.md) · Prev: [02](02_autograd.md) · Next: [04 · Activations](04_activations.md)
