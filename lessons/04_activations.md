# 04 · Activation functions

> In one sentence: an activation is a simple bend applied to every number, so stacked layers
> can learn curved shapes instead of only straight lines.

## Intuition

Two linear layers in a row are still just one linear layer:
`(x W1) W2 = x (W1 W2)`. No matter how many you stack, you can only draw straight lines.
Put a **non-linear** function between them and suddenly the network can bend.

- **ReLU**: `max(0, x)` — keep positives, zero out negatives. The simplest choice.
- **GELU**: a smooth version of ReLU. GPT uses it.

## Code

```python
import math
import torch

def relu(x):
    return torch.clamp(x, min=0)

def gelu(x):
    # x times "the probability a standard normal value is below x"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

`gelu` is in [`modules/function.py`](../modules/function.py).

## Try it

```python
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(relu(x))   # [0, 0, 0, 1, 3]
print(gelu(x))   # ≈ [-0.004, -0.159, 0, 0.841, 2.996]
```

GELU is almost ReLU for big `|x|`, but smooth near 0 and lets a little negative signal through.

```python
W1, W2 = torch.randn(4, 4), torch.randn(4, 4)
x = torch.randn(4)
print(torch.allclose((x @ W1) @ W2, x @ (W1 @ W2), atol=1e-5))      # True: two linears = one linear
print(torch.allclose(gelu(x @ W1) @ W2, x @ (W1 @ W2), atol=1e-5))  # False: GELU breaks this
```

## Check yourself

1. What is the gradient of ReLU for `x = -2`? Why can this be a problem?

<details><summary>Answer</summary>

0 — a neuron that only ever sees negative inputs never learns ("dead ReLU").
GELU's gradient is small but not exactly 0 there.
</details>

<details><summary>Going deeper: the tanh approximation</summary>

The original GPT-2 code used a faster approximation:
`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))`. It differs from exact GELU by less than 0.001.

```python
approx = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
print((approx - gelu(x)).abs().max().item() < 1e-3)   # True
```
</details>

---
[Course map](../README.md) · Prev: [03](03_linear_layer.md) · Next: [08 · Softmax](08_softmax.md)
*(lessons 05–07 are coming soon)*
