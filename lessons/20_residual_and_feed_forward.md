# 20 · Residual connections and the feed-forward network

> In one sentence: the feed-forward network lets each token "think" on its own, and the
> residual connection $x + f(x)$ makes sure no layer can destroy what came before.

## The idea

**Feed-forward (FFN / MLP).** Attention moves information *between* tokens. The FFN then processes
each token *alone*: grow the vector 4× wider, apply [GELU](06_activations.md), shrink it back.

```
[E] --Linear--> [4E] --GELU--> [4E] --Linear--> [E]
```

```math
\text{FFN}(x) = \text{GELU}(x W_1^\top + b_1) W_2^\top + b_2
```

**Residual connection.** Instead of replacing $x$ with $f(x)$, we add:

```math
x \leftarrow x + f(x)
```

 Each layer only adds a
small *correction* to the vector. If a layer is useless it can learn $f(x) \approx 0$ and do no harm.
It also gives gradients a direct path back to early layers, which is why deep models can train at all.

## Code

`FeedForward` from [`modules/module.py`](../modules/module.py):

```python
import torch
from modules.module import Module, Linear
from modules.function import gelu

class MyFeedForward(Module):
    def __init__(self, embed_dim, hidden_dim):
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def _param_keys(self): return 'fc1', 'fc2'
    def _config_keys(self): return self._param_keys()
    def forward(self, x): return self.fc2(gelu(self.fc1(x)))
```

## Try it

```python
ffn = MyFeedForward(embed_dim=8, hidden_dim=32)
x = torch.randn(2, 5, 8)
print(ffn(x).shape)                 # [2, 5, 8] — same shape in and out, so we can add it to x
x = x + ffn(x)                      # residual connection

# each token is processed independently: changing token 0 doesn't change token 1's output
a = torch.randn(1, 2, 8); b = a.clone(); b[0, 0] += 1.0
print(torch.allclose(ffn(a)[0, 1], ffn(b)[0, 1]))   # True
```

## Check yourself

1. Why must $f(x)$ have exactly the same shape as $x$?

<details><summary>Answer</summary>

Because we add them: $x + f(x)$. That's why the FFN shrinks back to `E`, and attention ends with an `E → E` projection.
</details>

<details><summary>Going deeper: how residuals help gradients</summary>

The gradient of $x + f(x)$ with respect to $x$ is

```math
\frac{d}{dx} \big( x + f(x) \big) = 1 + f'(x)
```

Even if $f'(x)$ is tiny, the $1$
passes the gradient straight through. Without it, gradients get multiplied by many small numbers
on the way back and vanish.

```python
x = torch.tensor(2.0, requires_grad=True)
(x + 0.001 * x).backward()
print(x.grad.item())                # 1.001 — the "1" survives
```
</details>

---
[Course map](../README.md) · Prev: [19](19_layer_norm.md) · Next: [21 · The transformer block](21_transformer_block.md)
