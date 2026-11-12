# 19 · Layer normalization

> In one sentence: LayerNorm rescales each token's vector to mean 0 and spread 1, then lets the
> model learn its own preferred scale and shift.

## Intuition

As vectors pass through many layers, their numbers can drift very large or very small, which
makes training unstable. LayerNorm resets them, **for every token separately**:

1. subtract the mean of the vector → centred at 0
2. divide by its standard deviation → spread of 1
3. multiply by `gamma` and add `beta` (both learned, start at 1 and 0)

## Code

From [`modules/function.py`](../modules/function.py):

```python
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta     # eps avoids dividing by 0
```

## Try it

```python
x = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
y = layer_norm(x, gamma=torch.ones(4), beta=torch.zeros(4))
print(y)                              # ≈ [-0.60, -0.58, -0.55, 1.73]
print(y.mean().item(), y.std(unbiased=False).item())   # ≈ 0.0, 1.0
```

## Check yourself

1. "Layer" norm normalises over the last axis `E`. Why not over the batch axis `B` (that's *batch* norm)?

<details><summary>Answer</summary>

With LayerNorm, each token is normalised on its own, so the result doesn't depend on which other texts
are in the batch — and it works the same for a batch of 1 during generation.
</details>

<details><summary>Going deeper: RMSNorm</summary>

LLaMA and most newer models use **RMSNorm**: skip the mean, just divide by the root-mean-square.
One less step, works just as well.

```python
def rms_norm(x, gamma, eps=1e-5):
    return gamma * x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)

print(rms_norm(x, torch.ones(4)))
```
</details>

---
[Course map](../README.md) · Prev: [18](18_multi_head_attention.md) · Next: [20 · Residuals and feed-forward](20_residual_and_feed_forward.md)
