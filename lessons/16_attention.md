# 16 · Attention

> In one sentence: every token looks at the other tokens, decides how relevant each one is,
> and takes a weighted average of their information.

## Intuition

Each token makes three vectors (each is a [linear layer](05_linear_layer.md) of the token):

| Vector | Question it answers |
|---|---|
| **Q**uery | "What am I looking for?" |
| **K**ey | "What do I contain?" |
| **V**alue | "What will I give you if you pick me?" |

Then, for every token:
1. **score** every other token: dot product `Q · K` (big = good match)
2. turn scores into weights with [softmax](08_softmax.md) (they sum to 1)
3. output = weighted average of the `V`s

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

## Code

From [`modules/function.py`](../modules/function.py):

```python
import math
import torch
from modules.function import softmax

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)           # [T, T]: token i vs token j
    if mask is not None:
        scores = scores.masked_fill(mask, -float('inf'))        # lesson 17
    weights = softmax(scores)                                   # each row sums to 1
    return weights @ V                                          # [T, d]
```

## Try it

Three tokens with 2-dim vectors. Token 0's query matches token 2's key best:

```python
Q = torch.tensor([[0.0, 5.0], [1.0, 0.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 5.0]])
V = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 9.0]])

weights = softmax(Q @ K.T / math.sqrt(2))
print(weights[0])                   # ≈ [0.00, 0.00, 1.00] — token 0 attends to token 2
print(attention(Q, K, V)[0])        # ≈ [0, 9] — so it receives token 2's value
```

## Check yourself

1. What is the shape of the score matrix for a sequence of 6 tokens? Why does attention get expensive for long texts?

<details><summary>Answer</summary>

`[6, 6]` — one score per pair. It grows with T², so doubling the text length makes attention 4× more expensive.
</details>

<details><summary>Going deeper: why divide by <code>√d</code>?</summary>

A dot product of two random `d`-dim vectors has a size of about `√d`. With large `d` the scores get
big, softmax becomes almost one-hot, and its gradients become almost 0 — learning stalls.
Dividing by `√d` keeps the scores around size 1:

```python
d = 512
q, k = torch.randn(1000, d), torch.randn(1000, d)
print((q * k).sum(-1).std().item())                    # ≈ 22.6 (= √512)
print(((q * k).sum(-1) / math.sqrt(d)).std().item())   # ≈ 1.0
```
</details>

---
[Course map](../README.md) · Prev: [15](15_positional_embeddings.md) · Next: [17 · Masks](17_masks.md)
