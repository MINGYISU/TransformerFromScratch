# 15 · Positional embeddings

> In one sentence: we add a "where am I" vector to each token vector, so the model can tell
> `dog bites man` from `man bites dog`.

## The idea

Attention (lesson 16) looks at all tokens as an unordered **set**. Without help it cannot know
which token came first. The fix is tiny: keep a second table with one vector per **position**
and add it to the token vector. For the token at position $t$:

```math
x_t = \text{TokenEmbed}(\text{token}_t) + \text{PosEmbed}(t)
```

GPT-2 simply **learns** the position table, just like the token table. That's what we do.

## Code

`MyGPT.encode` in [`models/example/model.py`](../models/example/model.py):

```python
import torch

V, E, max_seq_length = 87, 4, 100
token_embed = torch.randn(V, E) * 0.02
pos_embed = torch.randn(max_seq_length, E) * 0.02     # one row per position

def encode(ids):
    B, T = ids.shape
    return token_embed[ids] + pos_embed[:T]           # broadcasting adds pos to every batch
```

## Try it

```python
x = encode(torch.tensor([[1, 16, 16]]))   # 'app'
print(torch.equal(x[0, 1], x[0, 2]))      # False — the two 'p's now differ
```

`max_seq_length` is a hard limit: there is no row for position 100, so the model cannot read longer texts.

## Check yourself

1. What would happen if we *concatenated* instead of *added* the position vector?

<details><summary>Answer</summary>

It also works, but every vector becomes twice as long (more compute everywhere).
Adding is cheaper, and in high dimensions the model can still separate the two signals.
</details>

<details><summary>Going deeper: sinusoidal positions and RoPE</summary>

**Sinusoidal** (original Transformer, 2017): no learning — position $p$ gets waves of different speeds.
Nearby positions get similar vectors, and it works for any length. Number $2i$ and $2i+1$ of the vector are:

```math
PE_{p, 2i} = \sin\left( \frac{p}{10000^{2i/E}} \right), \qquad
PE_{p, 2i+1} = \cos\left( \frac{p}{10000^{2i/E}} \right)
```

```python
def sinusoidal(T, E):
    pos = torch.arange(T).unsqueeze(1)                          # [T, 1]
    freq = 1.0 / (10000 ** (torch.arange(0, E, 2) / E))         # [E/2]
    pe = torch.zeros(T, E)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe

print(sinusoidal(T=3, E=4))
```

**RoPE** (LLaMA and most modern LLMs): instead of adding, it *rotates* the query and key
vectors by an angle that depends on the position. Then the attention score between two
tokens depends only on how far apart they are.
</details>

---
[Course map](../README.md) · Prev: [14](14_token_embeddings.md) · Next: [16 · Attention](16_attention.md)
