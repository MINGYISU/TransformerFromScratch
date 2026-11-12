# 14 · Token embeddings

> In one sentence: an embedding table gives every token id its own learnable vector.

## Intuition

A token id like `16` is just a name — "16" is not "bigger" than "5" in any useful way.
So we keep a table with one row per token: row `i` is the vector for token `i`.
Looking up a token = picking its row. These vectors start random and are **learned**,
so tokens used in similar ways end up with similar vectors.

```
token_embed: [V, E]      ids: [B, T]   ->   token_embed[ids]: [B, T, E]
```

## Code

From `MyGPT.encode` in [`models/example/model.py`](../models/example/model.py):

```python
import torch

V, E = 87, 4
token_embed = torch.randn(V, E) * 0.02     # the table (a Parameter in the real model)

ids = torch.tensor([[1, 16, 16]])          # 'app'   [B=1, T=3]
x = token_embed[ids]                       # just indexing!
print(x.shape)                             # [1, 3, 4]
```

## Try it

Looking up a row is the same as multiplying a **one-hot** vector by the table.
This is why an embedding is really just a linear layer without bias:

```python
one_hot = torch.zeros(V)
one_hot[16] = 1.0
print(torch.allclose(one_hot @ token_embed, token_embed[16]))   # True
```

Both `p` tokens in `'app'` get the exact same vector — the embedding knows *what* a token is, not *where* it is:

```python
print(torch.equal(x[0, 1], x[0, 2]))   # True  -> we fix this in lesson 15
```

## Check yourself

1. GPT-2 small has `V = 50257` and `E = 768`. How many numbers are in its embedding table?

<details><summary>Answer</summary>

`50257 * 768 ≈ 38.6 million` — about a third of GPT-2 small's 124M parameters.
</details>

<details><summary>Going deeper: why <code>* 0.02</code>?</summary>

The same table is reused at the end of the model to score every token (weight tying, lesson 22).
If the vectors start large, the very first predictions are wildly over-confident and training
starts badly. GPT-2 initialises them with a standard deviation of 0.02.
</details>

---
[Course map](../README.md) · Prev: [13](13_tokenizer.md) · Next: [15 · Positional embeddings](15_positional_embeddings.md)
