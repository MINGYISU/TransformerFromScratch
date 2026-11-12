# 01 · Tensors and shapes

> In one sentence: a tensor is a grid of numbers, and its **shape** tells you what each axis means.

## Intuition

Almost every bug in a transformer is a shape bug. So we give every axis a name and write it
in a comment next to the code. In this whole course we use:

| Letter | Meaning | Example |
|---|---|---|
| `B` | batch size — how many texts at once | 4 |
| `T` | sequence length — tokens per text | 6 |
| `E` | embedding size — numbers per token | 32 |
| `V` | vocabulary size — how many different tokens exist | 87 |
| `H` | number of attention heads | 2 |

So `[B, T, E]` means: *B texts, each with T tokens, each token is a vector of E numbers.*

## Try it

```python
import torch

x = torch.randn(4, 6, 32)   # [B, T, E]
print(x.shape)              # torch.Size([4, 6, 32])
print(x[0].shape)           # first text:            [T, E]
print(x[0, 2].shape)        # 3rd token of 1st text: [E]
```

**Broadcasting**: when shapes differ, PyTorch repeats the smaller tensor along the missing axes.
We use this to add the same positional vector to every text in the batch (lesson 15):

```python
pos = torch.randn(6, 32)        # [T, E]  — one vector per position
y = x + pos                     # [B, T, E] + [T, E] -> [B, T, E]
print(y.shape)
```

**Matrix multiply** `@` works on the last two axes, and treats the rest as a batch:

```python
W = torch.randn(32, 10)         # [E, 10]
print((x @ W).shape)            # [B, T, E] @ [E, 10] -> [B, T, 10]
```

## Check yourself

1. `a` has shape `[4, 6, 32]`, `b` has shape `[32]`. What is the shape of `a + b`?
2. What is the shape of `x @ x.transpose(-2, -1)` for `x` of shape `[B, T, E]`?

<details><summary>Answers</summary>

1. `[4, 6, 32]` — `b` is added to every token vector.
2. `[B, T, T]` — one score for every pair of tokens. This is exactly the attention score matrix (lesson 16).
</details>

<details><summary>Going deeper: <code>view</code>, <code>reshape</code>, <code>contiguous</code></summary>

`view` changes the shape without copying data, but only works when the data is laid out
in memory in the right order ("contiguous"). After `permute` / `transpose` it usually isn't,
so you call `.contiguous()` first, or use `reshape` which copies only when needed.

```python
z = torch.arange(6).view(2, 3)
print(z.T.is_contiguous())          # False
print(z.T.contiguous().view(6))     # works
print(z.T.reshape(6))               # same result
```
</details>

---
[Course map](../README.md) · Prev: [00](00_how_to_use.md) · Next: [02 · Autograd](02_autograd.md)
