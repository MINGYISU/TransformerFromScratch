# 21 · The transformer block

> In one sentence: a transformer block is "talk to other tokens" (attention) followed by
> "think alone" (feed-forward), each wrapped in LayerNorm + residual.

## Intuition

We now have every part. One block is just these four lines:

```
x = x + attention(layer_norm(x))      # tokens exchange information
x = x + feed_forward(layer_norm(x))   # each token processes what it gathered
```

GPT stacks the same block `N` times (GPT-2 small: 12, our toy model: 2). Every block has its
own weights, and every block keeps the shape `[B, T, E]`.

## Code

`TransformerBlock` from [`modules/module.py`](../modules/module.py):

```python
import torch
from modules.module import Module, LayerNorm, MultiHeadSelfAttention, FeedForward

class MyBlock(Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio))

    def _param_keys(self): return 'ln1', 'attn', 'ln2', 'mlp'
    def _config_keys(self): return self._param_keys()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

## Try it

```python
block = MyBlock(embed_dim=8, num_heads=2)
x = torch.randn(2, 5, 8)
print(block(x).shape)                   # [2, 5, 8]

for name, p in block.named_parameters():
    print(f'{name:12} {tuple(p.shape)}')
print(sum(p.numel() for p in block.parameters()), 'parameters')   # 872
```

## Check yourself

1. Why can we stack as many blocks as we like?

<details><summary>Answer</summary>

Each block takes `[B, T, E]` and returns `[B, T, E]`, so the output of one fits straight into the next.
</details>

<details><summary>Going deeper: pre-norm vs post-norm</summary>

The original 2017 Transformer normalised **after** the residual: `x = norm(x + f(x))` ("post-norm").
GPT-2 moved it **before**: `x = x + f(norm(x))` ("pre-norm"). Pre-norm leaves the residual
path completely untouched, which makes deep models much easier to train. Almost all modern LLMs use pre-norm.
</details>

---
[Course map](../README.md) · Prev: [20](20_residual_and_feed_forward.md) · Next: [22 · Assembling GPT](22_assemble_gpt.md)
