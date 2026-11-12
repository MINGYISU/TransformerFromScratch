# 18 · Multi-head attention

> In one sentence: run several small attentions side by side, each free to look for something
> different, then glue their results back together.

## The idea

One attention gives each token **one** weighted average. But a token may want several things at
once — e.g. "the previous letter" *and* "the first letter of the word". So we cut each
$E$-dim Q, K, V into $H$ pieces ("heads") of size $D_h = E / H$, run attention on each piece
separately, and concatenate the results:

```math
\text{MultiHead}(x) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_{\text{out}},
\qquad
\text{head}_h = \text{Attention}(Q_h, K_h, V_h)
```

It costs the same as one big attention — it's just the same numbers split differently.

## Code

All of `mha` from [`modules/function.py`](../modules/function.py) is shape bookkeeping
around the `attention` function from lesson 16. Follow the shapes:

```python
import torch
from modules.function import linear, attention

def mha(x, W_qkv, b_qkv, W_out, b_out, num_heads):
    B, T, E = x.shape
    Dh = E // num_heads

    Q, K, V = linear(x, W_qkv, b_qkv).chunk(3, dim=-1)        # 3 x [B, T, E]   (one linear makes all three)

    # [B, T, E] -> [B, T, H, Dh] -> [B, H, T, Dh] -> [B*H, T, Dh]  : every head becomes its own "batch item"
    Q, K, V = (a.view(B, T, num_heads, Dh).transpose(1, 2).reshape(B * num_heads, T, Dh) for a in (Q, K, V))

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    out = attention(Q, K, V, mask=mask)                       # [B*H, T, Dh]

    # undo the split: [B*H, T, Dh] -> [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, E]
    out = out.view(B, num_heads, T, Dh).transpose(1, 2).reshape(B, T, E)
    return linear(out, W_out, b_out)                          # mix the heads together
```

## Try it

```python
from modules.function import mha as repo_mha

B, T, E, H = 2, 5, 8, 2
x = torch.randn(B, T, E)
W_qkv, b_qkv = torch.randn(3 * E, E), torch.zeros(3 * E)
W_out, b_out = torch.randn(E, E), torch.zeros(E)

out = mha(x, W_qkv, b_qkv, W_out, b_out, num_heads=H)
print(out.shape)                                                                         # [2, 5, 8]
print(torch.allclose(out, repo_mha(x, W_qkv, b_qkv, W_out, b_out, H), atol=1e-5))        # True
```

The class [`MultiHeadSelfAttention`](../modules/module.py) just owns `W_qkv`/`W_out` as two `Linear` layers and calls `mha`.

## Check yourself

1. $E = 32$, $H = 2$. What is $D_h$? What happens with $H = 3$?

<details><summary>Answer</summary>

$D_h = 32 / 2 = 16$. With $H = 3$, 32 can't be split evenly — the class raises an error (`embed_dim % num_heads == 0` is required).
</details>

<details><summary>Going deeper: why the output projection <code>W_out</code>?</summary>

After concatenation, the first `Dh` numbers came only from head 1, the next from head 2, and so on.
`W_out` is a linear layer that lets every output number combine information from all heads.
</details>

---
[Course map](../README.md) · Prev: [17](17_masks.md) · Next: [19 · LayerNorm](19_layer_norm.md)
