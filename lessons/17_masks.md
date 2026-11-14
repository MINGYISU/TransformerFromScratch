# 17 · Masks

**Level: 🟢 Basic**

> In one sentence: a mask hides some tokens from attention by setting their score to `-inf`,
> so softmax gives them weight 0.

## The idea

**Causal mask.** GPT learns by predicting the next token. If token 2 could look at token 3
while predicting token 3, that would be cheating. So each token may only look at itself
and the tokens **before** it:

```
          looks at:  t0  t1  t2  t3
token 0              ✓   ✗   ✗   ✗
token 1              ✓   ✓   ✗   ✗
token 2              ✓   ✓   ✓   ✗
token 3              ✓   ✓   ✓   ✓
```

The ✗ cells are the upper triangle. We set those scores to $-\infty$; since $e^{-\infty} = 0$, after
softmax they get exactly 0 weight:

```math
\text{score}_{ij} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt{d}} & \text{if } j \le i \quad \text{(past or itself)} \cr
-\infty & \text{if } j > i \quad \text{(future)}
\end{cases}
```

## Code

From `mha` in [`modules/function.py`](../modules/function.py):

```python
import torch
from modules.function import softmax

T = 4
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True = hidden
print(causal_mask)
```

## Try it

```python
scores = torch.zeros(T, T)                          # pretend all tokens match equally
weights = softmax(scores.masked_fill(causal_mask, -float('inf')))
print(weights)
# [[1.00, 0.00, 0.00, 0.00],
#  [0.50, 0.50, 0.00, 0.00],
#  [0.33, 0.33, 0.33, 0.00],
#  [0.25, 0.25, 0.25, 0.25]]
```

## What about padding?

You might expect a second mask to hide padding tokens. With **right padding** we don't need one:
padding comes *after* the real tokens, and the causal mask already stops real tokens from
looking forward. The padding positions do produce garbage outputs, but the loss ignores them
(`ignore_index=0`, lesson 23). Let's prove that padding doesn't change the real tokens:

```python
from modules.module import MultiHeadSelfAttention

torch.manual_seed(0)
attn = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
real = torch.randn(1, 3, 8)                                  # 3 real tokens
padded = torch.cat([real, torch.randn(1, 2, 8)], dim=1)      # + 2 padding tokens at the end
print(torch.allclose(attn(real), attn(padded)[:, :3], atol=1e-6))   # True
```

## Check yourself

1. Would we need a padding mask if we padded on the **left** instead?

<details><summary>Answer</summary>

Yes. Then padding comes *before* the real tokens, and the causal mask would let real tokens look at it.
</details>

<details><summary>Going deeper: why <code>-inf</code> and not just 0?</summary>

Setting a *score* to 0 doesn't hide it — $e^0 = 1$ still gets weight. We must make the
*weight* 0, and the only score that does that is $-\infty$. (Careful: if a whole row is `-inf`,
softmax returns `nan`. The causal mask never does that, since every token can see itself.)
</details>

---
[Course map](../README.md) · Prev: [16](16_attention.md) · Next: [18 · Multi-head attention](18_multi_head_attention.md)

🟢 First read? Skip ahead to [22 · Assembling GPT](22_assemble_gpt.md).
