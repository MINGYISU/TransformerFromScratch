# 22 · Assembling GPT

**Level: 🟢 Basic**

> In one sentence: GPT = embeddings → a stack of transformer blocks → final LayerNorm →
> a score for every word in the vocabulary.

## The idea

```
ids            [B, T]
 │ token_embed + pos_embed          (lessons 14, 15)
x              [B, T, E]
 │ block × N                        (lesson 21)
 │ final LayerNorm                  (lesson 19)
x              [B, T, E]
 │ x @ token_embedᵀ                 (score each vocabulary token)
logits         [B, T, V]
```

> **Coming from lesson 17 on the 🟢 path?** You skipped the details of the block, which is fine. A transformer block is
> *attention* (tokens look at earlier tokens, [lesson 16](16_attention.md)) followed by a *feed-forward* network (each token
> thinks on its own), and LayerNorm just keeps the numbers in a healthy range. Stacking blocks never changes the shape `[B, T, E]`.

`logits[b, t]` is a list of `V` scores: "how likely is each token to come **after** position `t`".
[Softmax](08_softmax.md) turns them into probabilities:

```math
P(\text{next token} = i) = \text{softmax}(\text{logits})_i
```

## Code

`MyGPT` in [`models/example/model.py`](../models/example/model.py), the important part:

```python
import torch
from modules.module import ModuleList, TransformerBlock, LayerNorm

class TinyGPT:
    def __init__(self, V, E, N, H, max_len):
        self.token_embed = torch.randn(V, E) * 0.02
        self.pos_embed = torch.randn(max_len, E) * 0.02
        self.blocks = ModuleList([TransformerBlock(E, H) for _ in range(N)])
        self.ln_f = LayerNorm(E)

    def __call__(self, ids):
        T = ids.shape[1]
        x = self.token_embed[ids] + self.pos_embed[:T]   # [B, T, E]
        x = self.blocks(x)                               # [B, T, E]
        x = self.ln_f(x)
        return x @ self.token_embed.T                    # [B, T, V]
```

## Try it

```python
from models.example.model import MyGPT

model = MyGPT(max_seq_length=100, embed_dim=32, num_layers=2, num_heads=2)
ids, _ = model.tokenizer.tokenize(['apple', 'fig'])
logits = model(ids)
print(ids.shape, '->', logits.shape)                          # [2, 6] -> [2, 6, 87]
print(sum(p.numel() for p in model.parameters()), 'parameters')
```

An untrained model is guessing: every token gets roughly the same probability, about $1/87 \approx 0.011$.

```python
from modules.function import softmax
print(softmax(logits[0, 0]).max().item())    # ≈ 0.016 — no token stands out
```

## Check yourself

1. The last step multiplies by `token_embed.T` — the same table we used at the input. What does each logit measure?

<details><summary>Answer</summary>

The dot product between the model's output vector and each token's embedding — "how similar is my
prediction to token `i`?". The most similar token gets the highest score.
</details>

<details><summary>Going deeper: weight tying</summary>

Using the input embedding table again as the output layer is called **weight tying**
(GPT-2 does it too). It saves $V \times E$ parameters and makes sense: a token should be predicted when
the output looks like that token's own embedding. Many larger models use a separate output matrix instead.
</details>

---
[Course map](../README.md) · Prev: [21](21_transformer_block.md) · Next: [23 · Next-token training](23_next_token_training.md)
