# 08 · Softmax

> In one sentence: softmax turns any list of scores into probabilities — all positive, summing to 1,
> and bigger scores get bigger probabilities.

## Intuition

A model outputs raw scores ("logits") like `[2.0, 1.0, 0.1]`. To read them as
"how likely is each option", we need:

1. all values positive → apply `exp`
2. all values sum to 1 → divide by the total

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

By hand, for `[2.0, 1.0, 0.1]`:

```
exp:      [7.39, 2.72, 1.11]      total = 11.21
divide:   [7.39/11.21, 2.72/11.21, 1.11/11.21]  =  [0.66, 0.24, 0.10]
```

GPT uses softmax twice: to turn attention scores into weights (lesson 16), and to turn the
final logits into "probability of each next token" (lesson 24).

## Code

From [`modules/function.py`](../modules/function.py):

```python
import torch

def softmax(x):
    """softmax along the last dimension"""
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)
```

Ignore the `- max` for now (see Going deeper) — it doesn't change the result.

## Try it

```python
p = softmax(torch.tensor([2.0, 1.0, 0.1]))
print(p)          # ≈ [0.659, 0.242, 0.099]
print(p.sum())    # 1.0
```

Adding the same number to all scores changes nothing — only **differences** between scores matter:

```python
print(softmax(torch.tensor([102.0, 101.0, 100.1])))   # same as above
```

## Check yourself

1. What does softmax return for `[5.0, 5.0, 5.0, 5.0]`?

<details><summary>Answer</summary>

`[0.25, 0.25, 0.25, 0.25]` — equal scores mean equal probabilities.
</details>

<details><summary>Going deeper: why subtract the max?</summary>

`exp(1000)` is too big for a float and becomes `inf`, and `inf / inf = nan`.
Since subtracting the same number from every score doesn't change the answer
(shown above), we subtract the largest one. Now the biggest exponent is `exp(0) = 1` — no overflow.

```python
big = torch.tensor([1000.0, 1001.0])
naive = torch.exp(big) / torch.exp(big).sum()
print(naive)          # tensor([nan, nan])
print(softmax(big))   # tensor([0.2689, 0.7311])
```
</details>

---
[Course map](../README.md) · Prev: [06](06_activations.md) · Next: [13 · Tokenizer](13_tokenizer.md)
*(lessons 09–12 are coming soon)*
