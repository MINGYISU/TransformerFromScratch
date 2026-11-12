# 23 · Next-token training

> In one sentence: we train GPT by showing it text and asking, at **every** position,
> "what comes next?" — the target is just the input shifted by one.

## The idea

One word gives us many training examples at once:

```
word:     a   p   p   l   e   <EOS>
input:    a   p   p   l   e            ids[:, :-1]
target:   p   p   l   e  <EOS>         ids[:, 1:]
```

At position 0 the model sees `a` and should say `p`; at position 1 it sees `a p` and should say `p`;
... at the end it sees `apple` and should say `<EOS>`. The [causal mask](17_masks.md) makes sure
each position can't peek at its own answer.

The training loop is always the same four steps:

1. **forward**: `logits = model(input)` → `[B, T, V]`
2. **loss**: how wrong were we? ([loss](03_loss.md); for tokens we use cross-entropy, lesson 09)
3. **backward**: `loss.backward()` computes every gradient ([gradients](02_gradients.md))
4. **step**: nudge every weight against its gradient ([gradient descent](04_gradient_descent.md); we use its smarter cousin Adam, lesson 11)

## Code

From [`models/example/train.py`](../models/example/train.py):

```python
import random
import torch
import torch.nn as nn
from models.example.model import MyGPT
from models.example.train import FRUITS

def get_train_batch(tokenizer, batch_size=8):
    ids, _ = tokenizer.tokenize(random.sample(FRUITS, batch_size))   # [B, T+1]
    return ids[:, :-1], ids[:, 1:]                                   # input, target
```

> For now `train.py` borrows `cross_entropy` and `Adam` from PyTorch. Lessons 09 and 11
> (coming soon) replace them with our own.

## Try it

A short training run (a few seconds):

```python
torch.manual_seed(0); random.seed(0)
model = MyGPT(max_seq_length=100, embed_dim=32, num_layers=2, num_heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(301):
    inputs, targets = get_train_batch(model.tokenizer)
    logits = model(inputs)                                            # 1. forward   [B, T, V]
    loss = nn.functional.cross_entropy(                               # 2. loss
        logits.reshape(-1, logits.size(-1)),                          #    [B*T, V]
        targets.reshape(-1),                                          #    [B*T]
        ignore_index=0)                                               #    skip padding
    optimizer.zero_grad()
    loss.backward()                                                   # 3. backward
    optimizer.step()                                                  # 4. step
    if step % 100 == 0:
        print(step, round(loss.item(), 3))
```

The loss starts around $\ln(87) \approx 4.47$ (pure guessing among 87 tokens) and drops fast.

## Check yourself

1. Why can't the loss reach 0 on this data?

<details><summary>Answer</summary>

Some answers are truly uncertain: after `p` the next letter could be `e` (peach, pear), `l` (plum) or
`a` (papaya). The best the model can do is spread probability between them.
</details>

<details><summary>Going deeper: why flatten to <code>[B*T, V]</code>?</summary>

Cross-entropy expects a list of predictions and a list of correct answers. Every position of
every text is one independent "pick the right token out of V" question, so we lay them all out
in one long list of `B*T` questions and average the loss over the non-padding ones.
</details>

---
[Course map](../README.md) · Prev: [22](22_assemble_gpt.md) · Next: [24 · Generating text](24_generation.md)
