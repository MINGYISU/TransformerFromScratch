# 24 · Generating text

> In one sentence: to write text, ask the model for the next token, add it to the input, and repeat.

## The idea

The model gives a prediction at every position, but for writing we only need the **last** one
(it has seen the whole text so far):

```
'ap'      -> model -> probabilities for next token -> pick 'p'
'app'     -> model -> ... -> pick 'l'
'appl'    -> model -> ... -> pick 'e'
'apple'   -> model -> ... -> pick <EOS>  -> stop
```

This is called **autoregressive** generation. How do we "pick"? We **sample** from the
probabilities: a token with probability 0.7 is picked 70% of the time. This makes the output varied.

## Code

`MyGPT.generate` in [`models/example/model.py`](../models/example/model.py), simplified to one text:

```python
import torch
from modules.function import softmax

def generate(model, ids, max_new_tokens=20, eos_token_id=None):
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]                     # [1, T, V] -> last position [1, V]
        probs = softmax(logits)                           # scores -> probabilities
        next_id = torch.multinomial(probs, num_samples=1) # sample one token   [1, 1]
        ids = torch.cat([ids, next_id], dim=-1)           # append and repeat
        if next_id.item() == eos_token_id:
            break
    return ids
```

## Try it

Train a small model for a few seconds (not saved), then let it spell fruit names:

```python
from models.example.train import train

torch.manual_seed(0)
model = train(steps=600, save_dir=None)
tok = model.tokenizer
eos = tok.vocabulary['<EOS>']

for prompt in ['ba', 'ch', 'gr']:
    ids, _ = tok.tokenize([prompt], add_eos=False)    # no <EOS> on a prompt!
    print(prompt, '->', tok.decode(generate(model, ids, eos_token_id=eos).tolist())[0])
```

## Check yourself

1. Our model re-reads the whole text for every new token. Why is that wasteful?

<details><summary>Answer</summary>

The keys and values of earlier tokens never change (the causal mask means they don't see new tokens),
yet we recompute them each step. Real systems cache them — the "KV cache".
</details>

<details><summary>Going deeper: temperature, greedy, top-k, top-p</summary>

All four just reshape the probabilities before sampling. **Temperature** $t$ divides the logits $z$
before softmax:

```math
p_i = \frac{e^{z_i / t}}{\sum_j e^{z_j / t}}
```

```python
logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])

# temperature: divide logits by t.  t < 1 = more confident, t > 1 = more random
print(softmax(logits / 0.5), softmax(logits / 2.0))

# greedy: always take the most likely token (temperature -> 0). Deterministic, but repetitive.
print(logits.argmax(dim=-1))

# top-k: keep only the k most likely tokens
k = 2
kth_best = logits.topk(k).values[:, -1:]
print(softmax(logits.masked_fill(logits < kth_best, -float('inf'))))

# top-p (nucleus): keep the smallest set of top tokens whose probabilities add up to p
p = 0.9
probs = softmax(logits)
sorted_probs, order = probs.sort(descending=True)
drop = sorted_probs.cumsum(-1) - sorted_probs > p        # tokens after we already reached p
sorted_probs[drop] = 0
print(torch.zeros_like(probs).scatter(-1, order, sorted_probs / sorted_probs.sum()))
```
</details>

---
[Course map](../README.md) · Prev: [23](23_next_token_training.md)
