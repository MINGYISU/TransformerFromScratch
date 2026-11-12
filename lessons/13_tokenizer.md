# 13 · Tokenizer

> In one sentence: a tokenizer turns text into a list of integer ids (and back), because models only understand numbers.

## The idea

The simplest tokenizer: **one character = one token**. We give every character a number:

```
'a' -> 1, 'b' -> 2, ..., 'apple' -> [1, 16, 16, 12, 5]
```

The full list of known tokens is the **vocabulary**. We add two special tokens:

- `<unk>` (id 0) for characters we don't know — we also reuse 0 as **padding** (see below)
- `<EOS>` ("end of sequence") — the model learns to emit it when the word is done

**Padding**: texts in a batch have different lengths, but a tensor must be a rectangle.
So we add 0s at the end of the short ones, plus a mask saying which tokens are real.

## Code

[`models/example/tokenizer.py`](../models/example/tokenizer.py), shortened:

```python
import string
import torch

vocabulary = {ch: i for i, ch in enumerate(string.ascii_letters + string.punctuation + ' ', start=1)}
vocabulary['<unk>'] = 0
vocabulary['<EOS>'] = len(vocabulary)

def tokenize(texts, add_eos=True):
    tokenized = []
    for text in texts:
        tokens = [vocabulary.get(ch, 0) for ch in text]
        if add_eos:
            tokens.append(vocabulary['<EOS>'])
        tokenized.append(tokens)
    max_len = max(len(t) for t in tokenized)
    padded = [t + [0] * (max_len - len(t)) for t in tokenized]           # pad on the right
    mask = [[1] * len(t) + [0] * (max_len - len(t)) for t in tokenized]  # 1 = real token
    return torch.tensor(padded), torch.tensor(mask)
```

## Try it

```python
from models.example.tokenizer import MyGptTokenizer

tok = MyGptTokenizer()
print(tok.vocab_size())                 # 87
ids, mask = tok.tokenize(['apple', 'fig'])
print(ids)    # [[ 1, 16, 16, 12,  5, 86],
              #  [ 6,  9,  7, 86,  0,  0]]   <- 86 = <EOS>, 0 = padding
print(mask)
print(tok.decode(ids.tolist()))         # ['apple', 'fig']
```

## Check yourself

1. Why do we add `<EOS>` for training data but not for a prompt we want the model to continue?

<details><summary>Answer</summary>

`<EOS>` means "this text is finished". During training, the model must learn *when* to stop,
so it needs to see `<EOS>`. A prompt like `'ap'` is unfinished — adding `<EOS>` would tell the model the word is already done.
</details>

<details><summary>Going deeper: subword tokenizers (BPE)</summary>

Character tokens make sequences long. Real GPTs use **Byte Pair Encoding**: start with
characters, then repeatedly merge the most frequent neighbouring pair into a new token.

```python
from collections import Counter

words = [list('low'), list('lower'), list('lowest')]
pairs = Counter(p for w in words for p in zip(w, w[1:]))
best = max(pairs, key=pairs.get)
print(best)    # ('l', 'o') — appears 3 times, becomes the new token 'lo'
```

Repeat that ~50,000 times and you get GPT-2's vocabulary, where common words are a single token.
</details>

---
[Course map](../README.md) · Prev: [12](12_training_tricks.md) · Next: [14 · Token embeddings](14_token_embeddings.md)
