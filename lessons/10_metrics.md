# 10 · Metrics

> In one sentence: the loss is what the *model* learns from; metrics are what *we* read to
> decide whether the model is any good.

## The idea

The loss is built for gradient descent: smooth, always has a slope. But "the loss is 0.14" doesn't tell
a person much. A **metric** answers a human question directly, like "how often is it right?".
Metrics don't need slopes, because we never train on them — we only look at them.

Which metric fits depends on the question. Three common kinds:

| Question | Metrics |
|---|---|
| Is the answer right? | accuracy, perplexity |
| For yes/no questions: are the "yes" answers trustworthy, and did we find them all? | precision, recall, F1 |
| Is the right answer near the top of a ranked list? | hit@k, MRR, nDCG |

```python
import math
import torch
from modules import metrics
```

## 1. Accuracy and perplexity

**Accuracy**: the fraction of predictions where the highest score is the correct answer.

```math
\text{accuracy} = \frac{\text{number correct}}{\text{number of examples}}
```

**Perplexity** is how language models are usually compared. It turns the cross-entropy loss back into
"how many tokens is the model choosing between?":

```math
\text{perplexity} = e^{\text{loss}}
```

A model that guesses uniformly among 87 tokens has loss $\ln 87$, so perplexity $87$.
A perfect model has loss 0, so perplexity $e^0 = 1$: no doubt at all.

```python
logits = torch.tensor([[2., 1.], [0., 3.], [5., 1.], [1., 2.]])   # 4 examples, 2 classes
targets = torch.tensor([0, 1, 1, 1])
print(metrics.accuracy(logits, targets))     # 0.75 — the 3rd example is wrong
print(metrics.perplexity(math.log(87)))      # 87.0
print(metrics.perplexity(0.14))              # ≈ 1.15 — our trained GPT is almost sure
```

## 2. Precision, recall and F1

Take a spam filter tested on 10 emails. Compare its answers with the truth:

| | really spam | really not spam |
|---|---|---|
| **filter says spam** | 3 — true positive (TP) | 1 — false positive (FP) |
| **filter says not spam** | 2 — false negative (FN) | 4 — true negative |

- **Precision** — when it says "spam", how often is it right? $\frac{TP}{TP + FP} = \frac{3}{4} = 0.75$
- **Recall** — of all the real spam, how much did it catch? $\frac{TP}{TP + FN} = \frac{3}{5} = 0.6$
- **F1** — one number that is only high when *both* are high:

```math
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \frac{2 \cdot 0.75 \cdot 0.6}{0.75 + 0.6} \approx 0.667
```

Accuracy can lie here: if only 1 email in 100 is spam, a filter that *never* says spam is 99% accurate — and useless.
Its recall is 0, which shows the problem immediately.

```python
predictions = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])   # 1 = "spam"
labels      = torch.tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
print(metrics.precision(predictions, labels))   # 0.75
print(metrics.recall(predictions, labels))      # 0.6
print(metrics.f1(predictions, labels))          # 0.667
```

## 3. Ranking: hit@k, MRR, nDCG

Sometimes it's enough for the right answer to be **near the top** — a search engine, a recommender,
or an LLM where several next tokens are fine.

- **hit@k**: is the correct answer among the top $k$? Averaged over examples.
- **MRR** (mean reciprocal rank): if the correct answer is in position $r$, score $\frac{1}{r}$ — so 1st place gives 1, 2nd gives 0.5, 3rd gives 0.33.

```math
\text{MRR} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{r_n}
```

```python
scores = torch.tensor([[0.1, 0.9, 0.5],     # correct = item 2 -> ranked 2nd
                       [0.8, 0.1, 0.3]])    # correct = item 0 -> ranked 1st
correct = torch.tensor([2, 0])
print(metrics.hit_at_k(scores, correct, k=1))   # 0.5
print(metrics.hit_at_k(scores, correct, k=2))   # 1.0
print(metrics.mrr(scores, correct))             # (1/2 + 1/1) / 2 = 0.75
```

- **nDCG@k** handles answers that are *partly* right, with a relevance score $\text{rel}_i$ per item
  (e.g. 3 = perfect, 1 = so-so, 0 = useless). Each item's relevance is discounted by how low it's ranked,
  then we divide by the best possible score so that 1.0 = perfect order:

```math
\text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}, \qquad
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{DCG@}k \text{ of the ideal order}}
```

By hand: we show 3 items whose relevances, in our order, are $[1, 3, 0]$. The ideal order is $[3, 1, 0]$.

```math
\text{DCG} = \frac{1}{\log_2 2} + \frac{3}{\log_2 3} + 0 = 1 + 1.89 = 2.89, \qquad
\text{ideal} = \frac{3}{\log_2 2} + \frac{1}{\log_2 3} = 3 + 0.63 = 3.63, \qquad
\text{nDCG} = \frac{2.89}{3.63} \approx 0.80
```

```python
model_scores = torch.tensor([0.9, 0.5, 0.1])    # our ranking: item 0, then 1, then 2
relevance    = torch.tensor([1.0, 3.0, 0.0])    # but item 1 was the best one
print(metrics.ndcg_at_k(model_scores, relevance, k=3))   # ≈ 0.80
```

All of these live in [`modules/metrics.py`](../modules/metrics.py), each in a few lines.

## Check yourself

1. A model's validation loss is 2.0. What's its perplexity, and what does that mean?
2. A cancer screening test: is it worse to have low precision or low recall?

<details><summary>Answers</summary>

1. $e^{2} \approx 7.4$ — on average, it's as unsure as picking between about 7 equally likely tokens.
2. Usually low **recall** — missed cancers (false negatives) are worse than false alarms, which a second test can rule out.
</details>

<details><summary>Going deeper: why not train on accuracy directly?</summary>

Accuracy only changes when a prediction *flips* from wrong to right. Nudge a weight a tiny bit and
accuracy almost never changes, so its slope is 0 almost everywhere — gradient descent would have no idea which way to go.
Cross-entropy changes smoothly with every tiny nudge. So: train on the loss, report the metrics.
</details>

---
[Course map](../README.md) · Prev: [09](09_cross_entropy.md) · Next: [11 · Momentum and Adam](11_momentum_and_adam.md)
