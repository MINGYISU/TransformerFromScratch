"""Metrics: numbers that tell US how good a model is (see lessons/10_metrics.md).
Unlike the loss, they don't need gradients and they are meant to be easy to read."""
import math
import torch

# ---------- predicting one class (e.g. the next token) ----------

def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=None) -> float:
    """Fraction of examples where the highest score is the correct class. logits [N, C], targets [N]."""
    correct = logits.argmax(dim=-1) == targets
    if ignore_index is not None:
        correct = correct[targets != ignore_index]
    return correct.float().mean().item()

def perplexity(cross_entropy_loss: float) -> float:
    """e^loss: 'the model is as unsure as if it were choosing uniformly among this many tokens'."""
    return math.exp(float(cross_entropy_loss))

# ---------- yes/no questions (binary classification) ----------

def precision(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Of everything we said YES to, how much was really yes?   TP / (TP + FP)"""
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    return tp / (tp + fp) if tp + fp > 0 else 0.0

def recall(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Of everything that was really yes, how much did we find?   TP / (TP + FN)"""
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    return tp / (tp + fn) if tp + fn > 0 else 0.0

def f1(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Harmonic mean of precision and recall: only high if BOTH are high."""
    p, r = precision(predictions, labels), recall(predictions, labels)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

# ---------- ranking (is the right answer near the top?) ----------

def hit_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fraction of examples whose correct class is among the k highest scores. logits [N, C], targets [N]."""
    top_k = logits.topk(k, dim=-1).indices                     # [N, k]
    return (top_k == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

def mrr(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Reciprocal Rank: average of 1/rank of the correct class (rank 1 = best)."""
    correct_score = logits[torch.arange(len(targets)), targets].unsqueeze(-1)   # [N, 1]
    rank = (logits > correct_score).sum(dim=-1) + 1                             # [N]
    return (1.0 / rank.float()).mean().item()

def ndcg_at_k(scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
    """Normalized Discounted Cumulative Gain for ONE ranked list.
    scores [C]: the model's score for each item; relevance [C]: how useful each item really is (0 = not at all).
    Items lower in the list count less (divided by log2(position + 1)); 1.0 = perfect order.
    """
    k = min(k, len(scores))
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))     # [1/log2(2), 1/log2(3), ...]
    dcg = (relevance[scores.argsort(descending=True)[:k]] * discount).sum()     # our order
    idcg = (relevance.sort(descending=True).values[:k] * discount).sum()        # best possible order
    return (dcg / idcg).item() if idcg > 0 else 0.0
