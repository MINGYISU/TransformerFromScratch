import torch
from .function import log_softmax

def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error: the average of (prediction - target)².
    0 means perfect; the further off, the bigger (see lessons/03_loss.md).
    """
    return ((prediction - target) ** 2).mean()

def cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int=None,
        label_smoothing: float=0.0,
) -> torch.Tensor:
    """Cross-entropy: the average of -log(probability given to the correct class).
    See lessons/09_cross_entropy.md.
    Args:
        logits: [N, C] raw scores, one row per example, one column per class
        targets: [N] index of the correct class for each example
        ignore_index: examples whose target equals this are skipped (e.g. padding)
        label_smoothing: 0.0 = all credit on the correct class; e.g. 0.1 = keep a little for every class
    Returns:
        a scalar tensor
    """
    log_probs = log_softmax(logits)                                   # [N, C]
    loss = -log_probs[torch.arange(len(targets)), targets]            # [N]: -log p(correct)
    if label_smoothing > 0.0:
        # mix in "-log p" averaged over ALL classes, so the model isn't pushed to 100% certainty
        loss = (1 - label_smoothing) * loss + label_smoothing * (-log_probs.mean(dim=-1))
    if ignore_index is not None:
        loss = loss[targets != ignore_index]
    return loss.mean()
