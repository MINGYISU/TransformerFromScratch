import torch

def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error: the average of (prediction - target)².
    0 means perfect; the further off, the bigger (see lessons/03_loss.md).
    """
    return ((prediction - target) ** 2).mean()
