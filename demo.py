from src.module import TransformerBlock
from src.function import cross_entropy_loss
import torch

def main():
    x = torch.randn(4, 10)
    targets = torch.randn(4, 1)
    model = TransformerBlock(10, 2)
    logits = model(x)
    # loss = cross_entropy_loss(logits, targets)
    logits.mean().backward()
    for param in model.parameters():
        print(param.grad.shape)

if __name__ == '__main__':
    main()
