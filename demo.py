from src.module import TransformerBlock
import torch

def main():
    x = torch.randn(4, 10)
    model = TransformerBlock(10, 2)
    y = model(x)
    y.mean().backward()
    for param in model.parameters():
        print(param.grad.shape)

if __name__ == '__main__':
    main()
