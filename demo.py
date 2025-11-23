from modules.model import MyGPT
# from modules.function import cross_entropy_loss
import torch

def main():
    x = torch.randint(0, 100, (1, 4))
    targets = torch.randn(1, 4)
    model = MyGPT(vocab_size=100, max_seq_length=100, embed_dim=10, num_layers=2, num_heads=2)
    print(*(f'{k}: {v.shape}' for k, v in model.named_parameters()), sep='\n')
    logits = model(x)
    # loss = cross_entropy_loss(logits, targets)
    logits.mean().backward()
    # for param in model.parameters():
    #     print(param.grad.shape)

if __name__ == '__main__':
    main()
