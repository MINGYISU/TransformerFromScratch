# from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .function import (
    linear, 
    layer_norm, 
    mha, 
    gelu, 
)

# class Module(ABC):
#     def __call__(self, x, *args, **kwargs):
#         self.forward(x, *args, **kwargs)

#     @abstractmethod
#     def forward(self, x, *args, **kwargs): pass

#     @abstractmethod
#     def parameters(self): pass

class Linear(nn.Module):
    """
    """
    in_features: int
    out_features: int
    W: nn.Parameter
    b: nn.Parameter | None

    def __init__(
            self,
            in_features: int, 
            out_features: int, 
            bias: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return linear(x, Weight=self.W, bias=self.b)
    
    def parameters(self):
        return [self.W, self.b] if self.b is not None else [self.W]
    
class LayerNorm(nn.Module):
    shape: int
    gamma: nn.Parameter
    beta: nn.Parameter
    eps: float
    
    def __init__(
            self,
            shape, 
            eps: float=1e-5):
        super().__init__()
        self.shape = shape
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, gamma=self.gamma, beta=self.beta, eps=self.eps)
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, f'Embed Size {embed_dim} cannot be divided by # of heads {num_heads} evenly!'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = Linear(embed_dim, embed_dim*3)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x):
        return mha(x, 
                     W_qkv=self.qkv_proj.W, 
                     b_qkv=self.qkv_proj.b, 
                     W_out=self.out_proj.W, 
                     b_out=self.out_proj.b, 
                     num_heads=self.num_heads)
    
    def parameters(self):
        return self.qkv_proj.parameters() + self.out_proj.parameters()
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        return x
    
    # FF implementation option 2
    # def forward(self, x):
    #     return F.ffn(x, 
    #                  W1=self.fc1.W, b1=self.fc1.b, 
    #                  W2=self.fc2.W, b2=self.fc2.b)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim=embed_dim, hidden_dim=int(embed_dim*mlp_ratio))

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x
    
    def parameters(self):
        return self.ln1.parameters() + \
                self.attn.parameters() + \
                self.ln2.parameters() + \
                self.mlp.parameters()