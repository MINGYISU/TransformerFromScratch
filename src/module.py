from abc import ABC, abstractmethod
import warnings
import pickle
import torch
import torch.nn as nn
from .function import (
    linear, 
    layer_norm, 
    mha, 
    gelu, 
)

class Module(ABC):
    """This is an abstract class used as a container to run the forward process when being called. 
    It is for a simple demo only, and almost does nothing. 
    Use torch.nn.Module in real production for more robust and efficient implementation. 
    """
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)
    
    @abstractmethod
    def _param_keys(self): raise NotImplementedError

    def _get_attrs_reference(self):
        """Return a list of direct references to the attributes listed in keys().
        """
        return {key: getattr(self, key) for key in self._param_keys()}.items()

    def class_name(self):
        return self.__class__.__name__

    @abstractmethod
    def forward(self, x, *args, **kwargs): raise NotImplementedError
    
    def named_parameters(self):
        params = {}
        for name, ref in self._get_attrs_reference():
            if isinstance(ref, Module):
                params.update({f'{name}.{k}': v for k, v in ref.named_parameters().items()})
            elif isinstance(ref, nn.Parameter):
                params[name] = ref
        return params

    def parameters(self): 
        return self.named_parameters().values()
    
    def save(self, filepath: str=None):
        """Save the module to a file using pickle.
        """
        filepath = filepath if filepath is not None else f'{self.class_name()}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

class ModuleList(Module):
    """This is a simple implementation of ModuleList that can hold a list of Modules and
    run them sequentially in the forward pass.
    It is similar to torch.nn.ModuleList, but with a forward method.
    It is possible, but not recommended to use ModuleList to hold Modules of different types.
    Please initialize separate ModuleLists for different types of Modules for better readability.
    """
    modules: list[Module]

    def __init__(self, modules: list[Module]):
        super().__init__()
        assert all(isinstance(m, Module) for m in modules), "ModuleList can only hold Module instances."
        unique_types = set(type(m) for m in modules)
        if len(unique_types) > 1:
            warnings.warn(
                message="ModuleList contains modules of different types. "
                        "It is recommended to use separate ModuleLists for different types of modules for better readability.")
        self.modules = modules

    def class_name(self):
        elem_name = self.modules[0].class_name() if self.modules else "Module"
        return f'{self.__class__.__name__}[{elem_name}]'

    def _param_keys(self):
        return [f'module_{i}' for i in range(len(self.modules))]
    
    def _get_attrs_reference(self):
        return {f'{m.class_name()}_{i}': m for i, m in enumerate(self.modules, start=1)}.items()
    
    def named_parameters(self):
        params = {}
        for i, module in enumerate(self.modules, start=1):
            sub_params = module.named_parameters()
            params.update({f'{module.class_name()}_{i}.{k}': v for k, v in sub_params.items()})
        return params

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    # def parameters(self):
    #     params = []
    #     for module in self.modules:
    #         params.extend(module.parameters())
    #     return params

class Linear(Module):
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

    def _param_keys(self):
        return ('W', 'b') if self.b is not None else ('W',)

    def forward(self, x):
        return linear(x, Weight=self.W, bias=self.b)
    
    # def parameters(self):
    #     return [self.W, self.b] if self.b is not None else [self.W]
    
class LayerNorm(Module):
    """
    """
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

    def _param_keys(self):
        return 'gamma', 'beta'
    
    def forward(self, x):
        return layer_norm(x, gamma=self.gamma, beta=self.beta, eps=self.eps)
    
    # def parameters(self):
    #     return [self.gamma, self.beta]
    
class MultiHeadSelfAttention(Module):
    """
    """
    embed_dim: int
    num_heads: int
    qkv_proj: Linear
    out_proj: Linear

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, f'Embed Size {embed_dim} cannot be divided by # of heads {num_heads} evenly!'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = Linear(embed_dim, embed_dim*3)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _param_keys(self):
        return 'qkv_proj', 'out_proj'

    def forward(self, x):
        return mha(x, 
                     W_qkv=self.qkv_proj.W, 
                     b_qkv=self.qkv_proj.b, 
                     W_out=self.out_proj.W, 
                     b_out=self.out_proj.b, 
                     num_heads=self.num_heads)
    
    # def parameters(self):
    #     return self.qkv_proj.parameters() + self.out_proj.parameters()
    
class FeedForward(Module):
    """
    """
    fc1: Linear
    fc2: Linear

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def _param_keys(self):
        return 'fc1', 'fc2'

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        return x
    
    # def parameters(self):
    #     return self.fc1.parameters() + self.fc2.parameters()
    
class TransformerBlock(Module):
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
    
    def _param_keys(self):
        return 'ln1', 'attn', 'ln2', 'mlp'
    
    # def parameters(self):
    #     params = []
    #     for layer in self.keys():
    #         params.extend(getattr(self, layer).parameters())
    #     return params
