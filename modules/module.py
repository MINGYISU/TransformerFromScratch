from abc import ABC, abstractmethod
import os
import json
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

class Parameter(nn.Parameter):
    """A subclass of torch.nn.Parameter to represent parameters in our custom Module.
    This is mainly for clarity and potential future extensions.
    """
    def configs(self):
        return {'shape': self.shape, 'requires_grad': self.requires_grad}

class Module(ABC):
    """This is an abstract class used as a container to run the forward process when being called, automatically tracking parameters, and saving/loading the module. 
    It is a simplified version of torch.nn.Module for educational purposes.
    Use torch.nn.Module in real production for more robust and efficient implementation. 
    """
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)
    
    def class_name(self):
        return self.__class__.__name__
    
    @abstractmethod
    def _param_keys(self): raise NotImplementedError

    @abstractmethod
    def _config_keys(self): 
        return self._param_keys()

    def configs(self): 
        return {name: (ref.configs() if isinstance(ref, (Parameter, Module)) else ref)
                for name, ref in self._get_attrs_reference(key_type='config')}

    def _get_attrs_reference(self, key_type='param'):
        """Return a dict of direct references to the param attributes listed in _param_keys().
        """
        # return {key: getattr(self, key) for key in self._param_keys()}.items()
        keys = self._param_keys() if key_type == 'param' else self._config_keys()
        for key in keys:
            yield key, getattr(self, key)

    @abstractmethod
    def forward(self, x, *args, **kwargs): raise NotImplementedError
    
    def named_parameters(self):
        """Return a generator of all parameters in the module and its sub-modules.
        Please override named_parameters() method for complex cases."""
        for name, ref in self._get_attrs_reference():
            if isinstance(ref, Parameter):
                yield name, ref
            elif isinstance(ref, Module):
                for k, v in ref.named_parameters():
                    yield f'{name}.{k}', v

    def parameters(self): 
        """Return a generator of all parameters in the module and its sub-modules."""
        for _, ref in self.named_parameters():
            yield ref
    
    def save(self, folder_path, file_name: str=None):
        """Save the module to a file using pickle.
        """
        os.makedirs(folder_path, exist_ok=True)
        file_name = file_name if file_name is not None else f'{self.class_name()}.pkl'
        with open(os.path.join(folder_path, file_name), 'wb') as f:
            pickle.dump(self, f)
        with open(os.path.join(folder_path, f'{self.class_name()}_config.json'), 'w') as f:
            json.dump(self.configs(), f, indent=4)

    @classmethod
    def load(cls, file_path):
        """Load the module from a file using pickle.
        """
        with open(file_path, 'rb') as f:
            loaded_module = pickle.load(f)
        return loaded_module

class ModuleList(Module):
    """This is a simple implementation of ModuleList that can hold a list of Modules and
    run them sequentially in the forward pass.
    It is similar to torch.nn.ModuleList, but with a forward method.
    It is possible, but not recommended to use ModuleList to hold Modules of different types.
    Please initialize separate ModuleLists for different types of Modules for better readability.
    """
    modules: list[Module]

    def __init__(self, modules: list[Module], *args, **kwargs):
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
    
    def _config_keys(self):
        return super()._config_keys()

    def configs(self):
        params = {'length': len(self.modules)}
        for i, module in enumerate(self.modules):
            params.update({f'{module.class_name()}_{i}': module.configs()})
        return params
    
    def _param_keys(self):
        return [f'module_{i}' for i in range(len(self.modules))]
    
    def _get_attrs_reference(self):
        return {f'{m.class_name()}_{i}': m for i, m in enumerate(self.modules, start=1)}.items()
    
    def named_parameters(self):
        for i, module in enumerate(self.modules, start=1):
            for sub_mod_name, sub_mod_ref in module.named_parameters():
                yield f'{module.class_name()}_{i}.{sub_mod_name}', sub_mod_ref

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

class Linear(Module):
    """Linear layer wrapper
    """
    in_features: int
    out_features: int
    W: Parameter
    b: Parameter | None

    def __init__(
            self,
            in_features: int, 
            out_features: int, 
            bias: bool=True, 
            *args, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(torch.randn(out_features, in_features))
        self.b = Parameter(torch.zeros(out_features)) if bias else None

    def _param_keys(self):
        return ('W', 'b') if self.b is not None else ('W',)
    
    def _config_keys(self):
        return ('in_features', 'out_features') + self._param_keys()

    def forward(self, x):
        return linear(x, weight=self.W, bias=self.b)
    
class LayerNorm(Module):
    """Wrapper for Layer Normalization
    """
    shape: int
    gamma: Parameter
    beta: Parameter
    eps: float
    
    def __init__(
            self,
            shape, 
            eps: float=1e-5, 
            *args, **kwargs):
        super().__init__()
        self.shape = shape
        self.gamma = Parameter(torch.ones(shape))
        self.beta = Parameter(torch.zeros(shape))
        self.eps = eps

    def _param_keys(self):
        return 'gamma', 'beta'
    
    def _config_keys(self):
        return ('shape', 'eps') + self._param_keys()

    def forward(self, x):
        return layer_norm(x, gamma=self.gamma, beta=self.beta, eps=self.eps)
    
class MultiHeadSelfAttention(Module):
    """
    Args:
        ...
        padding_mask: torch.Tensor, mask to apply on the attention scores, if provided. 
                       This is typically used to mask out padding tokens in the input.
    """
    embed_dim: int
    num_heads: int
    qkv_proj: Linear
    out_proj: Linear
    padding_mask: torch.Tensor | None

    def __init__(self, embed_dim, num_heads, padding_mask=None, *args, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, f'Embed Size {embed_dim} cannot be divided by # of heads {num_heads} evenly!'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = Linear(embed_dim, embed_dim*3)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.padding_mask = padding_mask

    def _param_keys(self):
        return 'qkv_proj', 'out_proj'
    
    def _config_keys(self):
        return ('embed_dim', 'num_heads') + self._param_keys()

    def forward(self, x):
        return mha(x, 
                     W_qkv=self.qkv_proj.W, 
                     b_qkv=self.qkv_proj.b, 
                     W_out=self.out_proj.W, 
                     b_out=self.out_proj.b, 
                     num_heads=self.num_heads)
    
class FeedForward(Module):
    """Wrapper for feed forward nn
    """
    embed_dim: int
    hidden_dim: int
    fc1: Linear
    fc2: Linear

    def __init__(self, embed_dim, hidden_dim, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def _param_keys(self):
        return 'fc1', 'fc2'
    
    def _config_keys(self):
        return ('embed_dim', 'hidden_dim') + self._param_keys()

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        return x
    
class TransformerBlock(Module):
    """A complete transformer (decoder-only) block."""
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    ln1: LayerNorm
    attn: MultiHeadSelfAttention
    ln2: LayerNorm
    mlp: FeedForward

    def __init__(self, embed_dim, num_heads, *args, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim=embed_dim, hidden_dim=int(embed_dim*mlp_ratio))

    def forward(self, x):
        attn_out = self.attn(self.ln1(x)) # attention score
        x = x + attn_out # residual connection

        mlp_out = self.mlp(self.ln2(x)) # feed-forward output
        x = x + mlp_out # residual connection
        return x
    
    def _param_keys(self):
        return 'ln1', 'attn', 'ln2', 'mlp'
    
    def _config_keys(self):
        return ('embed_dim', 'num_heads', 'mlp_ratio') + self._param_keys()
    
