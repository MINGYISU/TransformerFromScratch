import torch
import math

def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def softmax(x: torch.Tensor) -> torch.Tensor:
    """softmax()
    """
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)

def layer_norm(
        x: torch.Tensor, 
        gamma: torch.Tensor, 
        beta: torch.Tensor, 
        eps: float=1e-5
) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    # use reciprocal sqrt for numerical stability and clarity
    inv_std = torch.rsqrt(var + eps)
    return gamma * (x - mean) * inv_std + beta

def linear(
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor=None
) -> torch.Tensor:
    out = x @ weight.T
    return out + bias if bias is not None else out

def ffn(
        x: torch.Tensor, 
        W1: torch.Tensor, b1: torch.Tensor, 
        W2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    return linear(gelu(linear(x, weight=W1, bias=b1)), weight=W2, bias=b2)

def attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor=None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    # raw query attention scores (what to attend)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores.masked_fill(mask, -float('inf')) if mask is not None else scores

    # apply softmax to form a prob distribution
    weights = softmax(scores)

    # compute (result of attention)
    output = weights @ V
    return output

def mha(
        x: torch.Tensor, 
        W_qkv: torch.Tensor, b_qkv: torch.Tensor, 
        W_out: torch.Tensor, b_out: torch.Tensor, 
        num_heads: int
) -> torch.Tensor:
    """Multi-Head Self-Attention
    Terminologies:
        B: batch size
        T: sequence length
        E: embedding dimension
        H: number of heads
        Dh: dimension per head = E / num_heads"""
    B, T, E = x.shape # [B, T, E]
    head_dim = E // num_heads

    # qkv projection, [B, T, E] -> [B, T, 3*E]
    x = linear(x, weight=W_qkv, bias=b_qkv)

    # split [B, T, 3*E] into qkv, each with size [B, T, E]
    # [B, T, 3*E] -> 3 x [B, T, E], corresponding to q, k, v
    Q, K, V = torch.chunk(x, 3, dim=-1)

    # split into multiple heads
    # [B, T, E] -> [B, T, H, Dh]
    Q, K, V = (a.view(B, T, num_heads, head_dim) for a in (Q, K, V))

    # prevent preceding tokens attending to tokens afterwards 
    causal_mask = (torch.tril(torch.ones(T, T)) == 0).to(x.device)  # [T, T]

    # Purpose: compute attn scores for all heads at once (vectorized) and concat embeddings back to original [T, E] size
        # view(B, T, H, Dh): reshape QKV of shape [B, T, E (== H*Dh)] -> [B, T, H, Dh]
        # permute(0, 2, 1, 3): change to [B, H, T, Dh]
        # contiguous(): make sure the tensor is stored in a contiguous chunk of memory (for efficiency)
        # view(B * num_heads, T, head_dim): merge B and H dims -> [B*H, T, Dh], since attention function expects 3D tensors
        # Author's note: you can also keep B and H dims separate and modify line 49 from attention function to handle 4D tensors, but this is more efficient
    Q, K, V = (a.view(B, T, num_heads, head_dim)
               .permute(0, 2, 1, 3).contiguous()
               .view(B * num_heads, T, head_dim) 
               for a in (Q, K, V))

    # compute attention for all heads in one call
    x = attention(Q, K, V, mask=causal_mask)  # returns [B*num_heads, T, head_dim]

    # Purpose: reshape the output back to orig size, which is [B, T, E]
        # view(B, num_heads, T, head_dim): reshape [B*H, T, Dh] back to [B, H, T, Dh]
        # permute(0, 2, 1, 3): change to [B, T, H, Dh]
        # view(B, T, E): merge H and Dh dims back to E: [B, T, H, Dh] -> [B, T, E (== H*Dh)]
    x = x.view(B, num_heads, T, head_dim) \
            .permute(0, 2, 1, 3).contiguous() \
            .view(B, T, E)

    # final output projection
    return linear(x, weight=W_out, bias=b_out)
