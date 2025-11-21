import torch

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2, dtype=x.dtype) / torch.pi) * (x + 0.044715 * x**3)))

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
    variance = torch.var(x, dim=-1, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(variance + eps) + beta

def linear(
        x: torch.Tensor, 
        Weight: torch.Tensor, 
        bias: torch.Tensor=None
) -> torch.Tensor:
    out = x @ Weight.T
    return out + bias if bias is not None else out

def ffn(
        x: torch.Tensor, 
        W1: torch.Tensor, b1: torch.Tensor, 
        W2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    return linear(gelu(linear(x, Weight=W1, bias=b1)), Weight=W2, bias=b2)

def attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor=None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    # raw query attention scores (what to attend)
    scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    scores = scores.masked_fill(mask, -1e-10) if mask is not None else scores

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
    N, embed_size = x.shape
    head_dim = embed_size // num_heads

    # qkv projection, [N, embed size] -> [N, 3*embed size]
    x = linear(x, Weight=W_qkv, bias=b_qkv)

    # split [N, 3*embed size] into qkv, each with size [N, embed size]
    # [N, 3*embed size] -> 3 x [N, embed size], corresponding to q, k, v
    Q, K, V = torch.chunk(x, 3, dim=-1)

    # split into multiple heads
    Q, K, V = (a.view(N, num_heads, head_dim) for a in (Q, K, V))

    # prevent presecding tokens attending to tokens afterwards 
    causal_mask = (torch.tril(torch.ones(N, N)) == 0)

    # compute attn scores for each head and concat the embedings back to original [N, embed size]
    out_heads = [attention(Q[:,i], K[:,i],V[:,i], mask=causal_mask) for i in range(num_heads)]
    x = torch.cat(out_heads, dim=-1)

    # output projection, this doesn't change the shape [N, embed size]
    return linear(x, Weight=W_out, bias=b_out)

def cross_entropy_loss(
        logits: torch.Tensor, 
        targets: torch.Tensor
):
    # log_probs = torch.log_softmax(logits, dim=-1)
    # loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # return loss.mean()
    batch_size, num_classes = logits.shape
    probs = softmax(logits)
    mapping = {k: v for v, k in enumerate(torch.unique(targets, sorted=True))}
    one_hot = torch.zeros(batch_size, num_classes)
    one_hot[torch.arange(batch_size), targets] = 1.0

    log_probs = torch.log(probs + 1e-10)
    loss = -(one_hot * log_probs).sum(dim=-1)

    loss = loss.mean()
    return loss