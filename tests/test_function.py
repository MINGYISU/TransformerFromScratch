"""Our from-scratch functions vs. PyTorch's built-ins (used here only as the answer key)."""
import torch
import torch.nn.functional as F
from modules.function import gelu, softmax, layer_norm, linear, attention, mha
from modules.module import Linear

torch.manual_seed(0)
x = torch.randn(2, 5, 8)

def test_gelu():
    assert torch.allclose(gelu(x), F.gelu(x), atol=1e-6)

def test_softmax():
    assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-6)

def test_softmax_is_stable_for_large_values():
    assert torch.allclose(softmax(torch.tensor([1000., 1001.])), F.softmax(torch.tensor([1000., 1001.]), dim=-1))

def test_layer_norm():
    g, b = torch.randn(8), torch.randn(8)
    assert torch.allclose(layer_norm(x, g, b), F.layer_norm(x, (8,), g, b), atol=1e-5)

def test_linear():
    W, b = torch.randn(3, 8), torch.randn(3)
    assert torch.allclose(linear(x, W, b), F.linear(x, W, b), atol=1e-5)

def test_causal_attention():
    Q, K, V = torch.randn(3, 4, 6, 8).unbind(0)
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert torch.allclose(attention(Q, K, V, mask=mask), expected, atol=1e-5)

def test_mha_matches_torch_multihead_attention():
    E, H = 8, 2
    W_qkv, b_qkv, W_out, b_out = torch.randn(3 * E, E), torch.randn(3 * E), torch.randn(E, E), torch.randn(E)
    ref = torch.nn.MultiheadAttention(E, H, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(W_qkv); ref.in_proj_bias.copy_(b_qkv)
        ref.out_proj.weight.copy_(W_out); ref.out_proj.bias.copy_(b_out)
    T = x.shape[1]
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    expected, _ = ref(x, x, x, attn_mask=causal, need_weights=False)
    assert torch.allclose(mha(x, W_qkv, b_qkv, W_out, b_out, num_heads=H), expected, atol=1e-5)

def test_linear_init_keeps_variance():
    torch.manual_seed(0)
    out = Linear(512, 512)(torch.randn(1000, 512))
    assert 0.8 < out.std().item() < 1.2

def test_mse():
    from modules.loss import mse
    a, b = torch.randn(10), torch.randn(10)
    assert torch.allclose(mse(a, b), F.mse_loss(a, b))

def test_sgd_matches_torch():
    from modules.optim import SGD
    w1 = torch.randn(5, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    ours, ref = SGD([w1], lr=0.1), torch.optim.SGD([w2], lr=0.1)
    for opt, w in ((ours, w1), (ref, w2)):
        for _ in range(3):
            opt.zero_grad()
            (w ** 2).sum().backward()
            opt.step()
    assert torch.allclose(w1, w2)
