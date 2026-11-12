"""Loss, optimizers, schedules, metrics: ours vs. PyTorch / scikit-learn (answer keys only)."""
import math
import pytest
import torch
import torch.nn.functional as F
from modules.function import log_softmax, dropout, relu
from modules.loss import cross_entropy
from modules.optim import SGD, Adam, AdamW, warmup_cosine_lr, clip_grad_norm
from modules import metrics

torch.manual_seed(0)

def test_relu_and_log_softmax():
    x = torch.randn(4, 7) * 10
    assert torch.equal(relu(x), F.relu(x))
    assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-5)

@pytest.mark.parametrize('ignore_index, smoothing', [(None, 0.0), (0, 0.0), (None, 0.1), (0, 0.2)])
def test_cross_entropy(ignore_index, smoothing):
    logits, targets = torch.randn(12, 5), torch.randint(0, 5, (12,))
    expected = F.cross_entropy(logits, targets, ignore_index=-100 if ignore_index is None else ignore_index,
                               label_smoothing=smoothing)
    assert torch.allclose(cross_entropy(logits, targets, ignore_index, smoothing), expected, atol=1e-5)

def test_dropout():
    x = torch.ones(100_000)
    y = dropout(x, p=0.3)
    assert abs((y == 0).float().mean().item() - 0.3) < 0.01      # ~30% dropped
    assert abs(y.mean().item() - 1.0) < 0.02                     # average unchanged
    assert torch.equal(dropout(x, p=0.3, training=False), x)

def _run(make_ours, make_ref, steps=5):
    w0 = torch.randn(3, 4)
    results = []
    for make in (make_ours, make_ref):
        w = w0.clone().requires_grad_(True)
        opt = make([w])
        for _ in range(steps):
            opt.zero_grad()
            ((w ** 3).sum() + (w * 2).sum()).backward()
            opt.step()
        results.append(w.detach())
    return results

@pytest.mark.parametrize('ours, ref', [
    (lambda p: SGD(p, lr=0.01, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
    (lambda p: Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
    (lambda p: AdamW(p, lr=0.01, weight_decay=0.1), lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
], ids=['sgd_momentum', 'adam', 'adamw'])
def test_optimizers_match_torch(ours, ref):
    a, b = _run(ours, ref)
    assert torch.allclose(a, b, atol=1e-6)

def test_warmup_cosine_lr():
    lrs = [warmup_cosine_lr(s, max_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1) for s in range(120)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)         # linear warmup
    assert lrs[60] == pytest.approx(0.55)                                       # halfway down the cosine
    assert lrs[110] == pytest.approx(0.1) and lrs[119] == pytest.approx(0.1)    # stays at min_lr

def test_clip_grad_norm():
    w1 = torch.randn(10, requires_grad=True); w2 = w1.detach().clone().requires_grad_(True)
    for w in (w1, w2):
        (w * 100).sum().backward()
    norm = clip_grad_norm([w1], max_norm=1.0)
    ref = torch.nn.utils.clip_grad_norm_([w2], max_norm=1.0)
    assert norm == pytest.approx(ref.item(), rel=1e-5)
    assert torch.allclose(w1.grad, w2.grad)

def test_classification_metrics():
    logits = torch.tensor([[2., 1.], [0., 3.], [5., 1.], [1., 2.]])
    targets = torch.tensor([0, 1, 1, 1])
    assert metrics.accuracy(logits, targets) == 0.75
    assert metrics.accuracy(logits, targets, ignore_index=0) == pytest.approx(2 / 3)
    assert metrics.perplexity(math.log(87)) == pytest.approx(87)

def test_binary_metrics_match_sklearn():
    sk = pytest.importorskip('sklearn.metrics')
    preds, labels = torch.randint(0, 2, (50,)), torch.randint(0, 2, (50,))
    assert metrics.precision(preds, labels) == pytest.approx(sk.precision_score(labels.numpy(), preds.numpy()))
    assert metrics.recall(preds, labels) == pytest.approx(sk.recall_score(labels.numpy(), preds.numpy()))
    assert metrics.f1(preds, labels) == pytest.approx(sk.f1_score(labels.numpy(), preds.numpy()))

def test_ranking_metrics():
    logits = torch.tensor([[0.1, 0.9, 0.5], [0.8, 0.1, 0.3]])
    targets = torch.tensor([2, 0])                       # ranks: 2 and 1
    assert metrics.hit_at_k(logits, targets, k=1) == 0.5
    assert metrics.hit_at_k(logits, targets, k=2) == 1.0
    assert metrics.mrr(logits, targets) == pytest.approx((1 / 2 + 1) / 2)

def test_ndcg_matches_sklearn():
    sk = pytest.importorskip('sklearn.metrics')
    scores, relevance = torch.rand(8), torch.randint(0, 4, (8,)).float()
    relevance[0] = 3.0
    for k in (1, 3, 8):
        expected = sk.ndcg_score(relevance.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=k)
        assert metrics.ndcg_at_k(scores, relevance, k) == pytest.approx(expected, rel=1e-5)

def test_xor_network_learns():
    from models.mlp.xor import train, X, Y
    model = train(verbose=False)
    assert torch.allclose(model(X).detach(), Y, atol=0.1)
