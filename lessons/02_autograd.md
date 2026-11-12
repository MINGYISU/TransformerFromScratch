# 02 · Autograd

> In one sentence: PyTorch remembers every calculation you do, so `.backward()` can tell you
> how much each input affected the output.

## Intuition

Training = nudging weights so the loss goes down. To know *which way* to nudge a weight `w`,
we need the **gradient** `d loss / d w`:

- gradient positive → increasing `w` increases the loss → make `w` smaller
- gradient negative → make `w` bigger

Computing gradients by hand for millions of weights is impossible. Autograd does it for us.
**This is the one feature of PyTorch this whole project relies on.**

## Try it

Take `y = w * x + b` with `w = 3, x = 2, b = 1`. By hand: `dy/dw = x = 2`, `dy/db = 1`.

```python
import torch

w = torch.tensor(3.0, requires_grad=True)   # "please track gradients for me"
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)                        # data: no gradient needed

y = w * x + b
y.backward()                                 # fill .grad of every tracked tensor

print(y.item())          # 7.0
print(w.grad.item())     # 2.0  == x
print(b.grad.item())     # 1.0
```

It works through any chain of operations — including our whole GPT model:

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (w * w - 4) ** 2        # by hand: d/dw = 2*(w*w-4)*2w = 2*5*6 = 60
loss.backward()
print(w.grad.item())           # 60.0
```

## Check yourself

1. Gradients **add up** each time you call `.backward()`. What must you do before each training step?

<details><summary>Answer</summary>

Reset them to zero (`optimizer.zero_grad()` in [`train.py`](../models/example/train.py)).
Otherwise each step would use the sum of all past gradients.
</details>

<details><summary>Going deeper: the computation graph and <code>no_grad</code></summary>

Every result remembers the operation that created it (`y.grad_fn`). `.backward()` walks this
graph backwards, applying the chain rule at each step. When we only *use* a model
(for example generating text), we don't need the graph — `torch.no_grad()` skips it and saves memory:

```python
with torch.no_grad():
    z = w * 2
print(z.requires_grad)   # False
```
</details>

---
[Course map](../README.md) · Prev: [01](01_tensors_and_shapes.md) · Next: [03 · Linear layer](03_linear_layer.md)
