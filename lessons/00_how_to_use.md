# 00 · How to use this course

> In one sentence: small lessons, one idea each, every line of code runs.

## The rules of this project

1. **One idea per lesson.** If a lesson needs two ideas, it becomes two lessons.
2. **Simplest version first.** The main text shows the smallest code that works.
   Extra details live in a folded **Going deeper** box at the end — skip it on a first read.
3. **We only borrow two things from PyTorch:** tensors (fast arrays) and autograd
   (automatic gradients). Everything else — layers, attention, the GPT model — we write ourselves.
   PyTorch's own versions are only used in [`tests/`](../tests) as the answer key.
4. **Function first, then Module.** We first write the math as a plain function
   (in [`modules/function.py`](../modules/function.py)), then wrap it in a class that
   stores the weights (in [`modules/module.py`](../modules/module.py)).

## How every lesson looks

| Section | What it gives you |
|---|---|
| **In one sentence** | the idea, as short as possible |
| **Intuition** | why it exists, with a tiny example |
| **Code** | the real code from this repo |
| **Try it** | something to run and look at |
| **Check yourself** | a question or two (answers folded) |
| **Going deeper** | optional extras |

## Running the code

Run everything from the repository root:

```bash
pip install -r requirements.txt
python -m pytest              # checks our code against PyTorch AND runs every lesson
```

Every `python` block in a lesson can be run **top to bottom** in one Python session
(for example by pasting into `python` started at the repo root). Try it:

```python
import torch
print(torch.__version__)
print(torch.tensor([1.0, 2.0, 3.0]) * 2)   # tensor([2., 4., 6.])
```

---
[Course map](../README.md) · Next: [01 · Tensors and shapes](01_tensors_and_shapes.md)
