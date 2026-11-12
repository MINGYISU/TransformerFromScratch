# TransformerFromScratch

**Learn how a GPT works by building one, one small idea at a time.**

This is a tutorial, not a tool. Every part of a small GPT — layers, attention, the model,
and soon the loss and the optimizer — is written by hand in plain, commented Python.
From PyTorch we only borrow **tensors** and **autograd** (automatic gradients).
PyTorch's own `nn` / `optim` versions are used **only in the tests**, as the answer key.

The finished model learns to spell fruit names: give it `ba` and it writes `banana`.

## The rules

1. **One idea per lesson.** Each lesson fits on a screen.
2. **Simplest version first.** Extra details are folded away in a *Going deeper* box.
3. **Everything runs.** Each lesson's code runs top to bottom, and `pytest` checks every lesson.

## Quick start

```bash
pip install -r requirements.txt
python -m models.example.train    # train the tiny GPT (a few seconds on CPU)
python demo.py                    # 'ba' -> banana, 'gr' -> grape, ...
python -m pytest                  # check our code against PyTorch + run every lesson
```

Then start reading at [lesson 00](lessons/00_how_to_use.md).

## Course map

✅ = ready · 🚧 = coming soon

### Part 1 · How a model learns
| # | Lesson | Idea |
|---|---|---|
| 00 | ✅ [How to use this course](lessons/00_how_to_use.md) | rules, lesson layout, running code |
| 01 | ✅ [Tensors and shapes](lessons/01_tensors_and_shapes.md) | `[B, T, E]` and broadcasting |
| 02 | ✅ [Autograd](lessons/02_autograd.md) | `.backward()` gives gradients for free |
| 03 | ✅ [Linear layer](lessons/03_linear_layer.md) | `y = xWᵀ + b`, and why init scale matters |
| 04 | ✅ [Activations](lessons/04_activations.md) | ReLU, GELU: bending straight lines |
| 05 | 🚧 MSE loss | measuring "how wrong" with squared error |
| 06 | 🚧 Gradient descent (SGD) | `w -= lr * grad` |
| 07 | 🚧 Your first neural network | a tiny MLP learns XOR |

### Part 2 · Learning to choose
| # | Lesson | Idea |
|---|---|---|
| 08 | ✅ [Softmax](lessons/08_softmax.md) | scores → probabilities |
| 09 | 🚧 Cross-entropy loss | `-log p(correct answer)` |
| 10 | 🚧 Metrics | accuracy, perplexity |
| 11 | 🚧 Momentum and Adam | smarter steps than plain SGD |
| 12 | 🚧 Training tricks | lr warmup/decay, gradient clipping, dropout |

### Part 3 · Text → numbers
| # | Lesson | Idea |
|---|---|---|
| 13 | ✅ [Tokenizer](lessons/13_tokenizer.md) | characters → ids, padding, `<EOS>` |
| 14 | ✅ [Token embeddings](lessons/14_token_embeddings.md) | a learnable vector per token |
| 15 | ✅ [Positional embeddings](lessons/15_positional_embeddings.md) | telling the model *where* each token is |

### Part 4 · The Transformer
| # | Lesson | Idea |
|---|---|---|
| 16 | ✅ [Attention](lessons/16_attention.md) | `softmax(QKᵀ/√d) V` |
| 17 | ✅ [Masks](lessons/17_masks.md) | no peeking at the future |
| 18 | ✅ [Multi-head attention](lessons/18_multi_head_attention.md) | several attentions side by side |
| 19 | ✅ [LayerNorm](lessons/19_layer_norm.md) | keeping numbers in a healthy range |
| 20 | ✅ [Residuals and feed-forward](lessons/20_residual_and_feed_forward.md) | `x = x + f(x)` |
| 21 | ✅ [The transformer block](lessons/21_transformer_block.md) | putting 16–20 together |

### Part 5 · The language model
| # | Lesson | Idea |
|---|---|---|
| 22 | ✅ [Assembling GPT](lessons/22_assemble_gpt.md) | embeddings → blocks → logits |
| 23 | ✅ [Next-token training](lessons/23_next_token_training.md) | target = input shifted by one |
| 24 | ✅ [Generating text](lessons/24_generation.md) | sample, append, repeat |
| 25 | 🚧 Train it end to end | loss curves, perplexity, saving and loading |

## Project structure

```
lessons/              the course, one markdown file per lesson
modules/
  function.py         the math as plain functions: gelu, softmax, linear, layer_norm, attention, mha
  module.py           classes that hold weights: Module, Parameter, Linear, LayerNorm, attention, block
models/example/
  tokenizer.py        character-level tokenizer
  model.py            MyGPT: the full model + text generation
  train.py            next-token training on fruit names
demo.py               generate fruit names with the trained model
tests/                our code vs PyTorch's, plus a runner for every lesson
```

## Acknowledgements

Inspired by OpenAI's GPT-2 and many from-scratch tutorials.

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI. [GitHub](https://github.com/openai/gpt-2)
- Vaswani, A. et al. (2017). *Attention Is All You Need*. [arXiv](https://arxiv.org/abs/1706.03762)
- PyTorch. [GitHub](https://github.com/pytorch/pytorch)
- tiny-gpt2. [GitHub](https://github.com/sdiehl/tiny-gpt2)

## License

MIT — see [LICENSE](LICENSE).
