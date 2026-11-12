# TransformerFromScratch

**Learn how a transformer and a large language model (LLM) work, one small idea at a time.**

This is a tutorial, not a tool. Each lesson explains **one concept** of LLMs —
what a tensor is, what a gradient is, why we minimise a loss, how attention works — first in plain
words, pictures and a tiny example you could do by hand, and only then in code.

PyTorch is our **calculator**: we use its tensors and its automatic gradients, and nothing else.
Every part of the GPT itself — layers, attention, the model, the loss, the optimizer — is written by hand.
(PyTorch's ready-made versions appear only in the tests, as the answer key.)

The finished model learns to spell fruit names: give it `ba` and it writes `banana`.

## The rules

1. **Concept first, code second.** Every lesson starts with the idea and the math, small enough to do by hand.
   Code comes after, to check the idea and let you play with it.
2. **One idea per lesson.** Each lesson fits on a screen or two.
3. **Simplest version first.** Extra details are folded away in a *Going deeper* box — skip them on a first read.
4. **Everything runs.** Each lesson's code runs top to bottom, and `pytest` checks every lesson.

## How to read a lesson

| Section | What it gives you |
|---|---|
| **In one sentence** | the idea, as short as possible |
| **The idea** | why it exists, in words and pictures, with a tiny example |
| **Code** | the real code from this repo |
| **Try it** | something to run and look at |
| **Check yourself** | a question or two (answers folded) |
| **Going deeper** | optional extras |

## Quick start

```bash
pip install -r requirements.txt
python -m models.example.train    # train the tiny GPT (a few seconds on CPU)
python demo.py                    # 'ba' -> banana, 'gr' -> grape, ...
python -m pytest                  # check our code against PyTorch + run every lesson
```

Then start reading at [lesson 00](lessons/00_what_is_a_transformer.md) — no code there, just the big picture.
To run a lesson's code yourself, paste its `python` blocks in order into `python` started at the repo root.

## Course map

✅ = ready · 🚧 = coming soon

### Part 0 · The big picture
| # | Lesson | Idea |
|---|---|---|
| 00 | ✅ [What is a transformer?](lessons/00_what_is_a_transformer.md) | an LLM guesses the next token; what's inside; how it learns |

### Part 1 · The math tools
| # | Lesson | Idea |
|---|---|---|
| 01 | ✅ [Tensors](lessons/01_tensors.md) | scalars, vectors, matrices; the dot product |
| 02 | ✅ [Gradients](lessons/02_gradients.md) | the slope of `x²`, and which way is downhill |
| 03 | ✅ [Loss](lessons/03_loss.md) | one number for "how wrong", and why we want its minimum |
| 04 | ✅ [Gradient descent](lessons/04_gradient_descent.md) | walk downhill to the minimum, step by step |

### Part 2 · Neural networks
| # | Lesson | Idea |
|---|---|---|
| 05 | ✅ [Linear layer](lessons/05_linear_layer.md) | a neuron = weighted sum + bias |
| 06 | ✅ [Activations](lessons/06_activations.md) | ReLU, GELU: bending straight lines |
| 07 | 🚧 Your first neural network | a tiny network learns XOR |

### Part 3 · Choosing among options
| # | Lesson | Idea |
|---|---|---|
| 08 | ✅ [Softmax](lessons/08_softmax.md) | scores → probabilities |
| 09 | 🚧 Cross-entropy loss | `-log p(correct answer)` |
| 10 | 🚧 Metrics | accuracy, perplexity |
| 11 | 🚧 Momentum and Adam | smarter steps than plain SGD |
| 12 | 🚧 Training tricks | lr warmup/decay, gradient clipping, dropout |

### Part 4 · Text → numbers
| # | Lesson | Idea |
|---|---|---|
| 13 | ✅ [Tokenizer](lessons/13_tokenizer.md) | characters → ids, padding, `<EOS>` |
| 14 | ✅ [Token embeddings](lessons/14_token_embeddings.md) | a learnable vector per token |
| 15 | ✅ [Positional embeddings](lessons/15_positional_embeddings.md) | telling the model *where* each token is |

### Part 5 · The transformer
| # | Lesson | Idea |
|---|---|---|
| 16 | ✅ [Attention](lessons/16_attention.md) | `softmax(QKᵀ/√d) V` |
| 17 | ✅ [Masks](lessons/17_masks.md) | no peeking at the future |
| 18 | ✅ [Multi-head attention](lessons/18_multi_head_attention.md) | several attentions side by side |
| 19 | ✅ [LayerNorm](lessons/19_layer_norm.md) | keeping numbers in a healthy range |
| 20 | ✅ [Residuals and feed-forward](lessons/20_residual_and_feed_forward.md) | `x = x + f(x)` |
| 21 | ✅ [The transformer block](lessons/21_transformer_block.md) | putting 16–20 together |

### Part 6 · The language model
| # | Lesson | Idea |
|---|---|---|
| 22 | ✅ [Assembling GPT](lessons/22_assemble_gpt.md) | embeddings → blocks → logits |
| 23 | ✅ [Next-token training](lessons/23_next_token_training.md) | target = input shifted by one |
| 24 | ✅ [Generating text](lessons/24_generation.md) | sample, append, repeat |
| 25 | 🚧 Train it end to end | loss curves, perplexity, saving and loading |

## Project structure

```
lessons/              the course, one markdown file per lesson
  figures/            plots used in the lessons (+ make_figures.py to redraw them)
modules/
  function.py         the math as plain functions: gelu, softmax, linear, layer_norm, attention, mha
  module.py           classes that hold weights: Module, Parameter, Linear, LayerNorm, attention, block
  loss.py             mse (cross-entropy coming soon)
  optim.py            SGD (Adam coming soon)
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
