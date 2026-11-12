"""Regenerate the figures used in the lessons:  pip install matplotlib && python lessons/figures/make_figures.py"""
import math
import pathlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent
sys.path.insert(0, str(OUT.parent.parent))   # repo root, so we can import our own modules
import torch
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)

def xs(lo, hi, n=200):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

# 00: an LLM outputs a probability for every possible next token
fig, ax = plt.subplots(figsize=(6, 3))
tokens, probs = ['a', 'e', 'o', 'i', '<EOS>', 'z'], [0.86, 0.05, 0.04, 0.03, 0.01, 0.01]
ax.bar(tokens, probs, color=['tab:green'] + ['tab:gray'] * 5)
ax.set_title('Input "ban" → probability of each next token')
ax.set_ylabel('probability')
save(fig, '00_next_token.png')

# 01: scalar, vector, matrix, 3-D tensor
fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
def grid(ax, rows, cols, title, depth=1):
    for d in range(depth - 1, -1, -1):
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((c + 0.3 * d, rows - 1 - r + 0.3 * d), 1, 1,
                                           facecolor='tab:blue' if d == 0 else 'lightsteelblue', edgecolor='white', lw=2))
    ax.set_xlim(-0.5, 4.8); ax.set_ylim(-0.3, 3.8); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
grid(axes[0], 1, 1, 'scalar\nshape []')
grid(axes[1], 1, 4, 'vector\nshape [4]')
grid(axes[2], 3, 4, 'matrix\nshape [3, 4]')
grid(axes[3], 3, 4, '3-D tensor\nshape [2, 3, 4]', depth=2)
save(fig, '01_tensors.png')

# 02: f(x) = x² and its slope (tangent lines)
fig, ax = plt.subplots(figsize=(6, 4))
x = xs(-3, 3)
ax.plot(x, [v * v for v in x], color='black', label='f(x) = x²')
for x0, color in [(-2, 'tab:red'), (0, 'tab:gray'), (1, 'tab:blue')]:
    slope = 2 * x0
    t = xs(x0 - 1, x0 + 1, 2)
    ax.plot(t, [x0 * x0 + slope * (v - x0) for v in t], color=color, lw=2, label=f'slope at x={x0}: {slope}')
    ax.plot([x0], [x0 * x0], 'o', color=color)
ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.legend()
save(fig, '02_slope.png')

# 03: loss as a function of a guess
fig, ax = plt.subplots(figsize=(6, 4))
answer = 3.0
g = xs(-1, 7)
ax.plot(g, [(v - answer) ** 2 for v in g], color='black')
ax.plot([answer], [0], 'o', color='tab:green', ms=10, label='best guess = 3, loss = 0')
ax.plot([6], [9], 'o', color='tab:red', ms=10, label='guess = 6, loss = 9')
ax.set_xlabel('our guess'); ax.set_ylabel('loss = (guess − 3)²'); ax.legend()
save(fig, '03_loss.png')

# 04: gradient descent on f(x) = x² with three learning rates
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, lr, steps, title in zip(axes, [0.05, 0.3, 1.1], [8, 8, 4],
                                ['too small (lr = 0.05): slow', 'good (lr = 0.3)', 'too big (lr = 1.1): blows up']):
    x = xs(-5.5, 5.5)
    ax.plot(x, [v * v for v in x], color='lightgray')
    path = [3.0]
    for _ in range(steps):
        path.append(path[-1] - lr * 2 * path[-1])
    ax.plot(path, [p * p for p in path], 'o-', color='tab:blue', ms=5)
    ax.plot([path[0]], [path[0] ** 2], 'o', color='tab:red', ms=8)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylim(-0.5, 31)
axes[0].set_ylabel('f(x) = x²')
save(fig, '04_gradient_descent.png')

# 06: activation functions
fig, ax = plt.subplots(figsize=(6, 4))
x = xs(-3, 3)
ax.plot(x, [max(0.0, v) for v in x], label='ReLU', lw=2)
ax.plot(x, [v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x], label='GELU', lw=2, ls='--')
ax.axhline(0, color='lightgray', lw=1); ax.set_xlabel('input'); ax.set_ylabel('output'); ax.legend()
save(fig, '06_activations.png')
# 07: XOR — no single line separates the classes; a small network bends the boundary
from models.mlp.xor import train as train_xor, X as XOR_X, Y as XOR_Y
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax in axes:
    for (x1, x2), y in zip(XOR_X.tolist(), XOR_Y.squeeze().tolist()):
        ax.scatter(x1, x2, s=250, c='tab:orange' if y else 'tab:blue', edgecolors='black', zorder=3)
        ax.annotate(f'XOR = {int(y)}', (x1, x2), textcoords='offset points', xytext=(0, 14), ha='center')
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.6); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_aspect('equal')
axes[0].plot([-0.5, 1.5], [1.2, -0.3], '--', color='gray'); axes[0].set_title('a straight line always gets one wrong')
xor_net = train_xor(verbose=False)
g = torch.linspace(-0.5, 1.6, 120)
gx, gy = torch.meshgrid(g, g, indexing='xy')
with torch.no_grad():
    z = xor_net(torch.stack([gx.flatten(), gy.flatten()], dim=-1)).reshape(gx.shape)
axes[1].contourf(gx, gy, z > 0.5, levels=[-0.5, 0.5, 1.5], colors=['#cfe0f3', '#fde0c5'])
axes[1].set_title('a tiny network learns a bent boundary')
save(fig, '07_xor.png')

# 09: cross-entropy = -log(p of the correct answer)
fig, ax = plt.subplots(figsize=(6, 4))
p = xs(0.01, 1.0)
ax.plot(p, [-math.log(v) for v in p], color='black')
for pv, color in [(0.9, 'tab:green'), (0.5, 'tab:orange'), (0.05, 'tab:red')]:
    ax.plot([pv], [-math.log(pv)], 'o', ms=9, color=color, label=f'p = {pv} → loss = {-math.log(pv):.2f}')
ax.set_xlabel('probability the model gave to the correct answer'); ax.set_ylabel('loss = −log p'); ax.legend()
save(fig, '09_cross_entropy.png')

# 11: paths of SGD, momentum and Adam on a long, narrow valley f(x, y) = 0.05 x² + y²
from modules.optim import SGD, Adam
def valley(w): return 0.05 * w[0] ** 2 + w[1] ** 2
fig, ax = plt.subplots(figsize=(8, 4))
gx, gy = torch.meshgrid(torch.linspace(-5.5, 1, 100), torch.linspace(-2.2, 2.2, 100), indexing='xy')
ax.contour(gx, gy, 0.05 * gx ** 2 + gy ** 2, levels=15, colors='lightgray')
for name, make, color in [('SGD', lambda p: SGD(p, lr=0.9), 'tab:red'),
                          ('SGD + momentum', lambda p: SGD(p, lr=0.3, momentum=0.7), 'tab:blue'),
                          ('Adam', lambda p: Adam(p, lr=0.2), 'tab:green')]:
    w = torch.tensor([-5.0, 2.0], requires_grad=True)
    opt, path = make([w]), [w.detach().clone()]
    for _ in range(25):
        opt.zero_grad(); valley(w).backward(); opt.step(); path.append(w.detach().clone())
    path = torch.stack(path)
    ax.plot(path[:, 0], path[:, 1], 'o-', ms=3, color=color, label=name)
ax.plot([0], [0], '*', ms=15, color='black', label='minimum')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('25 steps on a long, narrow valley'); ax.legend(loc='lower right')
save(fig, '11_optimizers.png')

# 12: warmup + cosine learning-rate schedule
from modules.optim import warmup_cosine_lr
fig, ax = plt.subplots(figsize=(6, 3.5))
steps = list(range(2000))
ax.plot(steps, [warmup_cosine_lr(s, 3e-3, 100, 2000, 3e-4) for s in steps], color='tab:blue')
ax.axvspan(0, 100, color='tab:orange', alpha=0.2, label='warmup')
ax.set_xlabel('training step'); ax.set_ylabel('learning rate'); ax.legend()
save(fig, '12_lr_schedule.png')
print('figures written to', OUT)
