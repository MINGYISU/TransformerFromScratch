"""Training experiments on the fruit task (see lessons/26_training_experiments.md).

Change ONE setting at a time from a baseline, train with 3 seeds each, and compare with our metrics.
Run:  python -m models.example.experiments      (a few minutes on a laptop CPU)
Writes models/example/experiment_results.json (plotted by lessons/figures/make_figures.py)
and prints one markdown table per experiment.
"""
import contextlib
import io
import json
import math
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from models.example.train import FRUITS, train

BASELINE = dict(batch_size=8, embed_dim=32, num_layers=2, num_heads=2, max_lr=3e-3,
                schedule=True, optimizer='adamw', weight_decay=0.01, clip=1.0)
STEPS = 400          # a small budget, so the differences are visible
EVAL_EVERY = 20
SEEDS = [0, 1, 2]

EXPERIMENTS = {
    'batch size':    [dict(batch_size=b) for b in (1, 4, 8, 32)],
    'model width':   [dict(embed_dim=e) for e in (8, 16, 32, 64)],
    'model depth':   [dict(num_layers=n) for n in (1, 2, 4)],
    'heads':         [dict(num_heads=h) for h in (1, 2, 4)],
    'learning rate': [dict(max_lr=lr) for lr in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    'training tricks': [dict(), dict(schedule=False), dict(clip=None), dict(weight_decay=0.0),
                        dict(optimizer='sgd', max_lr=0.05)],
    # the winners of the experiments above, together; then one more "winner" on top
    'combining winners': [dict(), dict(batch_size=32, embed_dim=64, num_layers=1, max_lr=1e-2),
                          dict(batch_size=32, embed_dim=64, num_layers=1, max_lr=1e-2, schedule=False)],
}
METRICS = ['loss', 'perplexity', 'accuracy', 'hit@3', 'mrr', 'spelling']

def best_possible():
    """The lowest loss / highest accuracy ANY model can reach on this data.
    Some next letters are truly ambiguous (after 'p': e, l or a), so even a perfect model can't be sure."""
    following = defaultdict(Counter)          # prefix -> counts of the next token
    for word in FRUITS:
        tokens = list(word) + ['<EOS>']
        for t in range(1, len(tokens)):
            following[word[:t]][tokens[t]] += 1
    n = sum(sum(c.values()) for c in following.values())
    loss = sum(k * -math.log(k / sum(c.values())) for c in following.values() for k in c.values()) / n
    accuracy = sum(max(c.values()) for c in following.values()) / n
    return {'loss': loss, 'accuracy': accuracy}

def label(change):
    return ', '.join(f'{k}={v}' for k, v in change.items()) or 'baseline'

def run(change, seeds=SEEDS, steps=STEPS):
    """Train once per seed with BASELINE + change; return mean/std of the final metrics and the mean loss curve."""
    config = {**BASELINE, **change}
    runs = []
    for seed in seeds:
        start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            model = train(steps=steps, eval_every=EVAL_EVERY, seed=seed, verbose=False, save_dir=None, **config)
        runs.append({'seconds': time.time() - start, 'history': model.history,
                     'params': sum(p.numel() for p in model.parameters())})
    final = [r['history'][-1] for r in runs]
    return {
        'label': label(change),
        'config': config,
        'params': runs[0]['params'],
        'seconds': statistics.mean(r['seconds'] for r in runs),
        'mean': {m: statistics.mean(f[m] for f in final) for m in METRICS},
        'std': {m: statistics.stdev(f[m] for f in final) if len(final) > 1 else 0.0 for m in METRICS},
        'curve_steps': [h['step'] for h in runs[0]['history']],
        'curve_loss': [statistics.mean(r['history'][i]['loss'] for r in runs) for i in range(len(runs[0]['history']))],
    }

def table(results):
    rows = ['| setting | params | loss | perplexity | accuracy | hit@3 | MRR | spelling | sec/run |',
            '|---|---|---|---|---|---|---|---|---|']
    for r in results:
        m, s = r['mean'], r['std']
        rows.append(f"| {r['label']} | {r['params']:,} | {m['loss']:.3f} ± {s['loss']:.3f} | {m['perplexity']:.2f} "
                    f"| {m['accuracy']:.3f} | {m['hit@3']:.3f} | {m['mrr']:.3f} | {m['spelling']:.2f} | {r['seconds']:.1f} |")
    return '\n'.join(rows)

if __name__ == '__main__':
    cache, output = {}, {'steps': STEPS, 'seeds': SEEDS, 'baseline': BASELINE, 'best_possible': best_possible(), 'experiments': {}}
    print('best possible:', output['best_possible'])
    for name, changes in EXPERIMENTS.items():
        results = []
        for change in changes:
            key = json.dumps({**BASELINE, **change}, sort_keys=True)
            if key not in cache:                 # the baseline appears in several experiments: run it once
                cache[key] = run(change)
            results.append({**cache[key], 'label': label(change)})
        output['experiments'][name] = results
        print(f'\n### {name}\n\n{table(results)}')
    path = pathlib.Path(__file__).with_name('experiment_results.json')
    path.write_text(json.dumps(output, indent=1))
    print('\nwrote', path)
