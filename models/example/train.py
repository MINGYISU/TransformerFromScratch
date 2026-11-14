from models.example.model import MyGPT
from modules.loss import cross_entropy
from modules.optim import SGD, AdamW, warmup_cosine_lr, clip_grad_norm
from modules import metrics
import torch
import random

FRUITS = ['apple', 'banana', 'orange', 'grape', 'lemon', 'lime', 'cherry', 'peach', 'pear', 'plum',
          'berry', 'melon', 'mango', 'kiwi', 'fig', 'date', 'coconut', 'papaya', 'guava', 'lychee']
PROMPTS = sorted({fruit[:2] for fruit in FRUITS})   # 'ap', 'ba', ... used to test spelling

def get_train_batch(tokenizer, batch_size=4):
    """Next-token prediction: the target is the input shifted left by one.

        word:    a  p  p  l  e  <EOS>
        input:   a  p  p  l  e           (everything except the last token)
        target:  p  p  l  e  <EOS>       (everything except the first token)

    At every position the model learns "given what I have seen so far, what comes next?"
    See lessons/23_next_token_training.md.
    """
    words = random.choices(FRUITS, k=batch_size)  # with replacement, so any batch size works
    ids, _ = tokenizer.tokenize(words)  # [B, T+1], right-padded with 0
    input_ids = ids[:, :-1]             # [B, T]
    target_ids = ids[:, 1:]             # [B, T]
    return input_ids, target_ids

def evaluate(model) -> dict:
    """Score the model on ALL fruits at once with our metrics (see lessons/10_metrics.md):
    loss, perplexity, accuracy, hit@3 and MRR of the next letter, plus 'spelling':
    the fraction of 2-letter prompts that the model (greedily) completes into a real fruit name.
    """
    device = model.token_embed.device
    tok = model.tokenizer
    with torch.no_grad():
        ids, _ = tok.tokenize(FRUITS)
        ids = ids.to(device)
        logits = model(ids[:, :-1]).reshape(-1, model.vocab_size)
        targets = ids[:, 1:].reshape(-1)
        keep = targets != 0                                   # skip padding
        logits, targets = logits[keep], targets[keep]
        loss = cross_entropy(logits, targets).item()

        prompts, _ = tok.tokenize(PROMPTS, add_eos=False)
        out = model.generate(prompts.to(device), max_new_tokens=12, eos_token_id=tok.vocabulary['<EOS>'], greedy=True)
        words = tok.decode(out.tolist())
    return {
        'loss': loss,
        'perplexity': metrics.perplexity(loss),
        'accuracy': metrics.accuracy(logits, targets),
        'hit@3': metrics.hit_at_k(logits, targets, k=3),
        'mrr': metrics.mrr(logits, targets),
        'spelling': sum(word in FRUITS for word in words) / len(words),
    }

def train(steps=2000, batch_size=8, embed_dim=32, num_layers=2, num_heads=2,
          max_lr=3e-3, warmup_steps=100, schedule=True, optimizer='adamw', weight_decay=0.01, clip=1.0,
          seed=0, eval_every=200, verbose=True, save_dir='models/saved_models'):
    """Train MyGPT on fruit names. Everything here is our own code (see lessons/25 and 26).
    Every setting is a parameter, so we can experiment with it:
        batch_size, embed_dim, num_layers, num_heads   - data per step and model size
        max_lr, warmup_steps, schedule                 - learning rate (schedule=False: constant max_lr)
        optimizer ('adamw' or 'sgd'), weight_decay     - how to step
        clip                                           - max gradient length (None = no clipping)
    The evaluation results (every eval_every steps) are kept in model.history.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGPT(max_seq_length=100, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads).to(device)
    if optimizer == 'adamw':
        opt = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    else:
        opt = SGD(model.parameters(), lr=max_lr, momentum=0.9)
    model.history = []

    for step in range(steps):
        opt.lr = warmup_cosine_lr(step, max_lr, warmup_steps, total_steps=steps, min_lr=max_lr / 10) if schedule else max_lr

        input_ids, target_ids = get_train_batch(model.tokenizer, batch_size=batch_size)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)  # [B, T, V]

        # flatten [B, T, V] -> [B*T, V] so every position is one classification problem;
        # ignore_index=0 skips padding positions
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=0)

        opt.zero_grad()                                  # clear old gradients
        loss.backward()                                  # compute new gradients
        if clip is not None:
            clip_grad_norm(model.parameters(), clip)     # no giant steps
        opt.step()                                       # update parameters

        if (eval_every and step % eval_every == 0) or step == steps - 1:
            result = {'step': step, 'lr': opt.lr, 'train_loss': loss.item(), **evaluate(model)}
            model.history.append(result)
            if verbose:
                print(f"step {step:4d} | lr {opt.lr:.5f} | loss {result['loss']:.3f} "
                      f"| perplexity {result['perplexity']:6.2f} | accuracy {result['accuracy']:.2f} "
                      f"| spelling {result['spelling']:.2f}")
    if save_dir is not None:
        model.save(save_dir)
    return model

if __name__ == '__main__':
    train()
