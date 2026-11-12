from models.example.model import MyGPT
from modules.loss import cross_entropy
from modules.optim import AdamW, warmup_cosine_lr, clip_grad_norm
from modules.metrics import accuracy, perplexity
import torch
import random

FRUITS = ['apple', 'banana', 'orange', 'grape', 'lemon', 'lime', 'cherry', 'peach', 'pear', 'plum',
          'berry', 'melon', 'mango', 'kiwi', 'fig', 'date', 'coconut', 'papaya', 'guava', 'lychee']

def get_train_batch(tokenizer, batch_size=4):
    """Next-token prediction: the target is the input shifted left by one.

        word:    a  p  p  l  e  <EOS>
        input:   a  p  p  l  e           (everything except the last token)
        target:  p  p  l  e  <EOS>       (everything except the first token)

    At every position the model learns "given what I have seen so far, what comes next?"
    See lessons/23_next_token_training.md.
    """
    words = random.sample(FRUITS, batch_size)
    ids, _ = tokenizer.tokenize(words)  # [B, T+1], right-padded with 0
    input_ids = ids[:, :-1]             # [B, T]
    target_ids = ids[:, 1:]             # [B, T]
    return input_ids, target_ids

def train(steps=2000, batch_size=8, max_lr=3e-3, warmup_steps=100, save_dir='models/saved_models'):
    """Everything here is our own code (see lessons/25_train_end_to_end.md):
    1. Build the model and the AdamW optimizer.
    2. Repeat: set lr -> sample a batch -> logits -> cross-entropy -> backward -> clip -> update.
    3. Save the trained model (skipped if save_dir is None).
    """
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGPT(max_seq_length=100, embed_dim=32, num_layers=2, num_heads=2).to(device)
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

    for step in range(steps):
        optimizer.lr = warmup_cosine_lr(step, max_lr, warmup_steps, total_steps=steps, min_lr=max_lr / 10)

        input_ids, target_ids = get_train_batch(model.tokenizer, batch_size=batch_size)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)  # [B, T, V]

        # flatten [B, T, V] -> [B*T, V] so every position is one classification problem;
        # ignore_index=0 skips padding positions
        logits, target_ids = logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
        loss = cross_entropy(logits, target_ids, ignore_index=0)

        optimizer.zero_grad()                        # clear old gradients
        loss.backward()                              # compute new gradients
        clip_grad_norm(model.parameters(), 1.0)      # no giant steps
        optimizer.step()                             # update parameters

        if step % 200 == 0 or step == steps - 1:
            acc = accuracy(logits, target_ids, ignore_index=0)
            print(f'step {step:4d} | lr {optimizer.lr:.5f} | loss {loss.item():.3f} '
                  f'| perplexity {perplexity(loss.item()):6.2f} | accuracy {acc:.2f}')
    if save_dir is not None:
        model.save(save_dir)
    return model

if __name__ == '__main__':
    train()
