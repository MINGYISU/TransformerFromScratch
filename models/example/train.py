from models.example.model import MyGPT
import torch
import torch.nn as nn
import torch.optim as optim
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

def train(steps=2000, batch_size=8, save_dir='models/saved_models'):
    """
    1. Build the model and the Adam optimizer.
    2. Repeat: sample a batch -> logits -> cross-entropy loss -> backward -> update.
    3. Save the trained model (skipped if save_dir is None).
    """
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGPT(max_seq_length=100, embed_dim=32, num_layers=2, num_heads=2).to(device)
    # TODO(lessons 09 & 11): replace with our own cross_entropy and Adam
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    for step in range(steps):
        input_ids, target_ids = get_train_batch(model.tokenizer, batch_size=batch_size)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)  # [B, T, V]

        # flatten [B, T, V] -> [B*T, V] so every position is one classification problem;
        # ignore_index=0 skips padding positions
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=0)

        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # compute new gradients
        optimizer.step()       # update parameters

        if step % 200 == 0 or step == steps - 1:
            print(f'Step {step}, Loss: {loss.item():.4f}')
    if save_dir is not None:
        model.save(save_dir)
    return model

if __name__ == '__main__':
    train()
