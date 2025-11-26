from models.example.model import MyGPT
import torch.nn as nn
import torch.optim as optim
import torch
import random

def load_dataset():
    """Generate simple training data for character-level language modeling.
    Each data point is a tuple of (input_text, target_text), where target_text is
    the continuation of input_text.
    Returns:
        List of tuples (input_text: str, target_text: str)

    Note this is a simple hardcoded dataset for demonstration purposes. 
    Consider overloading with your own dataset for better results.
    """
    training_data = ['apple', 'banana', 'orange', 'grape', 'lemon', 'lime', 'cherry', 'peach', 'pear', 'plum', 'berry', 'melon', 'mango', 'kiwi', 'fig', 'date', 'coconut', 'papaya', 'guava', 'lychee']
    results = []
    for d in training_data:
        for input_text, target_text in [(d[:i], d[i:]) for i in range(1, len(d)+1)]:
            results.append((input_text, target_text))
    return results

def get_train_batch(tokenizer, batch_size=4):
    batch_data = random.sample(load_dataset(), batch_size)
    for in_text, tgt_text in batch_data:
        assert len(tgt_text) == 1, "Target text cannot be empty, and should only have one token (but character in our case)."
        if len(in_text) == 0:
            in_text = '.'  # add a period to avoid empty input
    input_texts, target_texts = zip(*batch_data)
    input_ids, _ = tokenizer.tokenize(input_texts)
    target_ids, _ = tokenizer.tokenize(target_texts)
    return input_ids, target_ids

def train():
    """
    Trains a simple MyGPT model on character-level language modeling task.
    1. Loads a simple dataset of fruit names.
    2. Initializes the MyGPT model and Adam optimizer.
    3. For a number of training steps:
        - Samples a batch of training data.
        - Computes the model's logits.
        - Computes cross-entropy loss against target tokens.
        - Backpropagates the loss and updates model parameters.
    4. Saves the trained model.
    Note: This is a simple demo only and is fully customizable. 
    Make your own dataset and training techniques for better results.
    """
    # Remember to adjust your tokenizer and model parameters here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGPT(vocab_size=87, max_seq_length=100, embed_dim=30, num_layers=2, num_heads=2).to(device)
    # adam optimizer is used
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for step in range(1000):
        input_ids, target_ids = get_train_batch(model.tokenizer, batch_size=4)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)

        # compute cross-entropy loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
       
        optimizer.zero_grad()  # clear optimizer grad cache
        loss.backward() # backpropagate loss
        optimizer.step() # update parameters
        
        # print loss every 100 steps
        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss.item()}')
    model.save('models/saved_models')

if __name__ == '__main__':
    train()