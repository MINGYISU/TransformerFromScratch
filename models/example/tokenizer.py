import json
import os
import string
import torch
class MyGptTokenizer:
    vocabulary: dict

    def __init__(self, vocab_path: str=None):
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocabulary = json.load(f)
        else:
            # dummpy vocabulary
            self.vocabulary = {k: i for i, k in enumerate(string.ascii_letters+string.punctuation+' ', start=1)}
            self.vocabulary['<unk>'] = 0
            self.vocabulary['<EOS>'] = len(self.vocabulary)

    def vocab_size(self):
        return len(self.vocabulary)
    
    def save_vocab(self, vocab_path: str):
        with open(vocab_path, 'w') as f:
            json.dump(self.vocabulary, f)
    
    def tokenize(self, texts):
        """Args:
            texts: List of strings, shape [batch size]
        
        Returns:
            List of List of token indices, shape [batch size, sequence length]
        """
        tokenized = []
        for text in texts:
            tokens = [self.vocabulary.get(char, 0) for char in text]
            tokens.append(self.vocabulary['<EOS>'])
            tokenized.append(tokens)
        max_len = max(len(t) for t in tokenized)
        padded = [t + [0]*(max_len - len(t)) for t in tokenized]
        mask = [[1]*len(t) + [0]*(max_len - len(t)) for t in tokenized]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
    
    def decode(self, token_ids):
        """Args:
            token_ids: List of List of token indices List[List[[int]]], shape [batch size, sequence length]
        
        Returns:
            List of strings, shape [batch size]
        """
        inv_vocab = {v: k for k, v in self.vocabulary.items()}
        texts = []
        for ids in token_ids:
            # Filter out padding and end-of-sequence tokens before decoding
            # map token ids to corresponding characters
            chars = [inv_vocab.get(i, '') 
                     for i in ids 
                     if i != 0 
                     and i != self.vocabulary['<EOS>']] 
            texts.append(''.join(chars))
        return texts