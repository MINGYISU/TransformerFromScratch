from models.example.model import MyGPT
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyGPT(vocab_size=model.tokenizer.vocab_size(), max_seq_length=100, embed_dim=10, num_layers=2, num_heads=2).to(device)
    model = model.load('models/saved_models/MyGPT.pkl')

    input = ['a', 'ap', 'app', 'appl', 'apple', 'b', 'ba', 'ban', 'bana', 'banan', 'banana']
    input_ids, mask = model.tokenizer.tokenize(input)
    print(input_ids)

    output_id = model.generate(input_ids=input_ids.to(device), max_new_tokens=20, eos_token_id=model.tokenizer.vocabulary['<EOS>'])
    decoded = model.tokenizer.decode(output_id.tolist())
    for inp, out in zip(input, decoded):
        print(f"Input: {inp}")
        print(f"Output: {out} ✅ {len(out)} characters generated")
        print('-'*50)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    main()
