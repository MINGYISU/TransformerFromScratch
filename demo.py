from models.example.model import MyGPT
import torch

def main():
    torch.manual_seed(0)
    model = MyGPT.load('models/saved_models/MyGPT.pkl').to('cpu')
    eos = model.tokenizer.vocabulary['<EOS>']

    for prompt in ['a', 'b', 'c', 'gr', 'me', 'pa']:
        # prompts must NOT end with <EOS>, otherwise the model thinks the word is already finished
        input_ids, _ = model.tokenizer.tokenize([prompt], add_eos=False)
        output_ids = model.generate(input_ids, max_new_tokens=20, eos_token_id=eos)
        print(f"{prompt!r:6} -> {model.tokenizer.decode(output_ids.tolist())[0]}")

if __name__ == '__main__':
    main()
