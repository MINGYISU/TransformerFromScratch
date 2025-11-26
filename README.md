# TransformerFromScratch

This project implements a simple transformer architecture (GPT2-like) from scratch, using PyTorch. It is just a minimal implementation for quick demo and educational purpose. It is not optimized for performance or production use. Use PyTorch's built-in transformer modules for serious applications.

## Project Structure

- 'modules/': Contains core modules such as attention mechanisms, feedforward networks, and utility functions.

    - 'function.py': Utility functions that impement the computation, such as softmax, attention calculation, linear transformation, etc.

    - 'module.py': Wrappers for core transformer modules (PyTorch-Like) that contain weights, such as Multi-Head Attention and Feedforward layers.

- 'models/': Contains concrete instances of transformer models, including model architecture, tokenizer, and vocabulary.

    - 'example/': An example GPT model implementation and tokenizer.
        - 'model.py': Implementation of a simple GPT-like transformer model.
        - 'tokenizer.py': A basic tokenizer for text processing.
        - 'train.py': A training script to train the example GPT model on sample data.
        - 'saved_models/': Directory to save trained model checkpoints. (Run the training script first to generate this folder.)

- 'demo.py': A demo script to showcase the model's text generation capabilities.

## How to run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run training script:

   ```bash
   python models/example/train.py
   ```

3. Run demo script:

   ```bash
   python demo.py
   ```

## Explanation of Key Components

The provided example model is a simplified GPT2-like transformer architecture. Its vocabulary is simply the set of characters and punctuations used in the training text. It is trained to spell the fruit names given the a simple prompt of the first character of the word. Cross entropy loss is used to compute the loss between the predicted token probabilities and the ground truth tokens. Unfortunately, I cannot see impressive performance of this small model yet, so please treat this as a concept demo only. You can modify the model architecture, training data and methods, and hyperparameters to experiment further.

## Acknowledgements

This project is inspired by the gpt2 github by openai, as well as various tutorials and implementations of transformer models in PyTorch.

Key references:
**Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.** (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI. [GitHub](https://github.com/openai/gpt-2)
PyTorch. [GitHub](https://github.com/pytorch/pytorch.git)
tiny-gpt2. [GitHub](https://github.com/sdiehl/tiny-gpt2.git)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
