from .module import Module, ModuleList, TransformerBlock, LayerNorm
import torch.nn as nn
import torch

class MyGPT(Module):
    vocab_size: int
    max_seq_length: int
    embed_dim: int
    token_embed: nn.Parameter
    pos_embed: nn.Parameter
    blocks: ModuleList
    ln_f: LayerNorm

    def __init__(self, 
                 vocab_size: int, 
                 max_seq_length: int, 
                 embed_dim, 
                 num_layers, 
                 num_heads, 
                 *args, 
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.token_embed = nn.Parameter(torch.randn(vocab_size, embed_dim))
        # positional embedding vocabulary: [max_seq_length, embed_dim]
        self.pos_embed = nn.Parameter(torch.randn(max_seq_length, embed_dim))
        self.blocks = ModuleList([TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, *args, **kwargs) 
                                     for _ in range(num_layers)]) # stack of transformer blocks
        self.ln_f = LayerNorm(embed_dim, *args, **kwargs) # final layer norm

    def __call__(self, input_ids: torch.Tensor):
        _, T = input_ids.shape
        assert T <= self.max_seq_length, f"Sequence length {T} exceeds model's max_seq_length {self.max_seq_length}"
        emb = self.encode(input_ids)
        logits = self.forward(emb)
        return logits

    def _param_keys(self):
        return 'token_embed', 'pos_embed', 'blocks', 'ln_f'
    
    def _configs(self):
        return {'vocab_size': self.vocab_size, 
                'max_seq_length': self.max_seq_length, 
                'embed_dim': self.embed_dim, 
                'num_layers': len(self.blocks.modules), 
                'num_heads': self.blocks.modules[0].num_heads
                }
    
    def encode(self, input_ids: torch.Tensor):
        """Args:
            input_ids: torch.Tensor of shape [batch size, sequence length]
        Returns:
            x: torch.Tensor of shape [batch size, sequence length, embed_dim]
        """
        # retrieve input embeddings for each token in each sequence in each batch
        input_emb = self.token_embed[input_ids] # [batch size, sequence length] -> [batch size, sequence length, embed_dim]
        B, T, E = input_emb.shape # [batch size, sequence length, embedding dimension]
        # positional embeddings for each input emb in each batch: [T, embed_dim]
        # getting first T positions' embeddings
        pos_emb = self.pos_embed[:T]  # [T, E]
        # [B, T, E] + [T, E] -> [B, T, E]
        x = input_emb + pos_emb
        return x

    def forward(self, x):
        """Args:
            x: torch.Tensor of shape [batch size, sequence length, embed_dim]
        Returns:
            logits: torch.Tensor of shape [batch size, sequence length, vocab size]
        """
        x = self.blocks(x)
        x = self.ln_f(x)
        # Purpose: project raw outputs to the vocabulary and get logits (dot product scores of vocabulary embeddings and output vectors) for each token position
        # x: [B, T, E], token_embed: [V, E]
        # x @ token_embed.T -> [B, T, V]
        logits = x @ self.token_embed.t()
        return logits
    
    def generate(
            self, 
            input_ids: torch.Tensor, 
            max_new_tokens: int, 
            eos_token_id: int=None
    ) -> torch.Tensor:
        """Auto-regressive generation
        Args:
            input_ids: torch.Tensor of shape [batch size, sequence length]
            max_new_tokens: int, number of tokens to generate
            eos_token_id: int, token ID for end-of-sequence (optional)
        Returns:
            generated_ids: torch.Tensor of shape [batch size, sequence length + max_new_tokens]
        """
        B, T = input_ids.shape
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # get logits for the current sequence
            logits = self(generated_ids)  # [B, T', V], T' is current sequence length

            # focus on the last time step
            logits = logits[:, -1, :]  # [B, V]

            # sample from the distribution
            probs = torch.softmax(logits, dim=-1)  # [B, V]
            next_token_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # append sampled token to the sequence
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)  # [B, T'+1]

            # if eos_token_id is specified, check if any sequence has generated it
            if eos_token_id is not None:
                if (next_token_ids == eos_token_id).any():
                    break

        return generated_ids
    