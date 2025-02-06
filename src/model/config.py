from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layer: int = 6         # Increased layers for better hierarchical learning
    n_head: int = 8         # More heads for better parallel attention
    n_embd: int = 384       # Larger embedding for richer representations
    block_size: int = 128    # Increased context window
    
    # Dropout and regularization
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    
    # Training
    learning_rate: float = 3e-4  # Slightly lower for larger model
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Dataset
    batch_size: int = 48     # Increased for better gradient estimates
    eval_interval: int = 500
    eval_iters: int = 50
    
    # Misc
    dtype: str = 'float32'
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def __post_init__(self):
        # Calculate approximate parameter count
        n_params = (
            self.vocab_size * self.n_embd +  # token embedding
            self.block_size * self.n_embd +  # position embedding
            self.n_layer * (
                4 * self.n_embd * self.n_embd +  # attention weights
                4 * self.n_embd * self.n_embd +  # mlp weights
                8 * self.n_embd  # layer norms
            )
        )
        print(f"\nModel Configuration:")
        print(f"- Parameters: {n_params/1e6:.2f}M")
        print(f"- Layers: {self.n_layer}")
        print(f"- Heads: {self.n_head}")
        print(f"- Embedding Dim: {self.n_embd}")
        print(f"- Context Length: {self.block_size}")
        print(f"- Device: {self.device}")
        print(f"- Batch Size: {self.batch_size}")
        print(f"- Learning Rate: {self.learning_rate}")
        print(f"- Gradient Clipping: {self.grad_clip}") 