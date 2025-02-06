from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken
import numpy as np

class TextDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        block_size: int = 128,
        batch_size: int = 32,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1"
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        
        # Load dataset
        print(f"Loading {dataset_name} ({dataset_config}) - {split} split...")
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        # Initialize tokenizer (using GPT-2 tokenizer)
        self.encoder = tiktoken.get_encoding("gpt2")
        
        # Tokenize all texts and concatenate
        all_tokens = []
        for item in dataset:
            if isinstance(item, dict):
                text = item['text']
            else:
                text = item
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue
            tokens = self.encoder.encode(text)
            if len(tokens) > 0:
                all_tokens.extend(tokens)
                all_tokens.append(self.encoder.eot_token)  # Add EOT token between texts
        
        # Convert to tensor
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # Calculate number of complete batches we can create
        n = len(self.tokens)
        if n < self.block_size + 1:
            raise ValueError(
                f"Dataset too small. Need at least {self.block_size + 1} tokens, "
                f"but only got {n}"
            )
        
        # Calculate number of sequences that will make complete batches
        self.n_sequences = ((n - self.block_size) // self.batch_size) * self.batch_size
        print(f"Dataset has {n:,} tokens")
        print(f"Creating {self.n_sequences:,} sequences in {self.n_sequences // self.batch_size:,} batches")
        print(f"Each sequence has length {self.block_size}")
        
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get a sequence starting at idx
        x = self.tokens[idx:idx + self.block_size]
        # Target is the sequence shifted by 1
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y 