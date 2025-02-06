from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

class LMOrderedDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        block_size: int = 256,
        batch_size: int = 64,
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
        
        # Estimate total tokens for progress bar
        estimated_tokens = sum(len(str(item['text'])) * 0.3 for item in dataset)  # rough estimate
        print(f"Estimated tokens: {int(estimated_tokens):,}")
        
        # Tokenize all texts with progress bar
        print("Tokenizing texts...")
        all_tokens = []
        for item in tqdm(dataset, desc="Tokenizing"):
            text = item['text'] if isinstance(item, dict) else item
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue
            tokens = self.encoder.encode(text)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(self.encoder.eot_token)
        
        # Convert to tensor efficiently
        print("Converting to tensor...")
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
        total_tokens = tokens_tensor.size(0)
        print(f"Total tokens before trimming: {total_tokens:,}")
        
        # Trim tokens efficiently
        trimmed_length = (total_tokens // batch_size) * batch_size
        tokens_tensor = tokens_tensor[:trimmed_length]
        total_tokens = tokens_tensor.size(0)
        print(f"Total tokens after trimming: {total_tokens:,}")
        
        # Reshape and store in pinned memory if CUDA is available
        self.tokens = tokens_tensor.view(batch_size, -1)
        if torch.cuda.is_available():  # Only use pin_memory for CUDA devices
            self.tokens = self.tokens.pin_memory()
        print(f"Reshaped tokens to {self.tokens.size()}")
        
        # Calculate valid indices once
        self.valid_indices = self.tokens.size(1) - self.block_size
        
    def __len__(self) -> int:
        return self.valid_indices
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence at idx efficiently
        x = self.tokens[:, idx:idx + self.block_size].clone()  # clone for safety with pin_memory
        y = self.tokens[:, idx + 1:idx + self.block_size + 1].clone()
        return x, y 