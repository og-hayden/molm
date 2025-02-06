import os
import time
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.serialization import safe_globals

from model.config import GPTConfig
from model.model import GPT
from data.lm_ordered_dataset import LMOrderedDataset

class TrainingLogger:
    def __init__(self, log_interval: int, config: GPTConfig):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.losses = []
        self.running_loss = 0.0
        self.best_loss = float('inf')
        self.config = config
        
    def log_step(self, iter_num: int, loss: float, val_loss: Optional[float] = None):
        self.running_loss += loss
        self.losses.append(loss)
        
        if iter_num % self.log_interval == 0:
            avg_loss = self.running_loss / self.log_interval
            time_elapsed = time.time() - self.start_time
            tokens_processed = iter_num * self.config.batch_size * self.config.block_size
            tokens_per_sec = tokens_processed / time_elapsed if time_elapsed > 0 else 0
            
            status = [
                f"iter {iter_num:,}",
                f"loss {avg_loss:.4f}",
                f"time {time_elapsed:.1f}s",
                f"tokens/sec {tokens_per_sec:.0f}",
            ]
            
            if val_loss is not None:
                status.append(f"val_loss {val_loss:.4f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    status.append("(best)")
            
            print(" | ".join(status))
            self.running_loss = 0.0

def train(
    # model/data params
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    out_dir: str = "out",
    checkpoint_path: Optional[str] = None,
    eval_interval: int = 500,
    log_interval: int = 50,
    eval_iters: int = 200,  # Increased for more stable evaluation
    eval_only: bool = False,
    always_save_checkpoint: bool = True,
    # adamw optimizer
    learning_rate: float = 1e-4,  # Lower learning rate for stability
    max_iters: int = 100000,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    grad_clip: float = 1.0,
    # system
    device_type: str = "mps" if torch.backends.mps.is_available() else "cpu",
    dtype: str = "float32",
    compile: bool = False,  # Disable compile for MPS
) -> None:
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs will be saved to: {out_dir}")
    
    # Initialize model and optimizer
    if checkpoint_path is not None:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        with safe_globals([GPTConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device_type, weights_only=False)
        config = checkpoint['config']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
        # Update config with optimized parameters
        config.n_layer = 8  # Increased from 6 to 8 layers
        config.n_head = 16  # Increased from 8 to 16 heads
        config.n_embd = 512  # Increased from 384 to 512
        config.block_size = 256  # Increased context length
        config.batch_size = 32  # Reduced for better fitting in memory
        config.learning_rate = learning_rate
        config.weight_decay = weight_decay
        config.beta1 = beta1
        config.beta2 = beta2
        config.grad_clip = grad_clip
        config.eval_interval = eval_interval
        config.eval_iters = eval_iters
        config.device = device_type
        config.dtype = dtype
        # Increased dropout for regularization
        config.embd_pdrop = 0.2
        config.resid_pdrop = 0.2
        config.attn_pdrop = 0.2
    else:
        config = GPTConfig()
        # Set optimized parameters for new training
        config.n_layer = 8
        config.n_head = 16
        config.n_embd = 512
        config.block_size = 256
        config.batch_size = 32
        config.learning_rate = learning_rate
        config.weight_decay = weight_decay
        config.beta1 = beta1
        config.beta2 = beta2
        config.grad_clip = grad_clip
        config.eval_interval = eval_interval
        config.eval_iters = eval_iters
        config.device = device_type
        config.dtype = dtype
        config.embd_pdrop = 0.2
        config.resid_pdrop = 0.2
        config.attn_pdrop = 0.2
        iter_num = 0
        best_val_loss = float('inf')
    
    model = GPT(config)
    if checkpoint_path is not None:
        model.load_state_dict(checkpoint['model'])
    model.to(device_type)
    print(f"\nModel initialized on {device_type}")
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Initialize datasets using LMOrderedDataset
    print("\nLoading datasets...")
    train_dataset = LMOrderedDataset(
        split="train",
        block_size=config.block_size,
        batch_size=config.batch_size,
        dataset_name=dataset_name,
        dataset_config=dataset_config
    )
    val_dataset = LMOrderedDataset(
        split="validation",
        block_size=config.block_size,
        batch_size=config.batch_size,
        dataset_name=dataset_name,
        dataset_config=dataset_config
    )
    print(f"Train size: {len(train_dataset):,} batches")
    print(f"Val size: {len(val_dataset):,} batches")
    
    # DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,  # Increased for better data loading
        pin_memory=True if device_type == "mps" else False,
        drop_last=True,
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device_type == "mps" else False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Training loop
    print("\nStarting training loop...")
    logger = TrainingLogger(log_interval, config)
    
    # Pre-compute validation indices for faster evaluation
    val_indices = torch.randperm(len(val_dataset))[:config.eval_iters]
    
    try:
        progress_bar = tqdm(total=max_iters, desc="Training", unit="iter")
        model.train()
        
        while iter_num < max_iters:
            for batch_idx, (x, y) in enumerate(train_loader):
                if iter_num >= max_iters:
                    break
                
                # Process batch
                x = x.squeeze(0).to(device_type, non_blocking=True)
                y = y.squeeze(0).to(device_type, non_blocking=True)
                
                # Forward and backward passes
                logits, loss = model(x, y)
                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                # Logging with less frequent updates
                logger.log_step(iter_num, loss.item())
                
                # Evaluation
                if iter_num > 0 and iter_num % eval_interval == 0:
                    model.eval()
                    val_loss = evaluate(model, val_loader, device_type, val_indices)
                    logger.log_step(iter_num, loss.item(), val_loss)
                    model.train()
                    
                    # Save best model
                    if val_loss < logger.best_loss or always_save_checkpoint:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'iter_num': iter_num,
                            'best_val_loss': val_loss,
                        }
                        checkpoint_path = os.path.join(out_dir, f'checkpoint_{iter_num:07d}.pt')
                        print(f"\nSaving checkpoint to {checkpoint_path}")
                        torch.save(checkpoint, checkpoint_path)
                
                iter_num += 1
                progress_bar.update(1)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        progress_bar.close()
        # Save final checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter_num': iter_num,
            'best_val_loss': logger.best_loss,
        }
        checkpoint_path = os.path.join(out_dir, f'checkpoint_final.pt')
        print(f"\nSaving final checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best validation loss: {logger.best_loss:.4f}")

def evaluate(model: nn.Module, val_loader: DataLoader, device_type: str, val_indices: torch.Tensor) -> float:
    total_loss = 0.0
    
    with torch.no_grad():
        for idx in val_indices:
            x, y = val_loader.dataset[idx.item()]
            x = x.to(device_type, non_blocking=True)
            y = y.to(device_type, non_blocking=True)
            _, loss = model(x, y)
            total_loss += loss.item()
    
    return total_loss / len(val_indices)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum iterations')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu', help='Device to use (mps/cpu)')
    args = parser.parse_args()
    
    train(
        checkpoint_path=args.checkpoint,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        device_type=args.device
    ) 