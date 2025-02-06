import os
import torch
import tiktoken
from model.model import GPT
from model.config import GPTConfig
from torch.serialization import safe_globals

def generate(
    checkpoint_path: str,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device_type: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> str:
    # Load the checkpoint with the appropriate settings
    with safe_globals([GPTConfig]):
        checkpoint = torch.load(checkpoint_path, map_location=device_type, weights_only=False)
    config = checkpoint['config']
    
    # Initialize model
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device_type)
    model.eval()
    
    # Initialize tokenizer
    encoder = tiktoken.get_encoding("gpt2")
    
    # Encode the prompt
    if prompt:
        tokens = encoder.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device_type).unsqueeze(0)
    else:
        tokens = torch.zeros((1, 1), dtype=torch.long, device=device_type)
    
    # Generate
    with torch.no_grad():
        tokens = model.generate(tokens, max_new_tokens, temperature, top_k)
    
    # Decode
    output = encoder.decode(tokens[0].tolist())
    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--prompt', type=str, default="", help='prompt to start generation with')
    parser.add_argument('--max_tokens', type=int, default=100, help='number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='top k sampling')
    args = parser.parse_args()
    
    output = generate(
        args.checkpoint,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k
    )
    print(output) 