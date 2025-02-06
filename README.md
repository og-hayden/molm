# Simple GPT Implementation

A PyTorch implementation of a GPT (Generative Pre-trained Transformer) model with approximately 1B parameters. This implementation includes training on the WikiText dataset and text generation capabilities.

## Features

- Transformer-based architecture with roughly 1B parameters
- Training on WikiText-103 dataset
- Efficient training with gradient clipping and weight decay
- Text generation with temperature and top-k sampling
- Checkpoint saving and loading
- Type hints and clean code structure

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── config.py      # Model configuration
│   │   └── model.py       # GPT model implementation
│   ├── data/
│   │   └── dataset.py     # Dataset loading and preprocessing
│   ├── train.py           # Training script
│   └── generate.py        # Text generation script
├── requirements.txt
└── README.md
```

## Training

To train the model:

```bash
python src/train.py
```

Training parameters can be modified in `src/model/config.py`.

The model will be trained on the WikiText-103 dataset. Checkpoints will be saved in the `out` directory.

## Text Generation

To generate text using a trained model:

```bash
python src/generate.py --checkpoint out/checkpoint.pt --prompt "Your prompt here" --max_tokens 100
```

Arguments:
- `--checkpoint`: Path to the model checkpoint
- `--prompt`: Text prompt to start generation (optional)
- `--max_tokens`: Number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling parameter (default: 200)

## Model Architecture

The model follows the GPT architecture with:
- 24 transformer layers
- 16 attention heads
- 2048 embedding dimension
- 1024 sequence length
- ~1B parameters

## License

MIT 