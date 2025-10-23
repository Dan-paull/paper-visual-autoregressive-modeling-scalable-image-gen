# Visual Autoregressive Modeling (VAR)

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS%202024-Best%20Paper%20Award-gold)](https://arxiv.org/abs/2404.02905)
[![Paper](https://img.shields.io/badge/arXiv-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction**, the NeurIPS 2024 Best Paper Award winner.

## Overview

VAR reimagines autoregressive image generation by introducing **next-scale prediction** instead of traditional raster-scan next-token prediction. This breakthrough approach generates images progressively from coarse to fine resolutions, achieving:

- **20× faster inference** than diffusion models
- **State-of-the-art quality**: FID of 1.73 vs 18.65 for baseline AR models
- **GPT-style scaling laws**: Power-law relationship with model size
- **Zero-shot generalization**: Works on unseen tasks like inpainting and outpainting

## Key Innovation

Traditional autoregressive models predict tokens in raster-scan order (left-to-right, top-to-bottom). VAR instead generates images hierarchically:

```
Scale 1: 1×1   →  1 token     (total: 1)
Scale 2: 2×2   →  4 tokens    (total: 5)
Scale 3: 3×3   →  9 tokens    (total: 14)
...
Scale N: N×N   →  N² tokens   (total: N(N+1)(2N+1)/6)
```

This **coarse-to-fine** approach naturally captures both global structure and fine details.

## Architecture

The implementation consists of three main components:

### 1. VQVAE Tokenizer (`src/var/vqvae.py`)
- Encodes images into discrete tokens using vector quantization
- Downsamples images by 4× (e.g., 64×64 → 16×16 tokens)
- Configurable codebook size and embedding dimensions

### 2. VAR Model (`src/var/var_model.py`)
- Transformer-based autoregressive model with Adaptive Layer Normalization (AdaLN)
- Generates token sequences scale-by-scale
- Supports conditional generation with class labels
- Implements top-k and nucleus (top-p) sampling

### 3. Training Infrastructure (`src/var/trainer.py`)
- Efficient training loop with gradient clipping
- Checkpoint saving/loading
- Sample generation during training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paper-visual-autoregressive-modeling:-scalable-image-gen.git
cd paper-visual-autoregressive-modeling:-scalable-image-gen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Generate Images

```python
import torch
from src.var import VAR, VQVAE

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQVAE(num_embeddings=512).to(device)
var = VAR(
    vocab_size=512,
    d_model=256,
    n_layers=6,
    n_heads=8,
    max_scale=8
).to(device)

# Generate images autoregressively
tokens = var.generate(
    batch_size=4,
    temperature=1.0,
    top_k=100,
    top_p=0.95,
    device=device
)

# Decode to images
final_scale_tokens = tokens[:, -64:].reshape(4, 8, 8)
images = vqvae.decode_from_indices(final_scale_tokens)
```

### Run Examples

```bash
# Simple generation example
python examples/generate_simple.py

# Full generation with visualization (requires matplotlib)
python examples/generate.py
```

### Run Tests

```bash
# Test VQVAE
python tests/test_vqvae.py

# Test VAR model
python tests/test_var.py
```

## Model Configuration

### Small Model (Demo)
```python
var = VAR(
    vocab_size=512,
    d_model=256,
    n_layers=6,
    n_heads=8,
    d_ff=1024,
    max_scale=8
)
# Parameters: ~6.6M
```

### Medium Model
```python
var = VAR(
    vocab_size=4096,
    d_model=512,
    n_layers=12,
    n_heads=8,
    d_ff=2048,
    max_scale=16
)
# Parameters: ~50M
```

### Large Model (Paper Configuration)
```python
var = VAR(
    vocab_size=4096,
    d_model=768,
    n_layers=24,
    n_heads=12,
    d_ff=3072,
    max_scale=16
)
# Parameters: ~300M
```

## Training

```python
from torch.utils.data import DataLoader
from src.var.trainer import VARTrainer

# Create trainer
trainer = VARTrainer(var, vqvae, optimizer, device)

# Training loop
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}")

    # Save checkpoint
    if epoch % 10 == 0:
        trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

    # Generate samples
    if epoch % 5 == 0:
        samples = trainer.generate_samples(num_samples=8)
```

## Project Structure

```
.
├── src/
│   └── var/
│       ├── __init__.py           # Package initialization
│       ├── vqvae.py              # Vector Quantized VAE tokenizer
│       ├── var_model.py          # VAR transformer model
│       ├── trainer.py            # Training utilities
│       └── utils.py              # Helper functions
├── tests/
│   ├── test_vqvae.py             # VQVAE tests
│   └── test_var.py               # VAR model tests
├── examples/
│   ├── generate.py               # Image generation example
│   └── generate_simple.py        # Simple generation demo
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Technical Details

### Multi-Scale Sequence Length

For a maximum scale of N, the total sequence length is:
```
L = Σ(i²) for i=1 to N = N(N+1)(2N+1)/6
```

Examples:
- max_scale=8:  L = 204 tokens
- max_scale=16: L = 1,496 tokens
- max_scale=32: L = 11,440 tokens

### Attention Mechanism

The model uses causal attention within each scale and across scales:
```python
# Token at position j can attend to all tokens at positions i ≤ j
mask[i, j] = (i <= j)
```

This ensures autoregressive generation while allowing parallel processing during training.

### Sampling Strategies

1. **Temperature Scaling**: Controls randomness (higher = more diverse)
2. **Top-k Filtering**: Keeps only k most likely tokens
3. **Nucleus (top-p) Sampling**: Keeps tokens with cumulative probability ≥ p

## Results

Our implementation demonstrates the core VAR concepts:

- ✅ Multi-scale autoregressive generation
- ✅ Coarse-to-fine image synthesis
- ✅ Transformer architecture with AdaLN
- ✅ Top-k and nucleus sampling
- ✅ Conditional generation support
- ✅ Efficient training infrastructure

Note: This is an educational implementation. For production use and pretrained models, see the [official repository](https://github.com/FoundationVision/VAR).

## Paper Citation

```bibtex
@inproceedings{tian2024var,
  title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction},
  author={Tian, Keyu and Jiang, Yi and Yuan, Zehuan and Peng, Bingyue and Wang, Liwei},
  booktitle={NeurIPS},
  year={2024}
}
```

## Comparison with Other Approaches

| Approach | Token Order | Speed | Quality | Scaling Laws |
|----------|-------------|-------|---------|--------------|
| Traditional AR | Raster-scan | Slow | Good | Limited |
| Diffusion | N/A | Very Slow | Excellent | Some |
| **VAR** | **Multi-scale** | **Fast** | **Excellent** | **Strong** |

## Key Advantages

1. **Natural Hierarchy**: Coarse-to-fine matches human perception
2. **Faster Generation**: Fewer steps than diffusion
3. **Better Scaling**: Power-law relationship with model size
4. **Flexible**: Works for various generative tasks

## Limitations

This implementation:
- Uses smaller models than the paper (for accessibility)
- Doesn't include all optimizations (flash attention, etc.)
- Requires pretrained VQVAE for best results
- Limited to square images

## Future Work

- [ ] Pretrain VQVAE on large datasets
- [ ] Implement classifier-free guidance
- [ ] Add image editing capabilities
- [ ] Support non-square images
- [ ] Integrate flash attention for speed
- [ ] Multi-GPU training support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper authors for the groundbreaking VAR approach
- [Official VAR implementation](https://github.com/FoundationVision/VAR)
- PyTorch team for the excellent framework

## Contact

For questions or discussions about this implementation, please open an issue on GitHub.

---

**Note**: This is an educational reimplementation created to understand and demonstrate the VAR methodology. For research and production use, please refer to the official implementation.
