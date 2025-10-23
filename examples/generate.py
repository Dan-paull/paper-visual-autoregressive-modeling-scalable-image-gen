"""
Example script for generating images with VAR.

This demonstrates how to use the VAR model for autoregressive image generation
using next-scale prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.var import VAR, VQVAE


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """
    Convert normalized image tensor to displayable numpy array.

    Args:
        img: Image tensor [3, H, W] in range [-1, 1]

    Returns:
        Image array [H, W, 3] in range [0, 255]
    """
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def main():
    """Generate sample images using VAR."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    vocab_size = 512  # Smaller for demo
    d_model = 256
    n_layers = 6
    n_heads = 8
    d_ff = 1024
    max_scale = 8  # Generate 8x8 token grid -> 32x32 image (with 4x downsampling)
    num_classes = 10

    print("\nInitializing models...")
    print(f"  - VQVAE with vocab_size={vocab_size}")
    print(f"  - VAR with d_model={d_model}, n_layers={n_layers}, max_scale={max_scale}")

    # Create models
    vqvae = VQVAE(
        num_embeddings=vocab_size,
        embedding_dim=128,
        hidden_channels=64
    ).to(device)

    var = VAR(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_scale=max_scale,
        num_classes=num_classes
    ).to(device)

    # Print model info
    var_params = var.count_parameters()
    vqvae_params = sum(p.numel() for p in vqvae.parameters())
    print(f"\nModel Statistics:")
    print(f"  - VAR parameters: {var_params:,}")
    print(f"  - VQVAE parameters: {vqvae_params:,}")
    print(f"  - Total parameters: {var_params + vqvae_params:,}")

    # Generate images
    print("\nGenerating images...")
    num_samples = 4

    # Optional: specify class labels for conditional generation
    # class_labels = torch.tensor([0, 1, 2, 3], device=device)
    class_labels = None

    with torch.no_grad():
        # Generate token sequences autoregressively
        print(f"  - Generating {num_samples} samples with next-scale prediction...")
        tokens = var.generate(
            batch_size=num_samples,
            class_labels=class_labels,
            temperature=1.0,
            top_k=100,
            top_p=0.95,
            device=device
        )

        print(f"  - Generated token sequence shape: {tokens.shape}")

        # Decode tokens to images
        # Extract final scale tokens (max_scale x max_scale grid)
        final_scale_tokens = tokens[:, -max_scale*max_scale:].reshape(num_samples, max_scale, max_scale)

        print(f"  - Decoding tokens to images...")
        images = vqvae.decode_from_indices(final_scale_tokens)

    # Visualize results
    print("\nVisualizing generated images...")

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    if num_samples == 1:
        axes = [axes]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        img_np = denormalize_image(img)
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f"Sample {idx+1}")

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_samples.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGenerated images saved to: {output_path}")

    plt.show()

    # Demonstrate multi-scale generation process
    print("\n" + "="*60)
    print("MULTI-SCALE GENERATION EXPLAINED")
    print("="*60)
    print("\nVAR generates images progressively from coarse to fine:")
    print(f"  Scale 1: 1x1 patches   -> 1 token    (total: 1)")
    print(f"  Scale 2: 2x2 patches   -> 4 tokens   (total: 5)")
    print(f"  Scale 3: 3x3 patches   -> 9 tokens   (total: 14)")
    print(f"  ...")
    print(f"  Scale {max_scale}: {max_scale}x{max_scale} patches -> {max_scale*max_scale} tokens   (total: {var.total_seq_len})")
    print(f"\nThis is fundamentally different from traditional autoregressive")
    print(f"models that predict tokens in raster-scan order!")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
