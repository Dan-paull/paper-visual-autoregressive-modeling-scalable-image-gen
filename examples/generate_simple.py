"""
Simple example script for generating images with VAR (no plotting).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.var import VAR, VQVAE


def main():
    """Generate sample images using VAR."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    vocab_size = 512
    d_model = 256
    n_layers = 6
    n_heads = 8
    d_ff = 1024
    max_scale = 4  # Smaller for faster demo
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
    num_samples = 2

    with torch.no_grad():
        print(f"  - Generating {num_samples} samples with next-scale prediction...")
        tokens = var.generate(
            batch_size=num_samples,
            class_labels=None,
            temperature=1.0,
            top_k=100,
            top_p=0.95,
            device=device
        )

        print(f"  - Generated token sequence shape: {tokens.shape}")

        # Extract final scale tokens
        final_scale_tokens = tokens[:, -max_scale*max_scale:].reshape(num_samples, max_scale, max_scale)

        print(f"  - Decoding tokens to images...")
        images = vqvae.decode_from_indices(final_scale_tokens)
        print(f"  - Generated images shape: {images.shape}")

    print("\n" + "="*60)
    print("MULTI-SCALE GENERATION EXPLAINED")
    print("="*60)
    print("\nVAR generates images progressively from coarse to fine:")
    print(f"  Scale 1: 1x1 patches   -> 1 token    (total: 1)")
    print(f"  Scale 2: 2x2 patches   -> 4 tokens   (total: 5)")
    print(f"  Scale 3: 3x3 patches   -> 9 tokens   (total: 14)")
    print(f"  Scale 4: 4x4 patches   -> 16 tokens  (total: 30)")
    print(f"\nTotal tokens generated: {var.total_seq_len}")
    print(f"\nThis is fundamentally different from traditional autoregressive")
    print(f"models that predict tokens in raster-scan order!")
    print("="*60)

    print("\n✅ Image generation successful!")


if __name__ == "__main__":
    main()
