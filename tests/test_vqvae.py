"""
Tests for VQVAE tokenizer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.var import VQVAE


def test_vqvae_forward():
    """Test VQVAE forward pass."""
    print("Testing VQVAE forward pass...")

    batch_size = 2
    image_size = 64
    vocab_size = 256

    # Create model
    vqvae = VQVAE(
        in_channels=3,
        hidden_channels=64,
        num_embeddings=vocab_size,
        embedding_dim=128
    )

    # Create dummy input
    x = torch.randn(batch_size, 3, image_size, image_size) * 2 - 1  # [-1, 1]

    # Forward pass
    x_recon, vq_loss = vqvae(x)

    # Check shapes
    assert x_recon.shape == x.shape, f"Expected shape {x.shape}, got {x_recon.shape}"
    assert vq_loss.dim() == 0, "VQ loss should be scalar"

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {x_recon.shape}")
    print(f"  ✓ VQ loss: {vq_loss.item():.4f}")


def test_vqvae_encode_decode():
    """Test VQVAE encoding and decoding."""
    print("\nTesting VQVAE encode/decode...")

    batch_size = 2
    image_size = 64
    vocab_size = 256

    vqvae = VQVAE(num_embeddings=vocab_size)

    x = torch.randn(batch_size, 3, image_size, image_size)

    # Encode
    quantized, encodings = vqvae.encode(x)

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Quantized shape: {quantized.shape}")
    print(f"  ✓ Encodings shape: {encodings.shape}")

    # Check that encodings are in valid range
    assert encodings.min() >= 0, "Encodings should be non-negative"
    assert encodings.max() < vocab_size, f"Encodings should be < {vocab_size}"

    # Decode
    x_recon = vqvae.decode(quantized)

    assert x_recon.shape == x.shape, "Decoded shape should match input"
    print(f"  ✓ Decoded shape: {x_recon.shape}")


def test_vqvae_indices():
    """Test getting and using codebook indices."""
    print("\nTesting codebook indices...")

    batch_size = 2
    image_size = 64

    vqvae = VQVAE()

    x = torch.randn(batch_size, 3, image_size, image_size)

    # Get indices
    indices = vqvae.get_codebook_indices(x)

    print(f"  ✓ Indices shape: {indices.shape}")
    print(f"  ✓ Indices range: [{indices.min()}, {indices.max()}]")

    # Decode from indices
    x_recon = vqvae.decode_from_indices(indices)

    print(f"  ✓ Reconstructed shape: {x_recon.shape}")

    # Check shapes are consistent
    expected_h = image_size // 4  # Due to downsampling
    expected_w = image_size // 4
    assert indices.shape == (batch_size, expected_h, expected_w)


def test_vqvae_downsampling():
    """Test that VQVAE downsamples by factor of 4."""
    print("\nTesting VQVAE downsampling factor...")

    image_size = 64
    vqvae = VQVAE()

    x = torch.randn(1, 3, image_size, image_size)

    quantized, encodings = vqvae.encode(x)

    # Check downsampling factor
    expected_size = image_size // 4
    assert quantized.shape[2] == expected_size, f"Expected size {expected_size}, got {quantized.shape[2]}"
    assert quantized.shape[3] == expected_size, f"Expected size {expected_size}, got {quantized.shape[3]}"

    print(f"  ✓ Input size: {image_size}x{image_size}")
    print(f"  ✓ Encoded size: {quantized.shape[2]}x{quantized.shape[3]}")
    print(f"  ✓ Downsampling factor: 4")


if __name__ == "__main__":
    print("="*60)
    print("VQVAE TESTS")
    print("="*60)

    test_vqvae_forward()
    test_vqvae_encode_decode()
    test_vqvae_indices()
    test_vqvae_downsampling()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
