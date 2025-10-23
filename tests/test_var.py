"""
Tests for VAR model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.var import VAR, get_multi_scale_patches


def test_var_initialization():
    """Test VAR model initialization."""
    print("Testing VAR initialization...")

    vocab_size = 512
    d_model = 256
    n_layers = 4
    n_heads = 8
    max_scale = 8

    var = VAR(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_scale=max_scale
    )

    # Check attributes
    assert var.vocab_size == vocab_size
    assert var.d_model == d_model
    assert var.max_scale == max_scale

    # Check sequence length calculation
    expected_seq_len = sum(s*s for s in range(1, max_scale + 1))
    assert var.total_seq_len == expected_seq_len

    print(f"  ✓ Vocab size: {vocab_size}")
    print(f"  ✓ Model dimension: {d_model}")
    print(f"  ✓ Max scale: {max_scale}")
    print(f"  ✓ Total sequence length: {var.total_seq_len}")

    # Count parameters
    params = var.count_parameters()
    print(f"  ✓ Parameters: {params:,}")


def test_var_forward():
    """Test VAR forward pass."""
    print("\nTesting VAR forward pass...")

    batch_size = 2
    seq_len = 64
    vocab_size = 512
    num_classes = 10

    var = VAR(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_scale=16,
        num_classes=num_classes
    )

    # Create dummy input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    class_labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    logits, loss = var(tokens, class_labels, return_loss=True)

    # Check shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.dim() == 0

    print(f"  ✓ Input tokens shape: {tokens.shape}")
    print(f"  ✓ Output logits shape: {logits.shape}")
    print(f"  ✓ Loss: {loss.item():.4f}")


def test_var_generate():
    """Test VAR autoregressive generation."""
    print("\nTesting VAR generation...")

    batch_size = 2
    vocab_size = 256
    max_scale = 4  # Small for fast testing

    var = VAR(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_scale=max_scale
    )
    var.eval()

    # Generate
    generated = var.generate(
        batch_size=batch_size,
        temperature=1.0,
        top_k=50,
        top_p=0.9
    )

    # Check shape
    expected_len = sum(s*s for s in range(1, max_scale + 1))
    assert generated.shape == (batch_size, expected_len)

    # Check values are in valid range
    assert generated.min() >= 0
    assert generated.max() < vocab_size

    print(f"  ✓ Generated shape: {generated.shape}")
    print(f"  ✓ Token range: [{generated.min()}, {generated.max()}]")


def test_multi_scale_sequence():
    """Test multi-scale patch sequence."""
    print("\nTesting multi-scale sequence...")

    max_scale = 8
    scales = get_multi_scale_patches(max_scale)

    print(f"  ✓ Scales: {scales}")

    # Calculate cumulative tokens
    cumulative = 0
    for i, scale in enumerate(scales):
        tokens_at_scale = scale * scale
        cumulative += tokens_at_scale
        print(f"  ✓ Scale {scale}x{scale}: {tokens_at_scale} tokens (cumulative: {cumulative})")

    total = sum(s*s for s in scales)
    print(f"  ✓ Total sequence length: {total}")


def test_var_conditional():
    """Test VAR conditional generation."""
    print("\nTesting VAR conditional generation...")

    batch_size = 2
    vocab_size = 256
    num_classes = 10
    max_scale = 4

    var = VAR(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_scale=max_scale,
        num_classes=num_classes
    )
    var.eval()

    # Generate with class labels
    class_labels = torch.tensor([0, 5])
    generated = var.generate(
        batch_size=batch_size,
        class_labels=class_labels,
        temperature=1.0
    )

    print(f"  ✓ Class labels: {class_labels.tolist()}")
    print(f"  ✓ Generated shape: {generated.shape}")


def test_var_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting VAR gradient flow...")

    var = VAR(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_scale=4
    )

    tokens = torch.randint(0, 256, (2, 10))
    logits, loss = var(tokens, return_loss=True)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_grad = False
    for name, param in var.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found!"

    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Gradients flowing correctly")


if __name__ == "__main__":
    print("="*60)
    print("VAR MODEL TESTS")
    print("="*60)

    test_var_initialization()
    test_var_forward()
    test_var_generate()
    test_multi_scale_sequence()
    test_var_conditional()
    test_var_gradient_flow()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
