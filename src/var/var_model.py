"""
VAR (Visual Autoregressive) Model

Implements the core VAR architecture that performs next-scale prediction
for autoregressive image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .utils import (
    get_multi_scale_patches,
    positional_encoding_2d,
    create_causal_mask,
    top_k_top_p_filtering,
    compute_scale_positions
)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) for conditional generation.

    Modulates layer norm parameters based on conditioning information.
    """

    def __init__(self, d_model: int, condition_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(condition_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            Normalized and modulated tensor [B, L, D]
        """
        # Normalize
        x = self.ln(x)

        # Get scale and shift from condition
        scale_shift = self.linear(condition).unsqueeze(1)  # [B, 1, 2*D]
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Apply affine transformation
        return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    Uses AdaLN for conditioning.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        condition_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Adaptive layer norms
        self.adaLN1 = AdaptiveLayerNorm(d_model, condition_dim)
        self.adaLN2 = AdaptiveLayerNorm(d_model, condition_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            condition: Conditioning vector [B, condition_dim]
            attn_mask: Attention mask [L, L]

        Returns:
            Output tensor [B, L, D]
        """
        # Self-attention with AdaLN
        x_norm = self.adaLN1(x, condition)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Feed-forward with AdaLN
        x_norm = self.adaLN2(x, condition)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class VAR(nn.Module):
    """
    Visual Autoregressive Model for image generation.

    Generates images in a coarse-to-fine manner using next-scale prediction.

    Args:
        vocab_size: Size of the token vocabulary (from VQVAE)
        d_model: Transformer hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_scale: Maximum scale (patches per side, e.g., 16 for 16x16)
        num_classes: Number of class labels for conditional generation
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_scale: int = 16,
        num_classes: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_scale = max_scale
        self.num_classes = num_classes

        # Calculate total sequence length across all scales
        self.scales = get_multi_scale_patches(max_scale)
        self.total_seq_len = sum(s * s for s in self.scales)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Class embedding for conditional generation
        self.class_embedding = nn.Embedding(num_classes, d_model)

        # Scale embeddings (which scale we're at)
        self.scale_embedding = nn.Embedding(max_scale, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_seq_len, d_model))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, d_model, dropout)
            for _ in range(n_layers)
        ])

        # Output projection to vocabulary
        self.to_logits = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        nn.init.normal_(self.scale_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.to_logits.weight, std=0.02)
        nn.init.zeros_(self.to_logits.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training with teacher forcing.

        Args:
            tokens: Token indices [B, seq_len] where seq_len <= total_seq_len
            class_labels: Class labels for conditional generation [B]
            return_loss: Whether to compute and return loss

        Returns:
            logits: Predicted logits [B, seq_len, vocab_size]
            loss: Cross-entropy loss (if return_loss=True)
        """
        B, seq_len = tokens.shape
        device = tokens.device

        # Token embeddings
        x = self.token_embedding(tokens)  # [B, seq_len, d_model]

        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]

        # Create conditioning from class labels
        if class_labels is not None:
            condition = self.class_embedding(class_labels)  # [B, d_model]
        else:
            condition = torch.zeros(B, self.d_model, device=device)

        # Apply dropout
        x = self.dropout(x)

        # Create causal mask for autoregressive modeling
        causal_mask = create_causal_mask(seq_len, device)
        # Convert to attention mask format (True = attend, False = mask)
        attn_mask = ~causal_mask  # Invert for PyTorch convention

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, condition, attn_mask=attn_mask)

        # Project to vocabulary
        logits = self.to_logits(x)  # [B, seq_len, vocab_size]

        loss = None
        if return_loss:
            # Compute cross-entropy loss
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tokens = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_tokens.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 100,
        top_p: float = 0.95,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate images autoregressively using next-scale prediction.

        Args:
            batch_size: Number of images to generate
            class_labels: Optional class labels for conditional generation [batch_size]
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            device: Device to generate on

        Returns:
            Generated token sequences [batch_size, total_seq_len]
        """
        if device is None:
            device = next(self.parameters()).device

        # Initialize with empty sequence
        generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # Create conditioning
        if class_labels is not None:
            condition = self.class_embedding(class_labels)
        else:
            condition = torch.zeros(batch_size, self.d_model, device=device)

        # Generate scale by scale (coarse to fine)
        for scale_idx, scale in enumerate(self.scales):
            num_tokens_this_scale = scale * scale

            # Generate tokens for this scale
            for _ in range(num_tokens_this_scale):
                # Get current sequence length
                seq_len = generated.shape[1]

                if seq_len == 0:
                    # First token: use just the conditioning
                    x = condition.unsqueeze(1)  # [B, 1, d_model]
                    logits = self.to_logits(x[:, -1, :])  # [B, vocab_size]
                else:
                    # Token embeddings
                    x = self.token_embedding(generated)

                    # Add positional embeddings
                    x = x + self.pos_embedding[:, :seq_len, :]

                    # Apply transformer blocks
                    causal_mask = create_causal_mask(seq_len, device)
                    attn_mask = ~causal_mask

                    for block in self.transformer_blocks:
                        x = block(x, condition, attn_mask=attn_mask)

                    # Get logits for next token
                    logits = self.to_logits(x[:, -1, :])  # [B, vocab_size]

                # Apply temperature
                logits = logits / temperature

                # Apply top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

                # Sample from the filtered distribution
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
