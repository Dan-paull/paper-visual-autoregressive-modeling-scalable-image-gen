"""
Utility functions for VAR multi-scale processing.
"""

import torch
import numpy as np
from typing import List, Tuple


def get_multi_scale_patches(max_patches: int = 16) -> List[int]:
    """
    Generate the sequence of patch numbers for multi-scale prediction.

    VAR generates images progressively from coarse to fine:
    1x1 -> 2x2 -> 3x3 -> ... -> max_patches x max_patches

    Args:
        max_patches: Maximum number of patches per side (e.g., 16 for 16x16 grid)

    Returns:
        List of patch counts at each scale
    """
    return list(range(1, max_patches + 1))


def rearrange_for_scale(tokens: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Rearrange tokens for a specific scale in the multi-scale hierarchy.

    Args:
        tokens: Token tensor [B, seq_len]
        scale: Current scale (number of patches per side)

    Returns:
        Rearranged tokens for the scale [B, scale*scale]
    """
    B = tokens.shape[0]
    num_tokens = scale * scale
    return tokens[:, :num_tokens].reshape(B, scale, scale)


def positional_encoding_2d(h: int, w: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Generate 2D positional encodings for spatial positions.

    Args:
        h: Height dimension
        w: Width dimension
        d_model: Model dimension
        device: Device to create tensor on

    Returns:
        Positional encoding tensor [h, w, d_model]
    """
    # Create position indices
    y_pos = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)  # [h, w]
    x_pos = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)  # [h, w]

    # Normalize positions to [0, 1]
    y_pos = y_pos.float() / max(h - 1, 1)
    x_pos = x_pos.float() / max(w - 1, 1)

    # Generate sinusoidal encodings
    pe = torch.zeros(h, w, d_model, device=device)

    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                         (-np.log(10000.0) / d_model))

    # Half the dimensions encode y position
    pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
    pe[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)

    return pe


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask [seq_len, seq_len] with True where attention is allowed
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: Logits tensor [batch_size, vocab_size]
        top_k: Keep only top k tokens with highest probability (0 = no filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (1.0 = no filtering)
        filter_value: Value to set filtered logits to

    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def compute_scale_positions(max_scale: int = 16) -> Tuple[List[int], List[int]]:
    """
    Compute start and end positions for each scale in the flattened sequence.

    Args:
        max_scale: Maximum scale (patches per side)

    Returns:
        start_positions: Starting index for each scale
        end_positions: Ending index for each scale
    """
    scales = get_multi_scale_patches(max_scale)
    start_positions = []
    end_positions = []

    cumulative = 0
    for scale in scales:
        num_tokens = scale * scale
        start_positions.append(cumulative)
        cumulative += num_tokens
        end_positions.append(cumulative)

    return start_positions, end_positions


def scale_to_patches(scale_idx: int) -> int:
    """
    Convert scale index to number of patches per side.

    Args:
        scale_idx: Index in the scale sequence (0-indexed)

    Returns:
        Number of patches per side at this scale
    """
    return scale_idx + 1
