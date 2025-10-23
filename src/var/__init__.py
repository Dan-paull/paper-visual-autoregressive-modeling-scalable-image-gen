"""
Visual Autoregressive Modeling (VAR)
NeurIPS 2024 Best Paper Award

This package implements VAR: a novel autoregressive image generation approach
that uses next-scale prediction instead of traditional next-token prediction.
"""

from .vqvae import VQVAE
from .var_model import VAR
from .utils import get_multi_scale_patches

__version__ = "0.1.0"
__all__ = ["VQVAE", "VAR", "get_multi_scale_patches"]
