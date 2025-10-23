"""
Vector Quantized Variational AutoEncoder (VQVAE)

This module implements the VQVAE used for tokenizing images into discrete representations
that are then processed by the VAR model in a multi-scale manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for discretizing continuous representations.

    Args:
        num_embeddings: Size of the codebook (vocabulary size)
        embedding_dim: Dimension of each codebook entry
        commitment_cost: Weight for the commitment loss
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            quantized: Quantized tensor with straight-through gradient
            vq_loss: Vector quantization loss
            encodings: Discrete token indices [B, H, W]
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape
        flat_x = x.view(-1, self.embedding_dim)  # [B*H*W, C]

        # Calculate distances to codebook entries
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )

        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # [B*H*W]

        # Quantize and unflatten
        quantized = self.embeddings(encoding_indices).view(input_shape)  # [B, H, W, C]

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Reshape back: [B, H, W, C] -> [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encodings = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

        return quantized, vq_loss, encodings


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class VQVAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder for image tokenization.

    This model encodes images into discrete tokens that can be processed
    autoregressively by the VAR model.

    Args:
        in_channels: Number of input channels (3 for RGB)
        hidden_channels: Number of hidden channels in encoder/decoder
        num_embeddings: Size of the codebook
        embedding_dim: Dimension of codebook entries
        num_res_blocks: Number of residual blocks in encoder/decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_embeddings: int = 4096,
        embedding_dim: int = 256,
        num_res_blocks: int = 2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Encoder: progressively downsample the image
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, embedding_dim, 3, padding=1)
        )

        # Vector quantizer
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: progressively upsample back to image
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, in_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to discrete tokens.

        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            quantized: Quantized features [B, embedding_dim, H/4, W/4]
            encodings: Discrete token indices [B, H/4, W/4]
        """
        z = self.encoder(x)
        quantized, vq_loss, encodings = self.quantizer(z)
        return quantized, encodings

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features back to images.

        Args:
            quantized: Quantized features [B, embedding_dim, H/4, W/4]

        Returns:
            x_recon: Reconstructed images [B, 3, H, W] in range [-1, 1]
        """
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            x_recon: Reconstructed images
            vq_loss: Vector quantization loss
        """
        z = self.encoder(x)
        quantized, vq_loss, encodings = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get discrete token indices for input images.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            indices: Token indices [B, H/4, W/4]
        """
        with torch.no_grad():
            _, encodings = self.encode(x)
        return encodings

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode images from discrete token indices.

        Args:
            indices: Token indices [B, H, W]

        Returns:
            x_recon: Reconstructed images [B, 3, H*4, W*4]
        """
        # Get embeddings from codebook
        quantized = self.quantizer.embeddings(indices)  # [B, H, W, C]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return self.decoder(quantized)
