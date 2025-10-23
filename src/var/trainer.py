"""
Training utilities for VAR model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import os


class VARTrainer:
    """
    Trainer class for VAR model.

    Handles training loop, loss computation, and checkpointing.
    """

    def __init__(
        self,
        var_model: nn.Module,
        vqvae_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Args:
            var_model: VAR model to train
            vqvae_model: Pre-trained VQVAE for tokenization
            optimizer: Optimizer for training
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.var_model = var_model.to(device)
        self.vqvae_model = vqvae_model.to(device)
        self.vqvae_model.eval()  # VQVAE is frozen
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.step = 0
        self.epoch = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary of training metrics
        """
        self.var_model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device) if labels is not None else None

            # Tokenize images using VQVAE
            with torch.no_grad():
                tokens = self.vqvae_model.get_codebook_indices(images)
                # Flatten to sequence: [B, H, W] -> [B, H*W]
                tokens = tokens.reshape(tokens.shape[0], -1)

            # Forward pass
            logits, loss = self.var_model(tokens, labels, return_loss=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.var_model.parameters(),
                max_grad_norm
            )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        self.epoch += 1

        return {
            "loss": total_loss / num_batches,
            "epoch": self.epoch,
            "step": self.step
        }

    def save_checkpoint(self, filename: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            filename: Optional checkpoint filename
        """
        if filename is None:
            filename = f"var_checkpoint_epoch_{self.epoch}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.var_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.var_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.epoch}, step {self.step}")

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 4,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 100,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Generate sample images.

        Args:
            num_samples: Number of samples to generate
            class_labels: Optional class labels
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling parameter

        Returns:
            Generated images [num_samples, 3, H, W]
        """
        self.var_model.eval()

        # Generate token sequences
        tokens = self.var_model.generate(
            batch_size=num_samples,
            class_labels=class_labels,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device
        )

        # Decode tokens to images
        # Reshape tokens back to 2D grid
        # Assuming tokens represent a downsampled grid (e.g., 16x16)
        grid_size = int(tokens.shape[1] ** 0.5)
        if grid_size * grid_size != tokens.shape[1]:
            # For multi-scale, use the final scale
            # This is a simplification - in practice, need to handle multi-scale properly
            grid_size = self.var_model.max_scale
            tokens_2d = tokens[:, -grid_size*grid_size:].reshape(-1, grid_size, grid_size)
        else:
            tokens_2d = tokens.reshape(-1, grid_size, grid_size)

        # Decode from VQVAE
        images = self.vqvae_model.decode_from_indices(tokens_2d)

        return images


def create_trainer(
    vocab_size: int = 4096,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    d_ff: int = 2048,
    max_scale: int = 8,
    num_classes: int = 10,
    learning_rate: float = 1e-4,
    device: torch.device = None
) -> VARTrainer:
    """
    Create a VAR trainer with default configuration.

    Args:
        vocab_size: VQVAE vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_scale: Maximum scale for multi-scale generation
        num_classes: Number of classes for conditional generation
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        Configured VARTrainer instance
    """
    from .var_model import VAR
    from .vqvae import VQVAE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    vqvae = VQVAE(num_embeddings=vocab_size)
    var = VAR(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_scale=max_scale,
        num_classes=num_classes
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(var.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    # Create trainer
    trainer = VARTrainer(var, vqvae, optimizer, device)

    return trainer
