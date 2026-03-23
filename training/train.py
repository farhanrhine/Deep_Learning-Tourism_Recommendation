"""
Training Pipeline for the Tourism Recommendation System.
Supports training NCF, Content-Based, and Hybrid models with
early stopping, learning rate scheduling, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.logger import get_project_logger
from utils.custom_exception import ModelTrainingError
from utils.config import (
    LEARNING_RATE, EPOCHS, EARLY_STOP_PATIENCE, MODEL_DIR
)

logger = get_project_logger(__name__)


class Trainer:
    """
    Handles model training with early stopping, LR scheduling, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "hybrid",
        lr: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        patience: int = EARLY_STOP_PATIENCE,
        device: str | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            user_ids = batch["user_id"].to(self.device)
            dest_ids = batch["dest_id"].to(self.device)
            user_feats = batch["user_features"].to(self.device)
            dest_feats = batch["dest_features"].to(self.device)
            ratings = batch["rating"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward (supports all model types)
            if hasattr(self.model, "ncf") and hasattr(self.model, "content"):
                # Hybrid model
                predictions = self.model(user_ids, dest_ids, user_feats, dest_feats)
            elif hasattr(self.model, "user_embedding_gmf"):
                # NCF model
                predictions = self.model(user_ids, dest_ids)
            else:
                # Content model
                predictions = self.model(user_feats, dest_feats)

            loss = self.criterion(predictions, ratings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            user_ids = batch["user_id"].to(self.device)
            dest_ids = batch["dest_id"].to(self.device)
            user_feats = batch["user_features"].to(self.device)
            dest_feats = batch["dest_features"].to(self.device)
            ratings = batch["rating"].to(self.device).unsqueeze(1)

            if hasattr(self.model, "ncf") and hasattr(self.model, "content"):
                predictions = self.model(user_ids, dest_ids, user_feats, dest_feats)
            elif hasattr(self.model, "user_embedding_gmf"):
                predictions = self.model(user_ids, dest_ids)
            else:
                predictions = self.model(user_feats, dest_feats)

            loss = self.criterion(predictions, ratings)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict:
        """
        Full training loop with early stopping and checkpointing.
        
        Returns:
            Dictionary with training history and best metrics.
        """
        logger.info("=" * 60)
        logger.info(f"Training {self.model_name.upper()} model")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.epochs} (patience={self.patience})")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("=" * 60)

        try:
            start_time = time.time()
            checkpoint_path = os.path.join(MODEL_DIR, f"{self.model_name}_best.pt")

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(train_loader)
                val_loss = self._validate(val_loader)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

                current_lr = self.optimizer.param_groups[0]["lr"]

                # Logging
                logger.info(
                    f"  Epoch {epoch:3d}/{self.epochs} │ "
                    f"Train Loss: {train_loss:.4f} │ "
                    f"Val Loss: {val_loss:.4f} │ "
                    f"LR: {current_lr:.2e}"
                )

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), checkpoint_path)
                    logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        logger.info(f"  ⏹ Early stopping at epoch {epoch}")
                        break

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"Training complete in {elapsed:.1f}s")
            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"  Checkpoint: {checkpoint_path}")
            logger.info("=" * 60)

            # Load best weights
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

            return {
                "model_name": self.model_name,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "total_epochs": len(self.train_losses),
                "time_seconds": elapsed,
                "checkpoint_path": checkpoint_path,
            }

        except Exception as e:
            raise ModelTrainingError(
                message=f"Training failed for {self.model_name}",
                details=str(e),
            ) from e
