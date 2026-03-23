"""
Evaluation Module for the Tourism Recommendation System.
Computes metrics (RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate)
and generates visualization plots.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_project_logger
from utils.config import TOP_K_VALUES, RATING_THRESHOLD, PLOTS_DIR

logger = get_project_logger(__name__)

# Set plot style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 120})


class Evaluator:
    """Compute recommendation metrics and generate plots."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(self, model: nn.Module, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Get model predictions and ground truth ratings."""
        model.eval()
        all_preds = []
        all_targets = []

        for batch in data_loader:
            user_ids = batch["user_id"].to(self.device)
            dest_ids = batch["dest_id"].to(self.device)
            user_feats = batch["user_features"].to(self.device)
            dest_feats = batch["dest_features"].to(self.device)
            ratings = batch["rating"]

            if hasattr(model, "ncf") and hasattr(model, "content"):
                preds = model(user_ids, dest_ids, user_feats, dest_feats)
            elif hasattr(model, "user_embedding_gmf"):
                preds = model(user_ids, dest_ids)
            else:
                preds = model(user_feats, dest_feats)

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(ratings.numpy().flatten())

        return np.concatenate(all_preds), np.concatenate(all_targets)

    def compute_regression_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        """Compute RMSE and MAE."""
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}

    def compute_ranking_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        k_values: list[int] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute Precision@K, Recall@K, NDCG@K, and Hit Rate@K.
        """
        k_values = k_values or TOP_K_VALUES
        relevant = targets >= RATING_THRESHOLD

        # Sort by predicted score (descending)
        sorted_indices = np.argsort(-predictions)
        sorted_relevant = relevant[sorted_indices]

        metrics = {}
        for k in k_values:
            top_k_relevant = sorted_relevant[:k]
            num_relevant_in_k = top_k_relevant.sum()
            total_relevant = relevant.sum()

            precision = num_relevant_in_k / k if k > 0 else 0
            recall = num_relevant_in_k / total_relevant if total_relevant > 0 else 0
            hit_rate = 1.0 if num_relevant_in_k > 0 else 0.0

            # NDCG@K
            dcg = sum(
                (2 ** rel - 1) / np.log2(i + 2)
                for i, rel in enumerate(top_k_relevant)
            )
            ideal_sorted = np.sort(relevant)[::-1][:k]
            idcg = sum(
                (2 ** rel - 1) / np.log2(i + 2)
                for i, rel in enumerate(ideal_sorted)
            )
            ndcg = dcg / idcg if idcg > 0 else 0

            metrics[f"@{k}"] = {
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "NDCG": round(ndcg, 4),
                "HitRate": round(hit_rate, 4),
            }

        return metrics

    def evaluate_model(
        self, model: nn.Module, test_loader: DataLoader, model_name: str = "model"
    ) -> dict:
        """Full evaluation: regression + ranking metrics."""
        logger.info(f"Evaluating {model_name.upper()}...")
        model.to(self.device)

        predictions, targets = self.predict(model, test_loader)
        reg_metrics = self.compute_regression_metrics(predictions, targets)
        rank_metrics = self.compute_ranking_metrics(predictions, targets)

        logger.info(f"  RMSE: {reg_metrics['RMSE']:.4f} │ MAE: {reg_metrics['MAE']:.4f}")
        for k, m in rank_metrics.items():
            logger.info(f"  {k} → P={m['Precision']:.4f}  R={m['Recall']:.4f}  "
                        f"NDCG={m['NDCG']:.4f}  HR={m['HitRate']:.4f}")

        return {
            "model_name": model_name,
            "regression": reg_metrics,
            "ranking": rank_metrics,
            "predictions": predictions,
            "targets": targets,
        }

    # ─────────────────────────── Plotting ───────────────────────────

    @staticmethod
    def plot_training_curves(histories: list[dict], save_path: str | None = None):
        """Plot training and validation loss curves for all models."""
        fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 5))
        if len(histories) == 1:
            axes = [axes]

        for ax, hist in zip(axes, histories):
            epochs = range(1, len(hist["train_losses"]) + 1)
            ax.plot(epochs, hist["train_losses"], "o-", label="Train Loss", markersize=3)
            ax.plot(epochs, hist["val_losses"], "s-", label="Val Loss", markersize=3)
            ax.set_title(f'{hist["model_name"].upper()} Training', fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or os.path.join(PLOTS_DIR, "training_curves.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Training curves saved to {path}")

    @staticmethod
    def plot_model_comparison(results: list[dict], save_path: str | None = None):
        """Bar chart comparing RMSE/MAE across models."""
        model_names = [r["model_name"].upper() for r in results]
        rmse_vals = [r["regression"]["RMSE"] for r in results]
        mae_vals = [r["regression"]["MAE"] for r in results]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="#4A90D9")
        bars2 = ax.bar(x + width / 2, mae_vals, width, label="MAE", color="#E8724A")

        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Error")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=9)

        plt.tight_layout()
        path = save_path or os.path.join(PLOTS_DIR, "model_comparison.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Model comparison saved to {path}")

    @staticmethod
    def plot_prediction_distribution(results: list[dict], save_path: str | None = None):
        """Plot predicted vs actual rating distributions."""
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]

        for ax, res in zip(axes, results):
            ax.hist(res["targets"], bins=20, alpha=0.6, label="Actual", color="#4A90D9")
            ax.hist(res["predictions"], bins=20, alpha=0.6, label="Predicted", color="#E8724A")
            ax.set_title(f'{res["model_name"].upper()} Predictions', fontsize=13, fontweight="bold")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or os.path.join(PLOTS_DIR, "prediction_dist.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Prediction distributions saved to {path}")
