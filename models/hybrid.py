"""
Hybrid Recommendation Model.
Combines Neural Collaborative Filtering and Content-Based models
with a learnable attention-weighted fusion mechanism.
"""

import torch
import torch.nn as nn

from models.ncf import NCF
from models.content_model import ContentModel
from utils.config import (
    EMBEDDING_DIM, GMF_DIM, MLP_HIDDEN_LAYERS,
    USER_FEATURE_DIM, DEST_FEATURE_DIM,
    CONTENT_HIDDEN_LAYERS, HYBRID_HIDDEN_LAYERS, DROPOUT_RATE,
)


class HybridRecommender(nn.Module):
    """
    Hybrid deep learning recommender combining NCF (collaborative)
    and Content-Based (attribute) signals via learned attention.
    
    Architecture:
        - NCF branch → collaborative prediction
        - Content branch → content prediction
        - Attention gate → learnable weight for combining both signals
        - Final fusion → weighted combination → rating
    """

    def __init__(
        self,
        num_users: int,
        num_destinations: int,
        embedding_dim: int = EMBEDDING_DIM,
        gmf_dim: int = GMF_DIM,
        mlp_hidden: list[int] = None,
        user_feat_dim: int = USER_FEATURE_DIM,
        dest_feat_dim: int = DEST_FEATURE_DIM,
        content_hidden: list[int] = None,
        hybrid_hidden: list[int] = None,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        mlp_hidden = mlp_hidden or MLP_HIDDEN_LAYERS
        content_hidden = content_hidden or CONTENT_HIDDEN_LAYERS
        hybrid_hidden = hybrid_hidden or HYBRID_HIDDEN_LAYERS

        # ─── Sub-models ───
        self.ncf = NCF(
            num_users, num_destinations,
            embedding_dim, gmf_dim, mlp_hidden, dropout,
        )
        self.content = ContentModel(
            user_feat_dim, dest_feat_dim, content_hidden, dropout,
        )

        # ─── Attention Gate ───
        # Learns to weight how much to rely on collaborative vs content signals
        self.attention_gate = nn.Sequential(
            nn.Linear(2, hybrid_hidden[0]),
            nn.ReLU(),
            nn.Linear(hybrid_hidden[0], 2),
            nn.Softmax(dim=-1),
        )

        # ─── Final Fusion Layer ───
        self.fusion = nn.Sequential(
            nn.Linear(2, hybrid_hidden[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hybrid_hidden[-1], 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize fusion/attention weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and ("attention" in name or "fusion" in name):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        dest_ids: torch.Tensor,
        user_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass combining both collaborative and content signals.
        
        Args:
            user_ids: (batch_size,) user indices
            dest_ids: (batch_size,) destination indices
            user_features: (batch_size, user_feat_dim) user attributes
            dest_features: (batch_size, dest_feat_dim) destination attributes
        
        Returns:
            (batch_size, 1) predicted ratings in range [1, 5]
        """
        # Get predictions from both branches
        ncf_pred = self.ncf(user_ids, dest_ids)           # (B, 1)
        content_pred = self.content(user_features, dest_features)  # (B, 1)

        # Stack predictions for attention
        combined = torch.cat([ncf_pred, content_pred], dim=-1)    # (B, 2)

        # Compute attention weights
        attention_weights = self.attention_gate(combined)          # (B, 2)

        # Weighted combination
        weighted = combined * attention_weights                    # (B, 2)

        # Final fusion
        output = self.fusion(weighted)                             # (B, 1)

        # Scale to 1-5
        return 1.0 + 4.0 * torch.sigmoid(output)

    def get_attention_weights(
        self,
        user_ids: torch.Tensor,
        dest_ids: torch.Tensor,
        user_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights showing how much the model relies on
        collaborative vs content signals. Useful for interpretability.
        
        Returns:
            Tuple of (attention_weights, predictions)
        """
        with torch.no_grad():
            ncf_pred = self.ncf(user_ids, dest_ids)
            content_pred = self.content(user_features, dest_features)
            combined = torch.cat([ncf_pred, content_pred], dim=-1)
            weights = self.attention_gate(combined)
            weighted = combined * weights
            output = self.fusion(weighted)
            prediction = 1.0 + 4.0 * torch.sigmoid(output)
        return weights, prediction
