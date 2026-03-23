"""
Content-Based Feature Extractor Model.
Uses user and destination attribute features to predict ratings,
independent of collaborative signals.
"""

import torch
import torch.nn as nn

from utils.config import (
    USER_FEATURE_DIM, DEST_FEATURE_DIM,
    CONTENT_HIDDEN_LAYERS, DROPOUT_RATE,
)


class ContentModel(nn.Module):
    """
    Content-based recommendation model using user and destination features.
    
    Architecture:
        - User feature MLP: user attributes → user content vector
        - Dest feature MLP: destination attributes → dest content vector
        - Concatenation → prediction MLP → rating
    """

    def __init__(
        self,
        user_feature_dim: int = USER_FEATURE_DIM,
        dest_feature_dim: int = DEST_FEATURE_DIM,
        hidden_layers: list[int] = None,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        hidden_layers = hidden_layers or CONTENT_HIDDEN_LAYERS

        # ─── User Feature Network ───
        self.user_net = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
        )

        # ─── Destination Feature Network ───
        self.dest_net = nn.Sequential(
            nn.Linear(dest_feature_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
        )

        # ─── Prediction Network ───
        combined_dim = hidden_layers[1] * 2
        self.prediction = nn.Sequential(
            nn.Linear(combined_dim, hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[1], 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_features: (batch_size, user_feature_dim) user attribute features
            dest_features: (batch_size, dest_feature_dim) destination attribute features
        
        Returns:
            (batch_size, 1) predicted ratings
        """
        user_vec = self.user_net(user_features)
        dest_vec = self.dest_net(dest_features)

        combined = torch.cat([user_vec, dest_vec], dim=-1)
        prediction = self.prediction(combined)

        # Scale to 1-5
        return 1.0 + 4.0 * torch.sigmoid(prediction)
