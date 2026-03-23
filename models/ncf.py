"""
Neural Collaborative Filtering (NCF) Model.
Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
pathways for learning user-destination collaborative interaction patterns.

Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
"""

import torch
import torch.nn as nn

from utils.config import EMBEDDING_DIM, GMF_DIM, MLP_HIDDEN_LAYERS, DROPOUT_RATE


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model (NeuMF variant).
    
    Architecture:
        - GMF path: element-wise product of user/dest embeddings
        - MLP path: concatenated embeddings → deep MLP
        - NeuMF: concatenation of GMF + MLP outputs → prediction
    """

    def __init__(
        self,
        num_users: int,
        num_destinations: int,
        embedding_dim: int = EMBEDDING_DIM,
        gmf_dim: int = GMF_DIM,
        mlp_hidden_layers: list[int] = None,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        mlp_hidden_layers = mlp_hidden_layers or MLP_HIDDEN_LAYERS

        # ─── GMF Embeddings ───
        self.user_embedding_gmf = nn.Embedding(num_users, gmf_dim)
        self.dest_embedding_gmf = nn.Embedding(num_destinations, gmf_dim)

        # ─── MLP Embeddings ───
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.dest_embedding_mlp = nn.Embedding(num_destinations, embedding_dim)

        # ─── MLP Layers ───
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        for hidden_dim in mlp_hidden_layers:
            mlp_layers.extend([
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            mlp_input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # ─── NeuMF Fusion ───
        neumf_input_dim = gmf_dim + mlp_hidden_layers[-1]
        self.prediction = nn.Sequential(
            nn.Linear(neumf_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, user_ids: torch.Tensor, dest_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: (batch_size,) user indices
            dest_ids: (batch_size,) destination indices
        
        Returns:
            (batch_size, 1) predicted ratings
        """
        # GMF path
        user_gmf = self.user_embedding_gmf(user_ids)
        dest_gmf = self.dest_embedding_gmf(dest_ids)
        gmf_out = user_gmf * dest_gmf  # Element-wise product

        # MLP path
        user_mlp = self.user_embedding_mlp(user_ids)
        dest_mlp = self.dest_embedding_mlp(dest_ids)
        mlp_input = torch.cat([user_mlp, dest_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # NeuMF fusion
        neumf_input = torch.cat([gmf_out, mlp_out], dim=-1)
        prediction = self.prediction(neumf_input)

        # Scale to 1-5 rating range
        return 1.0 + 4.0 * torch.sigmoid(prediction)
