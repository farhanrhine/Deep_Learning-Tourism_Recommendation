"""
Data Preprocessing & Feature Engineering Pipeline.
Handles encoding, normalization, splitting, and PyTorch data loading.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.logger import get_project_logger
from utils.custom_exception import DataNotFoundError, PreprocessingError
from utils.config import (
    DATA_DIR, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE, RANDOM_SEED
)

logger = get_project_logger(__name__)


class TourismDataset(Dataset):
    """PyTorch Dataset for tourism recommendation data."""

    def __init__(
        self,
        user_ids: np.ndarray,
        dest_ids: np.ndarray,
        user_features: np.ndarray,
        dest_features: np.ndarray,
        ratings: np.ndarray,
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.dest_ids = torch.LongTensor(dest_ids)
        self.user_features = torch.FloatTensor(user_features)
        self.dest_features = torch.FloatTensor(dest_features)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "dest_id": self.dest_ids[idx],
            "user_features": self.user_features[idx],
            "dest_features": self.dest_features[idx],
            "rating": self.ratings[idx],
        }


class DataPreprocessor:
    """
    Handles all data preprocessing: loading, encoding, normalization,
    splitting, and DataLoader creation.
    """

    def __init__(self):
        self.user_label_encoders: dict[str, LabelEncoder] = {}
        self.dest_label_encoders: dict[str, LabelEncoder] = {}
        self.user_scaler = MinMaxScaler()
        self.dest_scaler = MinMaxScaler()
        self.num_users = 0
        self.num_destinations = 0

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load CSV datasets from the data directory."""
        logger.info("Loading datasets...")

        files = {
            "users": os.path.join(DATA_DIR, "users.csv"),
            "destinations": os.path.join(DATA_DIR, "destinations.csv"),
            "interactions": os.path.join(DATA_DIR, "interactions.csv"),
        }

        for name, path in files.items():
            if not os.path.exists(path):
                raise DataNotFoundError(
                    message=f"Dataset file not found: {name}.csv",
                    details=f"Expected at {path}. Run 'python main.py generate-data' first.",
                )

        users = pd.read_csv(files["users"])
        destinations = pd.read_csv(files["destinations"])
        interactions = pd.read_csv(files["interactions"])

        self.num_users = len(users)
        self.num_destinations = len(destinations)

        logger.info(f"  ✓ Loaded {len(users)} users, {len(destinations)} destinations, "
                    f"{len(interactions)} interactions")
        return users, destinations, interactions

    def encode_features(
        self, users: pd.DataFrame, destinations: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode categorical features and normalize numerical features.
        
        Returns:
            Tuple of (user_features, dest_features) as numpy arrays.
        """
        logger.info("Encoding and normalizing features...")

        try:
            # ─── Encode User Features ───
            user_cat_cols = ["travel_style", "budget_level"]
            user_num_cols = ["age", "past_trips"]

            user_encoded = users.copy()
            for col in user_cat_cols:
                le = LabelEncoder()
                user_encoded[col] = le.fit_transform(user_encoded[col])
                self.user_label_encoders[col] = le

            user_feature_cols = user_cat_cols + user_num_cols
            user_features = user_encoded[user_feature_cols].values.astype(np.float32)
            user_features = self.user_scaler.fit_transform(user_features)

            # ─── Encode Destination Features ───
            dest_cat_cols = ["category", "climate", "best_season"]
            dest_num_cols = ["avg_cost", "popularity_score"]

            dest_encoded = destinations.copy()
            for col in dest_cat_cols:
                le = LabelEncoder()
                dest_encoded[col] = le.fit_transform(dest_encoded[col])
                self.dest_label_encoders[col] = le

            dest_feature_cols = dest_cat_cols + dest_num_cols
            dest_features = dest_encoded[dest_feature_cols].values.astype(np.float32)
            dest_features = self.dest_scaler.fit_transform(dest_features)

            logger.info(f"  ✓ User features shape: {user_features.shape}")
            logger.info(f"  ✓ Destination features shape: {dest_features.shape}")

            return user_features, dest_features

        except Exception as e:
            raise PreprocessingError(
                message="Feature encoding failed",
                details=str(e),
            ) from e

    def create_splits(
        self,
        interactions: pd.DataFrame,
        user_features: np.ndarray,
        dest_features: np.ndarray,
    ) -> tuple[TourismDataset, TourismDataset, TourismDataset]:
        """
        Split interactions into train/val/test and create TourismDataset instances.
        """
        logger.info("Creating train/val/test splits...")

        # Shuffle
        interactions = interactions.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        n = len(interactions)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        splits = {
            "train": interactions.iloc[:train_end],
            "val": interactions.iloc[train_end:val_end],
            "test": interactions.iloc[val_end:],
        }

        datasets = {}
        for split_name, split_df in splits.items():
            uids = split_df["user_id"].values
            dids = split_df["dest_id"].values
            ratings = split_df["rating"].values

            # Gather per-interaction features
            u_feats = user_features[uids]
            d_feats = dest_features[dids]

            datasets[split_name] = TourismDataset(uids, dids, u_feats, d_feats, ratings)
            logger.info(f"  ✓ {split_name}: {len(split_df)} samples")

        return datasets["train"], datasets["val"], datasets["test"]

    def get_dataloaders(
        self,
        train_ds: TourismDataset,
        val_ds: TourismDataset,
        test_ds: TourismDataset,
        batch_size: int = BATCH_SIZE,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for each split."""
        logger.info(f"Creating DataLoaders (batch_size={batch_size})...")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        logger.info(f"  ✓ Train batches: {len(train_loader)}")
        logger.info(f"  ✓ Val batches: {len(val_loader)}")
        logger.info(f"  ✓ Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    def prepare_all(
        self, batch_size: int = BATCH_SIZE
    ) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline: load → encode → split → dataloaders.
        
        Returns:
            (train_loader, val_loader, test_loader, user_features, dest_features)
        """
        users, destinations, interactions = self.load_data()
        user_features, dest_features = self.encode_features(users, destinations)
        train_ds, val_ds, test_ds = self.create_splits(interactions, user_features, dest_features)
        train_loader, val_loader, test_loader = self.get_dataloaders(
            train_ds, val_ds, test_ds, batch_size
        )
        return train_loader, val_loader, test_loader, user_features, dest_features
