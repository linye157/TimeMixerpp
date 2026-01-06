"""
Data utilities for TimeMixer++ training.

Includes:
- Dataset class for temperature/accident data
- NPZMultiScaleDataset for metric learning
- Data normalization utilities
- Data loading functions
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def load_file_strict(file_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load data file with strict format handling.
    
    Args:
        file_path: Path to data file (.xlsx or .csv)
        
    Returns:
        data: Original DataFrame
        X: Feature array, shape (n, 48)
        y: Label array, shape (n,)
        
    File format specifications:
        .xlsx: sheet_name=2, header=0, X=iloc[:, 3:51], y=iloc[:, 51]
        .csv: header=None, X=iloc[:, 0:48], y=iloc[:, 48]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        logger.info(f"Loading Excel file (Sheet3): {file_path}")
        # Read third sheet (index 2)
        data = pd.read_excel(file_path, sheet_name=2, header=0)
        # Extract columns 4-51 (index 3:51) as features, column 52 (index 51) as label
        X = data.iloc[:, 3:51].values
        y = data.iloc[:, 51].values
        
    elif file_path.endswith('.csv'):
        logger.info(f"Loading CSV file (no header): {file_path}")
        # Read CSV without header
        data = pd.read_csv(file_path, header=None)
        # Extract first 48 columns as features, column 49 (index 48) as label
        X = data.iloc[:, 0:48].values
        y = data.iloc[:, 48].values
        
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Ensure correct data types
    X = pd.to_numeric(X.flatten(), errors='coerce').reshape(X.shape).astype(float)
    y = pd.to_numeric(y, errors='coerce').astype(float)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    logger.info(f"Loaded {len(X)} samples, X shape: {X.shape}, y shape: {y.shape}")
    
    return data, X, y


class TemperatureDataset(Dataset):
    """
    Dataset for temperature time series with binary labels.
    
    Args:
        X: Feature array, shape (n, 48) or (n, 48, 1)
        y: Label array, shape (n,)
        normalize: Whether to normalize features
        mean: Pre-computed mean for normalization
        std: Pre-computed std for normalization
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        # Handle input shape
        if X.ndim == 1:
            X = X.reshape(-1, 48)
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        
        # Normalization
        self.normalize = normalize
        if normalize:
            if mean is None:
                self.mean = self.X.mean(axis=0, keepdims=True)
            else:
                self.mean = mean
            if std is None:
                self.std = self.X.std(axis=0, keepdims=True)
                self.std[self.std < 1e-6] = 1.0  # Avoid division by zero
            else:
                self.std = std
            
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
    def get_stats(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get normalization statistics."""
        return self.mean, self.std


class Normalizer:
    """
    Normalizer for time series data.
    
    Supports z-score normalization with optional saved statistics.
    """
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std
        self.fitted = mean is not None and std is not None
    
    def fit(self, X: np.ndarray) -> 'Normalizer':
        """
        Fit normalizer to data.
        
        Args:
            X: Data array, shape (n, seq_len) or (n, seq_len, c_in)
            
        Returns:
            self
        """
        if X.ndim == 3:
            self.mean = X.mean(axis=(0, 1), keepdims=True)
            self.std = X.std(axis=(0, 1), keepdims=True)
        else:
            self.mean = X.mean(axis=0, keepdims=True)
            self.std = X.std(axis=0, keepdims=True)
        
        self.std[self.std < 1e-6] = 1.0
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data.
        
        Args:
            X: Data array
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data.
        
        Args:
            X: Normalized data
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return X * self.std + self.mean
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X).transform(X)
    
    def save(self, path: str):
        """Save normalizer statistics."""
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Load normalizer from file."""
        data = np.load(path)
        return cls(mean=data['mean'], std=data['std'])


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create train and validation dataloaders.
    
    Args:
        X: Feature array, shape (n, 48)
        y: Label array, shape (n,)
        batch_size: Batch size
        val_split: Validation split ratio
        normalize: Whether to normalize
        shuffle: Whether to shuffle training data
        num_workers: Number of dataloader workers
        seed: Random seed for split
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        stats: Normalization statistics (mean, std) or None
    """
    # Split data
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    val_size = int(n * val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create datasets
    train_dataset = TemperatureDataset(X_train, y_train, normalize=normalize)
    mean, std = train_dataset.get_stats()
    val_dataset = TemperatureDataset(X_val, y_val, normalize=normalize, mean=mean, std=std)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    stats = (mean, std) if normalize else None
    
    return train_loader, val_loader, stats


# ============================================
# Multi-Scale Dataset for Metric Learning
# ============================================

class NPZMultiScaleDataset(Dataset):
    """
    Dataset for multi-scale features from NPZ file.
    
    Used for metric learning (SupCon) training.
    
    NPZ file should contain:
    - scale_0: (N, 48, 64) - Scale 0 features
    - scale_1: (N, 24, 64) - Scale 1 features
    - scale_2: (N, 12, 64) - Scale 2 features
    - labels: (N,) - Binary labels
    
    Returns:
        (x0, x1, x2, label, index) where:
        - x0: (48, 64) float32
        - x1: (24, 64) float32
        - x2: (12, 64) float32
        - label: float32
        - index: int (sample index)
    """
    
    def __init__(
        self,
        npz_path: str,
        indices: Optional[List[int]] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            npz_path: Path to NPZ file with multi-scale features
            indices: Optional list of indices to use (for train/val/test splits)
            transform: Optional transform to apply to features
        """
        logger.info(f"Loading NPZ file: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        
        # Load features
        self.scale_0 = data['scale_0'].astype(np.float32)
        self.scale_1 = data['scale_1'].astype(np.float32)
        self.scale_2 = data['scale_2'].astype(np.float32)
        self.labels = data['labels'].astype(np.float32)
        
        # Validate shapes
        N = len(self.labels)
        assert self.scale_0.shape[0] == N, f"scale_0 samples mismatch: {self.scale_0.shape[0]} vs {N}"
        assert self.scale_1.shape[0] == N, f"scale_1 samples mismatch: {self.scale_1.shape[0]} vs {N}"
        assert self.scale_2.shape[0] == N, f"scale_2 samples mismatch: {self.scale_2.shape[0]} vs {N}"
        
        logger.info(f"Loaded {N} samples:")
        logger.info(f"  scale_0: {self.scale_0.shape}")
        logger.info(f"  scale_1: {self.scale_1.shape}")
        logger.info(f"  scale_2: {self.scale_2.shape}")
        logger.info(f"  labels: {self.labels.shape}, positive ratio: {self.labels.mean():.4f}")
        
        # Apply indices filter
        if indices is not None:
            self.indices = np.array(indices)
            logger.info(f"Using subset of {len(self.indices)} samples")
        else:
            self.indices = np.arange(N)
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a sample.
        
        Returns:
            x0: Scale 0 features (48, 64)
            x1: Scale 1 features (24, 64)
            x2: Scale 2 features (12, 64)
            label: Binary label
            index: Original sample index
        """
        real_idx = self.indices[idx]
        
        x0 = self.scale_0[real_idx]
        x1 = self.scale_1[real_idx]
        x2 = self.scale_2[real_idx]
        label = self.labels[real_idx]
        
        if self.transform is not None:
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        
        return (
            torch.tensor(x0, dtype=torch.float32),
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            real_idx
        )
    
    def get_labels(self) -> np.ndarray:
        """Get labels for the current subset."""
        return self.labels[self.indices]
    
    def get_class_weights(self) -> np.ndarray:
        """Compute sample weights for balanced sampling."""
        labels = self.get_labels()
        # Threshold for binary
        binary_labels = (labels >= 0.5).astype(int)
        
        # Compute weights inversely proportional to class frequency
        class_counts = np.bincount(binary_labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum()  # Normalize
        
        sample_weights = class_weights[binary_labels]
        return sample_weights


def create_splits(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: Optional[str] = None
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits.
    
    Args:
        n_samples: Total number of samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        save_path: Optional path to save splits JSON
        
    Returns:
        Dictionary with 'train_ids', 'val_ids', 'test_ids'
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_ids = indices[:n_train].tolist()
    val_ids = indices[n_train:n_train + n_val].tolist()
    test_ids = indices[n_train + n_val:].tolist()
    
    splits = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'n_samples': n_samples,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed
    }
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Splits saved to: {save_path}")
    
    logger.info(f"Created splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    return splits


def load_splits(splits_path: str) -> Dict[str, List[int]]:
    """
    Load splits from JSON file.
    
    Args:
        splits_path: Path to splits JSON
        
    Returns:
        Dictionary with 'train_ids', 'val_ids', 'test_ids'
    """
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    logger.info(f"Loaded splits from: {splits_path}")
    logger.info(f"  train: {len(splits['train_ids'])}, val: {len(splits['val_ids'])}, test: {len(splits['test_ids'])}")
    
    return splits


def create_multiscale_dataloaders(
    npz_path: str,
    batch_size: int = 256,
    splits_path: Optional[str] = None,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    balanced_sampling: bool = False,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train/val/test dataloaders for multi-scale features.
    
    Args:
        npz_path: Path to NPZ file
        batch_size: Batch size
        splits_path: Optional path to existing splits JSON (will load if exists, create if not)
        split_ratio: (train, val, test) ratios
        balanced_sampling: Whether to use balanced sampling for training
        seed: Random seed
        num_workers: DataLoader workers
        
    Returns:
        train_loader, val_loader, test_loader, splits
    """
    # Load NPZ to get sample count
    data = np.load(npz_path, allow_pickle=True)
    n_samples = len(data['labels'])
    
    # Handle splits
    if splits_path and Path(splits_path).exists():
        splits = load_splits(splits_path)
    else:
        splits = create_splits(
            n_samples,
            train_ratio=split_ratio[0],
            val_ratio=split_ratio[1],
            test_ratio=split_ratio[2],
            seed=seed,
            save_path=splits_path
        )
    
    # Create datasets
    train_dataset = NPZMultiScaleDataset(npz_path, indices=splits['train_ids'])
    val_dataset = NPZMultiScaleDataset(npz_path, indices=splits['val_ids'])
    test_dataset = NPZMultiScaleDataset(npz_path, indices=splits['test_ids'])
    
    # Create samplers
    if balanced_sampling:
        sample_weights = train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, splits

