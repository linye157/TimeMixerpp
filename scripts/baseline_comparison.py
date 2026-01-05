#!/usr/bin/env python
"""
Baseline model comparison for time series binary classification.

Compares TimeMixer++ with baseline models:
- LSTM
- BiLSTM
- LSTM-Transformer
- CNN-BiLSTM
- Transformer
- MLP

Usage:
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --epochs 50
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --models lstm bilstm transformer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import os
import json
from typing import Dict, Type, Any, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict, create_dataloaders
from timemixerpp.utils import set_seed, compute_metrics, setup_logging, AverageMeter

logger = logging.getLogger(__name__)


# ============================================
# Baseline Models
# ============================================

class LSTMClassifier(nn.Module):
    """LSTM-based classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, (h_n, _) = self.lstm(x)
        # Concatenate forward and backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=-1)
        out = self.dropout(h_concat)
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class LSTMTransformerClassifier(nn.Module):
    """LSTM + Transformer hybrid classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, 
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        logits = self.fc(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class CNNBiLSTMClassifier(nn.Module):
    """CNN + BiLSTM hybrid classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # CNN: (B, L, 1) -> (B, 1, L) -> (B, hidden, L) -> (B, hidden, L//2)
        x = x.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        # LSTM: (B, hidden, L//2) -> (B, L//2, hidden)
        x = x.transpose(1, 2)
        out, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=-1)
        out = self.dropout(h_concat)
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class TransformerClassifier(nn.Module):
    """Pure Transformer classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class MLPClassifier(nn.Module):
    """Simple MLP classifier (flattened input)."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = seq_len
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        logits = self.mlp(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class GRUClassifier(nn.Module):
    """GRU-based classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


# ============================================
# Model Registry (Extensible)
# ============================================

@dataclass
class ModelConfig:
    """Configuration for a baseline model."""
    model_class: Type[nn.Module]
    default_kwargs: Dict[str, Any]
    description: str


# Registry of available models - can be extended by adding new entries
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    'lstm': ModelConfig(
        model_class=LSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='LSTM classifier'
    ),
    'bilstm': ModelConfig(
        model_class=BiLSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='Bidirectional LSTM classifier'
    ),
    'lstm_transformer': ModelConfig(
        model_class=LSTMTransformerClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'dropout': 0.1},
        description='LSTM + Transformer hybrid'
    ),
    'cnn_bilstm': ModelConfig(
        model_class=CNNBiLSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='CNN + BiLSTM hybrid'
    ),
    'transformer': ModelConfig(
        model_class=TransformerClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'dropout': 0.1},
        description='Pure Transformer classifier'
    ),
    'mlp': ModelConfig(
        model_class=MLPClassifier,
        default_kwargs={'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.1},
        description='Multi-layer Perceptron'
    ),
    'gru': ModelConfig(
        model_class=GRUClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='GRU classifier'
    ),
}


def register_model(name: str, model_class: Type[nn.Module], default_kwargs: Dict, description: str):
    """
    Register a new model to the registry.
    
    Example:
        register_model('my_model', MyModelClass, {'hidden_dim': 64}, 'My custom model')
    """
    MODEL_REGISTRY[name] = ModelConfig(model_class, default_kwargs, description)


def list_models() -> List[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def create_model(name: str, seq_len: int = 48, **kwargs) -> nn.Module:
    """Create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list_models()}")
    
    config = MODEL_REGISTRY[name]
    model_kwargs = {**config.default_kwargs, 'seq_len': seq_len, **kwargs}
    return config.model_class(**model_kwargs)


# ============================================
# Training and Evaluation
# ============================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str
) -> Dict[str, Any]:
    """Train a model and return metrics."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    best_metrics = {}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = AverageMeter()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['logits'], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss.update(loss.item(), batch_x.size(0))
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                all_preds.append(output['probs'].cpu().numpy())
                all_labels.append(batch_y.numpy())
        
        all_preds = np.concatenate(all_preds).squeeze()
        all_labels = np.concatenate(all_labels).squeeze()
        
        metrics = compute_metrics(all_labels, all_preds)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics.copy()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Loss={train_loss.avg:.4f}, F1={metrics['f1']:.4f}")
    
    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline model comparison')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models to compare. Available: {list_models()}. Default: all')
    parser.add_argument('--include_timemixer', action='store_true',
                        help='Include TimeMixer++ in comparison')
    
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for all models')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--output', type=str, default='results/baseline_comparison.json',
                        help='Path to save results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Baseline Model Comparison")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data: {args.data_path}")
    _, X, y = load_file_strict(args.data_path)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        X, y,
        batch_size=args.batch_size,
        val_split=args.val_split,
        normalize=True,
        seed=args.seed
    )
    
    # Select models
    model_names = args.models if args.models else list_models()
    logger.info(f"Models to compare: {model_names}")
    
    # Results
    results = {}
    
    # Train each baseline model
    for name in model_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {name}, skipping")
            continue
        
        logger.info(f"\nTraining {name}...")
        set_seed(args.seed)  # Reset seed for fair comparison
        
        model = create_model(name, seq_len=48, hidden_dim=args.hidden_dim)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        metrics = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device, model_name=name
        )
        
        results[name] = {
            'params': n_params,
            'description': MODEL_REGISTRY[name].description,
            **metrics
        }
    
    # Include TimeMixer++ if requested
    if args.include_timemixer:
        logger.info("\nTraining TimeMixer++...")
        set_seed(args.seed)
        
        config = TimeMixerPPConfig(
            seq_len=48, c_in=1, d_model=args.hidden_dim,
            n_layers=2, n_heads=4, top_k=3, dropout=0.1
        )
        model = TimeMixerPPForBinaryCls(config)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        metrics = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device, model_name='TimeMixer++'
        )
        
        results['timemixer++'] = {
            'params': n_params,
            'description': 'TimeMixer++ (ours)',
            **metrics
        }
    
    # Print results
    print("\n" + "=" * 80)
    print(" Comparison Results")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':>10} {'Acc':>8} {'F1':>8} {'AUROC':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * 80)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1].get('f1', 0)):
        print(f"{name:<20} {res['params']:>10,} {res['accuracy']:>8.4f} {res['f1']:>8.4f} "
              f"{res['auroc']:>8.4f} {res['fpr']:>8.4f} {res['fnr']:>8.4f}")
    
    print("=" * 80)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

