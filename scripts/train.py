#!/usr/bin/env python
"""
Training script for TimeMixer++ binary classification model.

支持消融模式训练：可以训练去掉某些组件的模型。

Usage:
    python scripts/train.py --data_path TDdata/TrainData.csv --epochs 50 --batch_size 32
    
    # 消融模式训练（去掉TID）
    python scripts/train.py --data_path TDdata/TrainData.csv --ablation no_tid --epochs 50
    
    # 消融模式训练（去掉MCM）
    python scripts/train.py --data_path TDdata/TrainData.csv --ablation no_mcm --epochs 50

For a minimal test with random data:
    python scripts/train.py --use_random_data --epochs 2
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict, create_dataloaders, TemperatureDataset
from timemixerpp.utils import (
    set_seed, compute_metrics, save_checkpoint, 
    EarlyStopping, setup_logging, AverageMeter
)
from timemixerpp.mrti import MRTI, TimeImageInfo
from timemixerpp.tid import TID
from timemixerpp.mcm import MCM
from timemixerpp.mrm import MRM
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ============================================
# Ablation Support
# ============================================

ABLATION_DESCRIPTIONS = {
    'full': '完整模型（无消融）',
    'no_fft': '使用固定周期代替FFT检测',
    'no_tid': '去掉TID（无季节性/趋势分解）',
    'no_mcm': '去掉MCM（无跨尺度混合）',
    'no_mrm': '去掉MRM（使用简单平均代替幅值加权）',
    'single_scale': '单尺度（无多尺度处理）',
}


class FixedPeriodMRTI(nn.Module):
    """MRTI with fixed periods (no FFT)."""
    def __init__(self, fixed_periods: List[int] = [4, 6, 8], min_period: int = 2):
        super().__init__()
        self.fixed_periods = fixed_periods
        self.min_period = min_period
        self.mrti = MRTI(top_k=len(fixed_periods), min_period=min_period)
    
    def forward(self, multi_scale_x):
        B = multi_scale_x[0].shape[0]
        device = multi_scale_x[0].device
        time_images = []
        amplitudes = torch.ones(B, len(self.fixed_periods), device=device)
        
        for k, period in enumerate(self.fixed_periods):
            images, original_lengths = [], []
            for x_m in multi_scale_x:
                image, orig_len = self.mrti.reshape_1d_to_2d(x_m, period)
                images.append(image)
                original_lengths.append(orig_len)
            time_images.append(TimeImageInfo(period=period, amplitude=amplitudes[:, k],
                                             images=images, original_lengths=original_lengths))
        return time_images, self.fixed_periods, amplitudes


class IdentityTID(nn.Module):
    """TID that returns input as both seasonal and trend (no decomposition)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)
    
    def forward(self, images):
        return [self.proj(img) for img in images], [img for img in images]


class IdentityMCM(nn.Module):
    """MCM that skips cross-scale mixing."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, seasonal_images, trend_images, original_lengths, period):
        z_list = []
        for m, (s, t) in enumerate(zip(seasonal_images, trend_images)):
            z_2d = s + t
            B, d_model, H, W = z_2d.shape
            z_1d = z_2d.permute(0, 2, 3, 1).reshape(B, H * W, d_model)
            z_list.append(z_1d[:, :original_lengths[m], :])
        return z_list


class IdentityMRM(nn.Module):
    """MRM that uses simple average instead of amplitude weighting."""
    def forward(self, z_per_period, amplitudes):
        K, num_scales = len(z_per_period), len(z_per_period[0])
        return [torch.stack([z_per_period[k][m] for k in range(K)], dim=0).mean(dim=0) 
                for m in range(num_scales)]


class AblatedMixerBlock(nn.Module):
    """MixerBlock with configurable ablations."""
    def __init__(self, config, use_fft_periods=True, use_tid=True, use_mcm=True, use_mrm=True,
                 fixed_periods=[4, 6, 8]):
        super().__init__()
        self.config = config
        self.mrti = MRTI(config.top_k, config.base_len_for_period, config.min_period) if use_fft_periods \
                    else FixedPeriodMRTI(fixed_periods)
        self.tid = TID(config.d_model, config.n_heads, config.dropout) if use_tid \
                   else IdentityTID(config.d_model, config.n_heads, config.dropout)
        self.mcm = MCM(config.d_model, config.mcm_kernel_size, config.mcm_use_two_stride_layers) if use_mcm \
                   else IdentityMCM(config.d_model)
        self.mrm = MRM(weight_mode=config.weight_mode) if use_mrm else IdentityMRM()
        self._layer_norms = None
        self._num_scales = None
    
    def _init_layer_norms(self, num_scales, device):
        if self._num_scales != num_scales:
            self._num_scales = num_scales
            self._layer_norms = nn.ModuleList([nn.LayerNorm(self.config.d_model) for _ in range(num_scales)]).to(device)
    
    def forward(self, multi_scale_x):
        self._init_layer_norms(len(multi_scale_x), multi_scale_x[0].device)
        residuals = multi_scale_x
        time_images, periods, amplitudes = self.mrti(multi_scale_x)
        z_per_period = []
        for ti in time_images:
            seasonal, trend = self.tid(ti.images)
            z_per_period.append(self.mcm(seasonal, trend, ti.original_lengths, ti.period))
        x_out = self.mrm(z_per_period, amplitudes)
        return [self._layer_norms[m](residuals[m] + x_out[m]) for m in range(len(multi_scale_x))]


class SingleScaleModel(nn.Module):
    """TimeMixer++ without multi-scale (single scale only)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Sequential(nn.Linear(config.c_in, config.d_model), nn.Dropout(config.dropout))
        self.blocks = nn.ModuleList([AblatedMixerBlock(config) for _ in range(config.n_layers)])
        self.head = nn.Linear(config.d_model, 1)
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)
        x = self.embed(x)
        multi_scale_x = [x]
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        logits = self.head(multi_scale_x[0].mean(dim=1))
        return {'logits': logits, 'probs': torch.sigmoid(logits)}
    
    def get_multi_scale_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)
        x = self.embed(x)
        multi_scale_x = [x]
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        return multi_scale_x


def create_ablated_model(config, ablation: str):
    """Create a model with specific ablation."""
    if ablation == 'full':
        return TimeMixerPPForBinaryCls(config)
    if ablation == 'single_scale':
        return SingleScaleModel(config)
    
    model = TimeMixerPPForBinaryCls(config)
    ablation_kwargs = {
        'use_fft_periods': ablation != 'no_fft',
        'use_tid': ablation != 'no_tid',
        'use_mcm': ablation != 'no_mcm',
        'use_mrm': ablation != 'no_mrm',
    }
    model.encoder.blocks = nn.ModuleList([
        AblatedMixerBlock(config, **ablation_kwargs) for _ in range(config.n_layers)
    ])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train TimeMixer++ for binary classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training data (.xlsx or .csv)')
    parser.add_argument('--use_random_data', action='store_true',
                        help='Use random synthetic data for testing')
    parser.add_argument('--n_samples', type=int, default=128,
                        help='Number of samples for random data')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of MixerBlock layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Top-K frequencies for MRTI')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Ablation arguments
    parser.add_argument('--ablation', type=str, default='full',
                        choices=list(ABLATION_DESCRIPTIONS.keys()),
                        help=f'Ablation mode. Available: {list(ABLATION_DESCRIPTIONS.keys())}')
    parser.add_argument('--list_ablations', action='store_true',
                        help='List available ablation types and exit')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight for BCEWithLogitsLoss')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    return parser.parse_args()


def generate_random_data(n_samples: int = 128, seq_len: int = 48) -> tuple:
    """
    Generate random synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        
    Returns:
        X: Features, shape (n_samples, seq_len)
        y: Binary labels, shape (n_samples,)
    """
    # Generate temperature-like data
    X = np.random.randn(n_samples, seq_len).astype(np.float32)
    
    # Add some temporal patterns
    for i in range(n_samples):
        # Add trend
        X[i] += np.linspace(0, 0.5, seq_len) * np.random.randn()
        # Add periodicity
        X[i] += 0.3 * np.sin(2 * np.pi * np.arange(seq_len) / 12)
    
    # Generate labels based on some pattern in the data
    # (High variance in latter half -> higher accident probability)
    late_variance = X[:, seq_len//2:].var(axis=1)
    threshold = np.median(late_variance)
    y = (late_variance > threshold).astype(np.float32)
    
    # Add some noise to labels
    noise_mask = np.random.rand(n_samples) < 0.1
    y[noise_mask] = 1 - y[noise_mask]
    
    return X, y


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        
        output = model(batch_x)
        logits = output['logits']
        
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_meter.update(loss.item(), batch_x.size(0))
    
    return loss_meter.avg


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate model."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            
            output = model(batch_x)
            logits = output['logits']
            probs = output['probs']
            
            loss = criterion(logits, batch_y)
            loss_meter.update(loss.item(), batch_x.size(0))
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = loss_meter.avg
    
    return loss_meter.avg, metrics


def get_ablation_suffix(ablation: str) -> str:
    """Get filename suffix for ablation type."""
    return '' if ablation == 'full' else f'_{ablation}'


def main():
    args = parse_args()
    
    # List ablations mode
    if args.list_ablations:
        print("\n可用的消融类型:")
        print("-" * 50)
        for k, v in ABLATION_DESCRIPTIONS.items():
            print(f"  {k:<15} {v}")
        print("-" * 50)
        return
    
    # Setup logging
    setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("TimeMixer++ Training")
    logger.info("=" * 60)
    
    # Show ablation info
    ablation = args.ablation
    logger.info(f"Ablation mode: {ablation} - {ABLATION_DESCRIPTIONS[ablation]}")
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load or generate data
    if args.use_random_data:
        logger.info(f"Generating random data with {args.n_samples} samples")
        X, y = generate_random_data(args.n_samples)
    else:
        if args.data_path is None:
            raise ValueError("Must provide --data_path or use --use_random_data")
        _, X, y = load_file_strict(args.data_path)
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Positive samples: {y.sum():.0f} ({100*y.mean():.1f}%)")
    
    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(
        X, y,
        batch_size=args.batch_size,
        val_split=args.val_split,
        normalize=True,
        seed=args.seed
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create or load model
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        # Resume from checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Reconstruct config from checkpoint
        if 'config' in checkpoint:
            config = TimeMixerPPConfig(**checkpoint['config'])
            logger.info("Loaded config from checkpoint")
        else:
            config = TimeMixerPPConfig(
                seq_len=48, c_in=1, d_model=args.d_model,
                n_layers=args.n_layers, n_heads=args.n_heads,
                top_k=args.top_k, dropout=args.dropout,
                pos_weight=args.pos_weight
            )
        
        # Get ablation from checkpoint or use command line
        saved_ablation = checkpoint.get('ablation', 'full')
        if saved_ablation != ablation:
            logger.warning(f"Checkpoint ablation '{saved_ablation}' differs from command line '{ablation}'")
            logger.warning(f"Using checkpoint ablation: {saved_ablation}")
            ablation = saved_ablation
        
        model = create_ablated_model(config, ablation).to(device)
        
        # Initialize dynamic layers by doing a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, config.seq_len, device=device)
            try:
                _ = model(dummy_input)
            except:
                pass
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'f1' in checkpoint['metrics']:
            best_f1 = checkpoint['metrics']['f1']
        
        logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    else:
        # Create new model
        config = TimeMixerPPConfig(
            seq_len=48,
            c_in=1,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            top_k=args.top_k,
            dropout=args.dropout,
            pos_weight=args.pos_weight
        )
        model = create_ablated_model(config, ablation).to(device)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    if ablation != 'single_scale':
        logger.info(f"Dynamic M (scales): {config.compute_dynamic_M()}")
        logger.info(f"Scale lengths: {config.get_scale_lengths()}")
    
    # Loss function
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Load optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state from checkpoint")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
            f"AUROC: {metrics['auroc']:.4f}"
        )
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            ablation_suffix = get_ablation_suffix(ablation)
            save_path = os.path.join(args.save_dir, f'best_model{ablation_suffix}.pt')
            
            # Add ablation info to checkpoint
            checkpoint_config = config.__dict__.copy()
            save_checkpoint(
                model, optimizer, epoch, metrics, save_path,
                config=checkpoint_config,
                normalizer_stats=stats
            )
            # Save ablation type separately
            checkpoint = torch.load(save_path, weights_only=False)
            checkpoint['ablation'] = ablation
            checkpoint['ablation_desc'] = ABLATION_DESCRIPTIONS[ablation]
            torch.save(checkpoint, save_path)
            
            logger.info(f"New best model! F1: {best_f1:.4f}")
        
        # Early stopping
        if early_stopping(metrics['f1']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    ablation_suffix = get_ablation_suffix(ablation)
    final_path = os.path.join(args.save_dir, f'final_model{ablation_suffix}.pt')
    
    checkpoint_config = config.__dict__.copy()
    save_checkpoint(
        model, optimizer, epoch, metrics, final_path,
        config=checkpoint_config,
        normalizer_stats=stats
    )
    # Save ablation type
    checkpoint = torch.load(final_path, weights_only=False)
    checkpoint['ablation'] = ablation
    checkpoint['ablation_desc'] = ABLATION_DESCRIPTIONS[ablation]
    torch.save(checkpoint, final_path)
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best F1: {best_f1:.4f}")
    logger.info(f"Ablation mode: {ablation} - {ABLATION_DESCRIPTIONS[ablation]}")
    logger.info(f"Models saved to: {args.save_dir}")
    logger.info(f"  Best model: best_model{ablation_suffix}.pt")
    logger.info(f"  Final model: final_model{ablation_suffix}.pt")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

