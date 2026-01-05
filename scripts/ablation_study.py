#!/usr/bin/env python
"""
Ablation study for TimeMixer++ model.

This script systematically removes or modifies components of TimeMixer++
to analyze their contribution to model performance.

Ablation experiments:
1. Remove MRTI (use fixed periods instead of FFT-detected)
2. Remove TID (no seasonal/trend decomposition)
3. Remove MCM (no cross-scale mixing)
4. Remove MRM (no multi-resolution aggregation)
5. Single scale (no multi-scale)
6. Different top_k values
7. Different number of layers
8. Different d_model sizes

Usage:
    python scripts/ablation_study.py --data_path TDdata/TrainData.csv --epochs 50
    python scripts/ablation_study.py --data_path TDdata/TrainData.csv --ablations no_mrti no_tid
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict, create_dataloaders
from timemixerpp.utils import set_seed, compute_metrics, setup_logging, AverageMeter
from timemixerpp.block import MixerBlock
from timemixerpp.mrti import MRTI, TimeImageInfo
from timemixerpp.tid import TID
from timemixerpp.mcm import MCM
from timemixerpp.mrm import MRM

logger = logging.getLogger(__name__)


# ============================================
# Ablated Modules
# ============================================

class FixedPeriodMRTI(nn.Module):
    """MRTI with fixed periods (no FFT)."""
    
    def __init__(self, fixed_periods: List[int] = [4, 6, 8], min_period: int = 2):
        super().__init__()
        self.fixed_periods = fixed_periods
        self.min_period = min_period
        self.mrti = MRTI(top_k=len(fixed_periods), min_period=min_period)
    
    def forward(self, multi_scale_x):
        # Use fixed periods instead of FFT-detected ones
        M = len(multi_scale_x) - 1
        B = multi_scale_x[0].shape[0]
        device = multi_scale_x[0].device
        
        time_images = []
        periods = self.fixed_periods
        amplitudes = torch.ones(B, len(periods), device=device)
        
        for k, period in enumerate(periods):
            images = []
            original_lengths = []
            
            for m, x_m in enumerate(multi_scale_x):
                image, orig_len = self.mrti.reshape_1d_to_2d(x_m, period)
                images.append(image)
                original_lengths.append(orig_len)
            
            time_images.append(TimeImageInfo(
                period=period,
                amplitude=amplitudes[:, k],
                images=images,
                original_lengths=original_lengths
            ))
        
        return time_images, periods, amplitudes


class IdentityTID(nn.Module):
    """TID that returns input as both seasonal and trend (no decomposition)."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Just a linear layer to maintain parameter count
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)
    
    def forward(self, images):
        seasonal = []
        trend = []
        for img in images:
            s = self.proj(img)
            t = img  # Identity
            seasonal.append(s)
            trend.append(t)
        return seasonal, trend


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
            z_1d = z_1d[:, :original_lengths[m], :]
            z_list.append(z_1d)
        return z_list


class IdentityMRM(nn.Module):
    """MRM that uses simple average instead of amplitude weighting."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, z_per_period, amplitudes):
        K = len(z_per_period)
        num_scales = len(z_per_period[0])
        
        x_list = []
        for m in range(num_scales):
            z_m_list = [z_per_period[k][m] for k in range(K)]
            z_m_stack = torch.stack(z_m_list, dim=0)
            x_m = z_m_stack.mean(dim=0)  # Simple average
            x_list.append(x_m)
        
        return x_list


class AblatedMixerBlock(nn.Module):
    """MixerBlock with configurable ablations."""
    
    def __init__(
        self,
        config: TimeMixerPPConfig,
        use_fft_periods: bool = True,
        use_tid: bool = True,
        use_mcm: bool = True,
        use_mrm: bool = True,
        fixed_periods: List[int] = [4, 6, 8]
    ):
        super().__init__()
        self.config = config
        
        # MRTI
        if use_fft_periods:
            self.mrti = MRTI(
                top_k=config.top_k,
                base_len_for_period=config.base_len_for_period,
                min_period=config.min_period
            )
        else:
            self.mrti = FixedPeriodMRTI(fixed_periods=fixed_periods)
        
        # TID
        if use_tid:
            self.tid = TID(config.d_model, config.n_heads, config.dropout)
        else:
            self.tid = IdentityTID(config.d_model, config.n_heads, config.dropout)
        
        # MCM
        if use_mcm:
            self.mcm = MCM(config.d_model, config.mcm_kernel_size, config.mcm_use_two_stride_layers)
        else:
            self.mcm = IdentityMCM(config.d_model)
        
        # MRM
        if use_mrm:
            self.mrm = MRM(weight_mode=config.weight_mode)
        else:
            self.mrm = IdentityMRM()
        
        self._layer_norms = None
        self._num_scales = None
    
    def _init_layer_norms(self, num_scales: int, device: torch.device):
        if self._num_scales == num_scales:
            return
        self._num_scales = num_scales
        self._layer_norms = nn.ModuleList([
            nn.LayerNorm(self.config.d_model) for _ in range(num_scales)
        ]).to(device)
    
    def forward(self, multi_scale_x):
        num_scales = len(multi_scale_x)
        device = multi_scale_x[0].device
        self._init_layer_norms(num_scales, device)
        
        residuals = multi_scale_x
        
        # MRTI
        time_images, periods, amplitudes = self.mrti(multi_scale_x)
        
        # TID + MCM for each period
        z_per_period = []
        for ti in time_images:
            seasonal, trend = self.tid(ti.images)
            z_list = self.mcm(seasonal, trend, ti.original_lengths, ti.period)
            z_per_period.append(z_list)
        
        # MRM
        x_out = self.mrm(z_per_period, amplitudes)
        
        # Residual + LayerNorm
        output = []
        for m in range(num_scales):
            out_m = residuals[m] + x_out[m]
            out_m = self._layer_norms[m](out_m)
            output.append(out_m)
        
        return output


class SingleScaleModel(nn.Module):
    """TimeMixer++ without multi-scale (single scale only)."""
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Sequential(
            nn.Linear(config.c_in, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.blocks = nn.ModuleList([
            AblatedMixerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(config.d_model, 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x = self.embed(x)
        
        # Only use single scale (no downsampling)
        multi_scale_x = [x]
        
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        
        # Pool and project
        pooled = multi_scale_x[0].mean(dim=1)
        logits = self.head(pooled)
        
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


def create_ablated_model(
    config: TimeMixerPPConfig,
    ablation: str
) -> nn.Module:
    """
    Create a model with specific ablation.
    
    Args:
        config: Base configuration
        ablation: Ablation type:
            - 'full': Full model (no ablation)
            - 'no_fft': Fixed periods instead of FFT
            - 'no_tid': No TID decomposition
            - 'no_mcm': No MCM cross-scale mixing
            - 'no_mrm': No MRM amplitude weighting
            - 'single_scale': Single scale only
    """
    if ablation == 'full':
        return TimeMixerPPForBinaryCls(config)
    
    if ablation == 'single_scale':
        return SingleScaleModel(config)
    
    # Create model with ablated blocks
    model = TimeMixerPPForBinaryCls(config)
    
    # Replace blocks with ablated versions
    ablation_kwargs = {
        'use_fft_periods': ablation != 'no_fft',
        'use_tid': ablation != 'no_tid',
        'use_mcm': ablation != 'no_mcm',
        'use_mrm': ablation != 'no_mrm',
    }
    
    model.encoder.blocks = nn.ModuleList([
        AblatedMixerBlock(config, **ablation_kwargs)
        for _ in range(config.n_layers)
    ])
    
    return model


# ============================================
# Ablation Registry
# ============================================

@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    ablation_type: str
    config_override: Dict[str, Any] = None


ABLATION_REGISTRY: Dict[str, AblationConfig] = {
    'full': AblationConfig('Full Model', 'Complete TimeMixer++ model', 'full'),
    'no_fft': AblationConfig('No FFT', 'Fixed periods instead of FFT-detected', 'no_fft'),
    'no_tid': AblationConfig('No TID', 'No seasonal/trend decomposition', 'no_tid'),
    'no_mcm': AblationConfig('No MCM', 'No cross-scale mixing', 'no_mcm'),
    'no_mrm': AblationConfig('No MRM', 'Simple average instead of amplitude weighting', 'no_mrm'),
    'single_scale': AblationConfig('Single Scale', 'No multi-scale processing', 'single_scale'),
    'top_k_1': AblationConfig('Top-K=1', 'Only 1 frequency/period', 'full', {'top_k': 1}),
    'top_k_5': AblationConfig('Top-K=5', '5 frequencies/periods', 'full', {'top_k': 5}),
    'layers_1': AblationConfig('1 Layer', 'Single MixerBlock layer', 'full', {'n_layers': 1}),
    'layers_4': AblationConfig('4 Layers', 'Four MixerBlock layers', 'full', {'n_layers': 4}),
    'd_model_32': AblationConfig('d_model=32', 'Smaller hidden dimension', 'full', {'d_model': 32}),
    'd_model_128': AblationConfig('d_model=128', 'Larger hidden dimension', 'full', {'d_model': 128}),
}


def list_ablations() -> List[str]:
    """List all available ablation experiments."""
    return list(ABLATION_REGISTRY.keys())


# ============================================
# Training
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
    """Train a model and return best metrics."""
    model = model.to(device)
    
    # Initialize dynamic layers
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            try:
                _ = model(batch_x)
            except:
                pass
            break
    
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    best_metrics = {}
    
    for epoch in range(epochs):
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
        all_preds, all_labels = [], []
        
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
    parser = argparse.ArgumentParser(description='TimeMixer++ Ablation Study')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--ablations', nargs='+', default=None,
                        help=f'Ablations to run. Available: {list_ablations()}. Default: all')
    
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='Base hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Base number of layers')
    parser.add_argument('--top_k', type=int, default=3, help='Base top-K')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--output', type=str, default='results/ablation_study.json',
                        help='Path to save results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("TimeMixer++ Ablation Study")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data: {args.data_path}")
    _, X, y = load_file_strict(args.data_path)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    train_loader, val_loader, _ = create_dataloaders(
        X, y, batch_size=args.batch_size, val_split=args.val_split,
        normalize=True, seed=args.seed
    )
    
    # Select ablations
    ablation_names = args.ablations if args.ablations else list_ablations()
    logger.info(f"Ablations to run: {ablation_names}")
    
    # Base config
    base_config = TimeMixerPPConfig(
        seq_len=48, c_in=1, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=4, top_k=args.top_k, dropout=0.1
    )
    
    results = {}
    
    for name in ablation_names:
        if name not in ABLATION_REGISTRY:
            logger.warning(f"Unknown ablation: {name}, skipping")
            continue
        
        ablation = ABLATION_REGISTRY[name]
        logger.info(f"\nRunning ablation: {name} - {ablation.description}")
        
        set_seed(args.seed)
        
        # Create config with overrides
        config_dict = copy.deepcopy(base_config.__dict__)
        if ablation.config_override:
            config_dict.update(ablation.config_override)
        config = TimeMixerPPConfig(**config_dict)
        
        # Create model
        model = create_ablated_model(config, ablation.ablation_type)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        # Train
        metrics = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device, model_name=name
        )
        
        results[name] = {
            'description': ablation.description,
            'params': n_params,
            **metrics
        }
    
    # Print results
    print("\n" + "=" * 90)
    print(" Ablation Study Results")
    print("=" * 90)
    print(f"{'Ablation':<20} {'Description':<30} {'Params':>10} {'Acc':>8} {'F1':>8} {'AUROC':>8}")
    print("-" * 90)
    
    # Sort by F1 descending
    for name, res in sorted(results.items(), key=lambda x: -x[1].get('f1', 0)):
        desc = res['description'][:28] + '..' if len(res['description']) > 30 else res['description']
        print(f"{name:<20} {desc:<30} {res['params']:>10,} {res['accuracy']:>8.4f} "
              f"{res['f1']:>8.4f} {res['auroc']:>8.4f}")
    
    print("=" * 90)
    
    # Compute relative performance
    if 'full' in results:
        full_f1 = results['full']['f1']
        print("\nRelative F1 (vs Full Model):")
        for name, res in results.items():
            if name != 'full':
                diff = res['f1'] - full_f1
                pct = (diff / full_f1) * 100
                print(f"  {name}: {diff:+.4f} ({pct:+.1f}%)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

