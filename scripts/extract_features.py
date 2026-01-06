#!/usr/bin/env python
"""
Extract multi-scale features from TimeMixer++ model.

This script loads a trained model and extracts the multi-scale features
(output before the classification head) for a given dataset.

支持消融模式：可以选择去掉某些组件后提取特征。

Usage:
    # 从完整模型提取特征
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --output features/train_features.npz

    # 从消融模型提取特征（去掉TID）
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --ablation no_tid

    # 从消融模型提取特征（去掉MCM）
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --ablation no_mcm

    # 从消融模型提取特征（去掉MRM）
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --ablation no_mrm

    # 使用单尺度模型提取特征
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --ablation single_scale

    # 查看已保存的特征
    python scripts/extract_features.py --view features/train_features.npz
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict
from timemixerpp.utils import setup_logging
from timemixerpp.mrti import MRTI, TimeImageInfo
from timemixerpp.tid import TID
from timemixerpp.mcm import MCM
from timemixerpp.mrm import MRM

logger = logging.getLogger(__name__)


# ============================================
# Ablated Modules (从 ablation_study.py 复制)
# ============================================

class FixedPeriodMRTI(nn.Module):
    """MRTI with fixed periods (no FFT)."""
    
    def __init__(self, fixed_periods: List[int] = [4, 6, 8], min_period: int = 2):
        super().__init__()
        self.fixed_periods = fixed_periods
        self.min_period = min_period
        self.mrti = MRTI(top_k=len(fixed_periods), min_period=min_period)
    
    def forward(self, multi_scale_x):
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
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)
    
    def forward(self, images):
        seasonal = []
        trend = []
        for img in images:
            s = self.proj(img)
            t = img
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
            x_m = z_m_stack.mean(dim=0)
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
    
    def get_multi_scale_features(self, x):
        """Get multi-scale features (single scale for this model)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x = self.embed(x)
        multi_scale_x = [x]
        
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        
        return multi_scale_x


# ============================================
# Ablation Configuration
# ============================================

ABLATION_DESCRIPTIONS = {
    'full': '完整模型（无消融）',
    'no_fft': '使用固定周期代替FFT检测',
    'no_tid': '去掉TID（无季节性/趋势分解）',
    'no_mcm': '去掉MCM（无跨尺度混合）',
    'no_mrm': '去掉MRM（使用简单平均代替幅值加权）',
    'single_scale': '单尺度（无多尺度处理）',
}


def create_ablated_model(config: TimeMixerPPConfig, ablation: str) -> nn.Module:
    """
    Create a model with specific ablation.
    
    Args:
        config: Base configuration
        ablation: Ablation type
    
    Returns:
        Model with ablation applied
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


def get_ablation_suffix(ablation: str) -> str:
    """Get filename suffix for ablation type."""
    if ablation == 'full':
        return ''
    return f'_{ablation}'


def parse_args():
    ablation_help = "消融类型:\n"
    for k, v in ABLATION_DESCRIPTIONS.items():
        ablation_help += f"  {k}: {v}\n"
    
    parser = argparse.ArgumentParser(
        description='Extract multi-scale features from TimeMixer++ (支持消融模式)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode: extract or view
    parser.add_argument('--view', type=str, default=None,
                        help='Path to saved features file to view (skip extraction)')
    
    # Extraction arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (.xlsx or .csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save extracted features (auto-generated if not provided)')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='Directory to save features (used when --output is not provided)')
    
    # Ablation arguments
    parser.add_argument('--ablation', type=str, default='full',
                        choices=list(ABLATION_DESCRIPTIONS.keys()),
                        help=f'Ablation type. Available: {list(ABLATION_DESCRIPTIONS.keys())}')
    
    # Other arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_labels', action='store_true',
                        help='Also save labels in the output file')
    
    # List ablations
    parser.add_argument('--list_ablations', action='store_true',
                        help='List available ablation types and exit')
    
    return parser.parse_args()


def view_features(feature_path: str):
    """
    View saved multi-scale features.
    
    Args:
        feature_path: Path to .npz file containing features
    """
    logger.info(f"Loading features from: {feature_path}")
    
    data = np.load(feature_path, allow_pickle=True)
    
    print("\n" + "=" * 60)
    print(" Multi-Scale Features Summary")
    print("=" * 60)
    
    # Show ablation info if present
    if 'ablation' in data:
        ablation = str(data['ablation'])
        ablation_desc = str(data['ablation_desc']) if 'ablation_desc' in data else ''
        print(f"\nAblation: {ablation}")
        if ablation_desc:
            print(f"  Description: {ablation_desc}")
    
    # List all arrays in the file
    print(f"\nKeys in file: {list(data.keys())}")
    
    # Show shape and statistics for each scale
    scale_keys = [k for k in data.keys() if k.startswith('scale_')]
    scale_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"\nNumber of scales: {len(scale_keys)}")
    print("-" * 60)
    
    for key in scale_keys:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Min:   {arr.min():.6f}")
        print(f"  Max:   {arr.max():.6f}")
        print(f"  Mean:  {arr.mean():.6f}")
        print(f"  Std:   {arr.std():.6f}")
        
        # Show first sample's first few values
        if arr.ndim >= 2:
            print(f"  First sample, first 5 time steps, first 3 dims:")
            sample = arr[0, :5, :3] if arr.ndim == 3 else arr[0, :5]
            print(f"  {sample}")
    
    # Show labels if present
    if 'labels' in data:
        labels = data['labels']
        print(f"\nLabels:")
        print(f"  Shape: {labels.shape}")
        print(f"  Unique values: {np.unique(labels)[:10]}...")
        print(f"  Positive ratio: {labels.mean():.4f}")
    
    # Show metadata if present
    if 'config' in data:
        print(f"\nModel config: {data['config']}")
    
    print("\n" + "=" * 60)


def extract_features(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    normalizer_mean: np.ndarray = None,
    normalizer_std: np.ndarray = None
) -> dict:
    """
    Extract multi-scale features from the model.
    
    Args:
        model: TimeMixerPPForBinaryCls model
        X: Input data, shape (n, 48)
        batch_size: Batch size
        device: Device
        normalizer_mean: Normalization mean
        normalizer_std: Normalization std
        
    Returns:
        Dictionary with scale features
    """
    model.eval()
    
    # Normalize if stats provided
    if normalizer_mean is not None and normalizer_std is not None:
        X = (X - normalizer_mean) / normalizer_std
    
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Collect features for each scale
    all_features = {}
    
    logger.info(f"Extracting features from {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_x = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
            
            # Get multi-scale features
            features = model.get_multi_scale_features(batch_x)
            
            # Store features by scale
            for m, feat in enumerate(features):
                key = f'scale_{m}'
                feat_np = feat.cpu().numpy()
                
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(feat_np)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate all batches
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)
    
    return all_features


def generate_output_path(args, ablation: str) -> str:
    """
    Generate output path based on data path and ablation type.
    
    Format: {output_dir}/{data_name}_{ablation}.npz
    """
    import os
    
    if args.output is not None:
        # User specified output path, but add ablation suffix if not 'full'
        if ablation != 'full' and ablation not in args.output:
            base, ext = os.path.splitext(args.output)
            return f"{base}{get_ablation_suffix(ablation)}{ext}"
        return args.output
    
    # Auto-generate output path
    data_name = Path(args.data_path).stem  # e.g., "TrainData"
    ablation_suffix = get_ablation_suffix(ablation)
    filename = f"{data_name}_features{ablation_suffix}.npz"
    
    return os.path.join(args.output_dir, filename)


def main():
    args = parse_args()
    setup_logging()
    
    # List ablations mode
    if args.list_ablations:
        print("\n可用的消融类型:")
        print("-" * 50)
        for k, v in ABLATION_DESCRIPTIONS.items():
            print(f"  {k:<15} {v}")
        print("-" * 50)
        return
    
    # View mode
    if args.view:
        view_features(args.view)
        return
    
    # Extraction mode
    if args.checkpoint is None:
        raise ValueError("Must provide --checkpoint for feature extraction")
    if args.data_path is None:
        raise ValueError("Must provide --data_path for feature extraction")
    
    logger.info("=" * 60)
    logger.info("TimeMixer++ Feature Extraction")
    logger.info("=" * 60)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Reconstruct config
    if 'config' in checkpoint:
        config = TimeMixerPPConfig(**checkpoint['config'])
    else:
        config = TimeMixerPPConfig()
    
    # Determine ablation type: command line > checkpoint > default
    if args.ablation != 'full':
        # User explicitly specified ablation
        ablation = args.ablation
        logger.info(f"Using command-line ablation: {ablation}")
    elif 'ablation' in checkpoint:
        # Auto-detect from checkpoint
        ablation = checkpoint['ablation']
        logger.info(f"Auto-detected ablation from checkpoint: {ablation}")
    else:
        ablation = 'full'
        logger.info(f"Using default ablation: {ablation}")
    
    logger.info(f"Ablation mode: {ablation} - {ABLATION_DESCRIPTIONS[ablation]}")
    
    # Create model with ablation
    model = create_ablated_model(config, ablation).to(device)
    
    # Initialize dynamic layers
    with torch.no_grad():
        dummy_input = torch.randn(1, config.seq_len, device=device)
        try:
            _ = model(dummy_input)
        except Exception as e:
            logger.warning(f"Dummy forward pass warning: {e}")
    
    # Load weights (with strict=False to handle ablated model differences)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed, trying non-strict: {e}")
        # For ablated models, some weights may not match
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()
    
    if ablation == 'single_scale':
        logger.info(f"Model loaded (single scale mode)")
    else:
        logger.info(f"Model loaded. M={config.compute_dynamic_M()}, scales={config.get_scale_lengths()}")
    
    # Load normalization stats
    normalizer_mean = checkpoint.get('normalizer_mean', None)
    normalizer_std = checkpoint.get('normalizer_std', None)
    
    # Load data
    logger.info(f"Loading data: {args.data_path}")
    _, X, y = load_file_strict(args.data_path)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Extract features
    features = extract_features(
        model, X, args.batch_size, device,
        normalizer_mean, normalizer_std
    )
    
    # Add labels and config if requested
    if args.save_labels:
        features['labels'] = y
    
    # Save ablation info in features
    features['config'] = np.array(str(checkpoint.get('config', {})))
    features['ablation'] = np.array(ablation)
    features['ablation_desc'] = np.array(ABLATION_DESCRIPTIONS[ablation])
    
    # Generate output path
    output_path = generate_output_path(args, ablation)
    
    # Save features
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez(output_path, **features)
    
    logger.info(f"Features saved to: {output_path}")
    
    # Show summary
    print("\n" + "=" * 60)
    print(" Extracted Features Summary")
    print("=" * 60)
    print(f"  Ablation: {ablation} - {ABLATION_DESCRIPTIONS[ablation]}")
    print("-" * 60)
    for key, arr in features.items():
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            print(f"  {key}: {arr.shape}")
    print("=" * 60)
    
    # Optionally view
    print(f"\nTo view features: python scripts/extract_features.py --view {output_path}")


if __name__ == '__main__':
    main()

