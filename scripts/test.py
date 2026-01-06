#!/usr/bin/env python
"""
Test/Evaluate script for TimeMixer++ binary classification model.

支持消融模式测试：可以测试使用消融模式训练的模型。

Evaluates a trained model on a test dataset and computes metrics.

Usage:
    python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv
    
    # 测试消融模型（会自动从 checkpoint 读取消融类型）
    python scripts/test.py --checkpoint checkpoints/best_model_no_tid.pt --test_path TDdata/TestData.csv
    
    # 手动指定消融类型（覆盖 checkpoint 中的设置）
    python scripts/test.py --checkpoint checkpoints/model.pt --test_path TDdata/TestData.csv --ablation no_tid
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
import pandas as pd
import torch
import torch.nn as nn

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict
from timemixerpp.utils import setup_logging, compute_metrics
from timemixerpp.mrti import MRTI, TimeImageInfo
from timemixerpp.tid import TID
from timemixerpp.mcm import MCM
from timemixerpp.mrm import MRM

logger = logging.getLogger(__name__)


# ============================================
# Ablation Support (same as train.py)
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
    def __init__(self, fixed_periods: List[int] = [4, 6, 8], min_period: int = 2):
        super().__init__()
        self.fixed_periods = fixed_periods
        self.mrti = MRTI(top_k=len(fixed_periods), min_period=min_period)
    
    def forward(self, multi_scale_x):
        B, device = multi_scale_x[0].shape[0], multi_scale_x[0].device
        time_images, amplitudes = [], torch.ones(B, len(self.fixed_periods), device=device)
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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)
    
    def forward(self, images):
        return [self.proj(img) for img in images], list(images)


class IdentityMCM(nn.Module):
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
    def forward(self, z_per_period, amplitudes):
        K, num_scales = len(z_per_period), len(z_per_period[0])
        return [torch.stack([z_per_period[k][m] for k in range(K)], dim=0).mean(dim=0) 
                for m in range(num_scales)]


class AblatedMixerBlock(nn.Module):
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
        self._layer_norms, self._num_scales = None, None
    
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Sequential(nn.Linear(config.c_in, config.d_model), nn.Dropout(config.dropout))
        self.blocks = nn.ModuleList([AblatedMixerBlock(config) for _ in range(config.n_layers)])
        self.head = nn.Linear(config.d_model, 1)
    
    def forward(self, x, return_features=False):
        if x.dim() == 2: x = x.unsqueeze(-1)
        x = self.embed(x)
        multi_scale_x = [x]
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        logits = self.head(multi_scale_x[0].mean(dim=1))
        result = {'logits': logits, 'probs': torch.sigmoid(logits)}
        if return_features:
            result['features'] = multi_scale_x
        return result


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
    parser = argparse.ArgumentParser(description='Test/Evaluate TimeMixer++ on a test set')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data file (.xlsx or .csv)')
    
    # Ablation arguments
    parser.add_argument('--ablation', type=str, default=None,
                        choices=list(ABLATION_DESCRIPTIONS.keys()),
                        help='Ablation mode (auto-detected from checkpoint if not specified)')
    parser.add_argument('--list_ablations', action='store_true',
                        help='List available ablation types and exit')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--output_features', action='store_true',
                        help='Also output multi-scale features')
    parser.add_argument('--features_output', type=str, default='test_features.npz',
                        help='Path to save features')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for predictions')
    parser.add_argument('--label_threshold', type=float, default=None,
                        help='Threshold for converting labels to binary (default: same as --threshold)')
    
    return parser.parse_args()


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
    setup_logging()
    logger.info("=" * 60)
    logger.info("TimeMixer++ Test/Evaluation")
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
        logger.warning("No config in checkpoint, using defaults")
        config = TimeMixerPPConfig()
    
    # Determine ablation type
    if args.ablation is not None:
        ablation = args.ablation
        logger.info(f"Using command-line ablation: {ablation}")
    else:
        ablation = checkpoint.get('ablation', 'full')
        logger.info(f"Using checkpoint ablation: {ablation}")
    
    ablation_desc = ABLATION_DESCRIPTIONS.get(ablation, ablation)
    logger.info(f"Ablation mode: {ablation} - {ablation_desc}")
    
    # Create model with ablation
    model = create_ablated_model(config, ablation).to(device)
    
    # Initialize dynamic layers by doing a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, config.seq_len, device=device)
        try:
            _ = model(dummy_input)
        except:
            pass
    
    # Now load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load normalization stats
    normalizer_mean = checkpoint.get('normalizer_mean', None)
    normalizer_std = checkpoint.get('normalizer_std', None)
    
    # Load test data
    logger.info(f"Loading test data: {args.test_path}")
    _, X_test, y_test = load_file_strict(args.test_path)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Test positive samples: {y_test.sum():.0f} ({100*y_test.mean():.1f}%)")
    
    # Normalize
    if normalizer_mean is not None and normalizer_std is not None:
        logger.info("Applying normalization from checkpoint")
        X_test = (X_test - normalizer_mean) / normalizer_std
    else:
        logger.warning("No normalization stats in checkpoint, using raw data")
    
    # Evaluation
    all_probs = []
    all_features = []
    
    n_samples = len(X_test)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    logger.info(f"Evaluating on {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, n_samples)
            
            batch_x = torch.tensor(X_test[start_idx:end_idx], dtype=torch.float32, device=device)
            
            if args.output_features:
                output = model(batch_x, return_features=True)
                features = output['features']
                batch_features = [f.cpu().numpy() for f in features]
                all_features.append(batch_features)
            else:
                output = model(batch_x)
            
            probs = output['probs'].cpu().numpy()
            all_probs.append(probs)
    
    # Combine results
    probs = np.concatenate(all_probs, axis=0).squeeze()  # (n,)
    predictions = (probs >= args.threshold).astype(int)
    
    # Compute metrics
    # Both predictions and labels are thresholded for computing classification metrics
    label_threshold = args.label_threshold if args.label_threshold is not None else args.threshold
    metrics = compute_metrics(y_test, probs, threshold=args.threshold, label_threshold=label_threshold)
    
    # Print detailed results
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    logger.info(f"  Ablation:        {ablation} - {ablation_desc}")
    logger.info(f"  Samples:         {n_samples}")
    logger.info(f"  Pred Threshold:  {args.threshold}")
    logger.info(f"  Label Threshold: {label_threshold}")
    logger.info("-" * 40)
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
    logger.info(f"  AUROC:      {metrics['auroc']:.4f}")
    logger.info("-" * 40)
    logger.info(f"  误报率 FPR: {metrics['fpr']:.4f}  (FP/(FP+TN))")
    logger.info(f"  漏报率 FNR: {metrics['fnr']:.4f}  (FN/(TP+FN))")
    logger.info("-" * 40)
    logger.info("Confusion Matrix:")
    logger.info(f"  TP: {metrics['tp']:4d}  |  FP: {metrics['fp']:4d}")
    logger.info(f"  FN: {metrics['fn']:4d}  |  TN: {metrics['tn']:4d}")
    logger.info("=" * 60)
    
    # Save predictions if requested
    if args.output:
        results = pd.DataFrame({
            'sample_id': np.arange(n_samples),
            'true_label': y_test.astype(int),
            'probability': probs,
            'prediction': predictions,
            'correct': (predictions == y_test).astype(int)
        })
        results.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to: {args.output}")
    
    # Save features if requested
    if args.output_features and all_features:
        num_scales = len(all_features[0])
        combined_features = {}
        
        for m in range(num_scales):
            scale_features = [batch[m] for batch in all_features]
            combined_features[f'scale_{m}'] = np.concatenate(scale_features, axis=0)
        
        np.savez(args.features_output, **combined_features)
        logger.info(f"Features saved to: {args.features_output}")
    
    # Return metrics for programmatic use
    return metrics


if __name__ == '__main__':
    main()

