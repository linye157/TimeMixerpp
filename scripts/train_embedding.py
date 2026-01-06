#!/usr/bin/env python
"""
训练序列级 embedding 编码器（SupCon + 可选 BCE）。

使用已提取的多尺度特征训练 TemporalConvEmbedder，
输出可用于向量检索的 L2 归一化 embedding。

Usage:
    python scripts/train_embedding.py --npz_path features/alldata_features_no_tid.npz --out_dir runs/emb_exp1 --epochs 20 --batch_size 256 --lr 1e-3 --use_bce true --lambda_bce 0.5 --balanced_sampling true
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from timemixerpp.metric_encoder import TemporalConvEmbedder, MultiScaleEmbedder
from timemixerpp.losses import MultiScaleSupConLoss
from timemixerpp.data import (
    NPZMultiScaleDataset, create_splits, load_splits,
    create_multiscale_dataloaders
)
from timemixerpp.utils import set_seed, setup_logging, AverageMeter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train sequence-level embedding encoder with SupCon',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to NPZ file with multi-scale features')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to existing splits.json (optional)')
    parser.add_argument('--split_ratio', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratio (comma-separated)')
    
    # Model arguments
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for conv layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Loss arguments
    parser.add_argument('--tau', type=float, default=0.07,
                        help='Temperature for SupCon loss')
    parser.add_argument('--use_bce', type=str, default='false',
                        help='Whether to use BCE loss (true/false)')
    parser.add_argument('--lambda_bce', type=float, default=0.5,
                        help='Weight for BCE loss')
    parser.add_argument('--scale_weights', type=str, default='0.5,0.3,0.2',
                        help='Weights for each scale (comma-separated)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--balanced_sampling', type=str, default='false',
                        help='Whether to use balanced sampling (true/false)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Returns metrics including accuracy, F1, AUROC.
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x0, x1, x2, labels, _ = batch
            x0, x1, x2 = x0.to(device), x1.to(device), x2.to(device)
            
            out = model(x0, x1, x2)
            
            # Use scale 0 logits for evaluation (if available)
            if 'logits0' in out:
                probs = torch.sigmoid(out['logits0']).squeeze(-1)
            else:
                # Use embedding norm as proxy (not ideal, but fallback)
                probs = (out['e0'].norm(dim=-1) > 0.5).float()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Binary predictions
    preds = (all_probs >= threshold).astype(int)
    binary_labels = (all_labels >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(binary_labels, preds),
        'f1': f1_score(binary_labels, preds, zero_division=0),
        'auroc': roc_auc_score(binary_labels, all_probs) if len(np.unique(binary_labels)) > 1 else 0.0,
    }
    
    return metrics


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    supcon_meter = AverageMeter()
    bce_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(loader):
        x0, x1, x2, labels, _ = batch
        x0, x1, x2, labels = x0.to(device), x1.to(device), x2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        out = model(x0, x1, x2)
        
        # Compute loss
        loss_dict = criterion(
            e0=out['e0'],
            e1=out['e1'],
            e2=out['e2'],
            labels=labels,
            logits0=out.get('logits0'),
            logits1=out.get('logits1'),
            logits2=out.get('logits2')
        )
        
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update meters
        batch_size = x0.size(0)
        loss_meter.update(loss.item(), batch_size)
        supcon_meter.update(loss_dict['supcon'].item(), batch_size)
        if 'bce' in loss_dict:
            bce_meter.update(loss_dict['bce'].item(), batch_size)
    
    return {
        'loss': loss_meter.avg,
        'supcon': supcon_meter.avg,
        'bce': bce_meter.avg if bce_meter.count > 0 else 0.0
    }


def main():
    args = parse_args()
    
    # Parse boolean arguments
    use_bce = str_to_bool(args.use_bce)
    balanced_sampling = str_to_bool(args.balanced_sampling)
    
    # Parse list arguments
    split_ratio = tuple(float(x) for x in args.split_ratio.split(','))
    scale_weights = tuple(float(x) for x in args.scale_weights.split(','))
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, 'train.log'))
    
    logger.info("=" * 60)
    logger.info("Embedding Encoder Training (SupCon)")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info(f"Loading data from: {args.npz_path}")
    
    # Handle splits path
    splits_path = args.splits_path
    if splits_path is None:
        splits_path = os.path.join(args.out_dir, 'splits.json')
    
    train_loader, val_loader, test_loader, splits = create_multiscale_dataloaders(
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        splits_path=splits_path,
        split_ratio=split_ratio,
        balanced_sampling=balanced_sampling,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    logger.info("Creating model...")
    model = MultiScaleEmbedder(
        input_dim=64,  # Feature dimension from TimeMixer++
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_classification_head=use_bce
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss
    criterion = MultiScaleSupConLoss(
        temperature=args.tau,
        scale_weights=scale_weights,
        use_bce=use_bce,
        lambda_bce=args.lambda_bce
    )
    
    logger.info(f"Loss: SupCon (tau={args.tau}) + BCE (use={use_bce}, lambda={args.lambda_bce})")
    logger.info(f"Scale weights: {scale_weights}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    history = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {train_metrics['loss']:.4f} (SupCon: {train_metrics['supcon']:.4f}, BCE: {train_metrics['bce']:.4f}) - "
            f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}"
        )
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_supcon': train_metrics['supcon'],
            'train_bce': train_metrics['bce'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'val_auroc': val_metrics['auroc']
        })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': {
                    'input_dim': 64,
                    'hidden_dim': args.hidden_dim,
                    'emb_dim': args.emb_dim,
                    'dropout': args.dropout,
                    'use_classification_head': use_bce,
                    'tau': args.tau,
                    'scale_weights': scale_weights,
                    'use_bce': use_bce,
                    'lambda_bce': args.lambda_bce,
                },
                'fusion_logits': model.fusion_logits.detach().cpu().numpy().tolist(),
            }
            
            torch.save(checkpoint, os.path.join(args.out_dir, 'checkpoint.pt'))
            logger.info(f"  -> New best model! F1: {best_f1:.4f}")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoint.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
    
    # Save metrics
    metrics_output = {
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_auroc': test_metrics['auroc'],
        'history': history,
        'config': {
            'npz_path': args.npz_path,
            'emb_dim': args.emb_dim,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'tau': args.tau,
            'use_bce': use_bce,
            'lambda_bce': args.lambda_bce,
            'scale_weights': scale_weights,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'balanced_sampling': balanced_sampling,
            'seed': args.seed,
        }
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger.info(f"\nMetrics saved to: {os.path.join(args.out_dir, 'metrics.json')}")
    logger.info(f"Checkpoint saved to: {os.path.join(args.out_dir, 'checkpoint.pt')}")
    logger.info(f"Splits saved to: {splits_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model at epoch {best_epoch} with val F1: {best_f1:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

