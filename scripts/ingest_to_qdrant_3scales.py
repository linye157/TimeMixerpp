#!/usr/bin/env python
"""
将多尺度 embedding 入库到 Qdrant 三个 collection。

从训练好的 embedding encoder 提取 embedding，
分别存入 {prefix}_scale0, {prefix}_scale1, {prefix}_scale2。

Usage:

# 方式1：入库全部 NPZ 数据（推荐，无需 splits.json）
python scripts/ingest_to_qdrant_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --use_all_data --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --batch_size 256

    
# 方式2：按 splits.json 入库指定划分
python scripts/ingest_to_qdrant_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --splits_path runs/emb_exp1/splits.json --split train --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --batch_size 256
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
import os
from typing import Dict, List, Any, Optional

import numpy as np
import torch

from timemixerpp.metric_encoder import MultiScaleEmbedder
from timemixerpp.data import NPZMultiScaleDataset, load_splits
from timemixerpp.qdrant_utils import (
    get_client, create_or_validate_collection, upsert_points
)
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ingest multi-scale embeddings to Qdrant',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to NPZ file with multi-scale features')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained embedding encoder checkpoint')
    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to splits.json file (optional, if not provided use --use_all_data)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to ingest (ignored if --use_all_data)')
    parser.add_argument('--use_all_data', action='store_true',
                        help='Ingest all data from NPZ file (ignores splits_path and split)')
    
    # Qdrant arguments
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection_prefix', type=str, default='accident_kb',
                        help='Prefix for collection names')
    parser.add_argument('--recreate', action='store_true',
                        help='Recreate collections if they exist')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for embedding extraction and upserting')
    parser.add_argument('--id_offset', type=int, default=0,
                        help='Offset to add to point IDs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()


def extract_embeddings(
    model: torch.nn.Module,
    dataset: NPZMultiScaleDataset,
    batch_size: int,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from dataset.
    
    Returns:
        Dictionary with:
        - e0, e1, e2: embeddings for each scale (N, emb_dim)
        - attn0, attn1, attn2: attention weights (N, L)
        - top_ts0, top_ts1, top_ts2: top timesteps (N, 3)
        - labels: labels (N,)
        - indices: sample indices (N,)
    """
    model.eval()
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = {
        'e0': [], 'e1': [], 'e2': [],
        'attn0': [], 'attn1': [], 'attn2': [],
        'top_ts0': [], 'top_ts1': [], 'top_ts2': [],
        'labels': [], 'indices': []
    }
    
    logger.info(f"Extracting embeddings from {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x0, x1, x2, labels, indices = batch
            x0, x1, x2 = x0.to(device), x1.to(device), x2.to(device)
            
            out = model(x0, x1, x2, return_attention=True)
            
            results['e0'].append(out['e0'].cpu().numpy())
            results['e1'].append(out['e1'].cpu().numpy())
            results['e2'].append(out['e2'].cpu().numpy())
            
            results['attn0'].append(out['attn0'].cpu().numpy())
            results['attn1'].append(out['attn1'].cpu().numpy())
            results['attn2'].append(out['attn2'].cpu().numpy())
            
            results['top_ts0'].append(out['top_timesteps0'].cpu().numpy())
            results['top_ts1'].append(out['top_timesteps1'].cpu().numpy())
            results['top_ts2'].append(out['top_timesteps2'].cpu().numpy())
            
            results['labels'].append(labels.numpy())
            results['indices'].append(indices.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples")
    
    # Concatenate
    for key in results:
        results[key] = np.concatenate(results[key], axis=0)
    
    return results


def create_payloads(
    indices: np.ndarray,
    labels: np.ndarray,
    top_timesteps: np.ndarray,
    scale: int,
    emb_dim: int,
    ckpt_basename: str
) -> List[Dict[str, Any]]:
    """
    Create payload dictionaries for Qdrant points.
    
    Payload includes:
    - label: binary label
    - sample_id: original sample index
    - scale: scale index (0, 1, 2)
    - embed_dim: embedding dimension
    - ckpt_path: checkpoint basename
    - attn_top_timesteps: top-3 attended timestep indices
    """
    payloads = []
    
    for i in range(len(indices)):
        payload = {
            'label': int(labels[i] >= 0.5),  # Binary
            'label_raw': float(labels[i]),
            'sample_id': int(indices[i]),
            'scale': scale,
            'embed_dim': emb_dim,
            'ckpt': ckpt_basename,
            'attn_top_timesteps': top_timesteps[i].tolist(),
        }
        payloads.append(payload)
    
    return payloads


def main():
    args = parse_args()
    
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Qdrant 3-Scale Ingestion")
    logger.info("=" * 60)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = MultiScaleEmbedder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        emb_dim=config['emb_dim'],
        dropout=config.get('dropout', 0.1),
        use_classification_head=config.get('use_classification_head', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    emb_dim = config['emb_dim']
    logger.info(f"Model loaded. Embedding dim: {emb_dim}")
    
    # Determine which samples to ingest
    if args.use_all_data:
        # Use all data from NPZ
        logger.info("Using all data from NPZ file (--use_all_data)")
        dataset = NPZMultiScaleDataset(args.npz_path, indices=None)
        split_name = "all_npz"
    elif args.splits_path is not None:
        # Load splits
        logger.info(f"Loading splits: {args.splits_path}")
        splits = load_splits(args.splits_path)
        
        # Get indices for the requested split
        if args.split == 'all':
            indices = splits['train_ids'] + splits['val_ids'] + splits['test_ids']
        else:
            indices = splits[f'{args.split}_ids']
        
        logger.info(f"Processing split '{args.split}' with {len(indices)} samples")
        dataset = NPZMultiScaleDataset(args.npz_path, indices=indices)
        split_name = args.split
    else:
        # No splits_path and no use_all_data - default to all data
        logger.info("No splits_path provided, using all data from NPZ file")
        dataset = NPZMultiScaleDataset(args.npz_path, indices=None)
        split_name = "all_npz"
    
    logger.info(f"Total samples to ingest: {len(dataset)}")
    
    # Extract embeddings
    embeddings = extract_embeddings(model, dataset, args.batch_size, device)
    
    logger.info(f"Extracted embeddings shapes:")
    logger.info(f"  e0: {embeddings['e0'].shape}")
    logger.info(f"  e1: {embeddings['e1'].shape}")
    logger.info(f"  e2: {embeddings['e2'].shape}")
    
    # Connect to Qdrant
    client = get_client(args.qdrant_url)
    
    # Collection names
    collection_names = [
        f"{args.collection_prefix}_scale0",
        f"{args.collection_prefix}_scale1",
        f"{args.collection_prefix}_scale2"
    ]
    
    # Create collections
    for coll_name in collection_names:
        create_or_validate_collection(
            client, coll_name, emb_dim,
            distance="Cosine", recreate=args.recreate
        )
    
    # Checkpoint basename for payload
    ckpt_basename = Path(args.ckpt_path).name
    
    # Upsert to each scale collection
    scale_data = [
        (0, embeddings['e0'], embeddings['top_ts0']),
        (1, embeddings['e1'], embeddings['top_ts1']),
        (2, embeddings['e2'], embeddings['top_ts2']),
    ]
    
    for scale, embs, top_ts in scale_data:
        coll_name = collection_names[scale]
        
        # Create IDs (with offset)
        ids = (embeddings['indices'] + args.id_offset).tolist()
        
        # Create vectors
        vectors = embs.tolist()
        
        # Create payloads
        payloads = create_payloads(
            embeddings['indices'],
            embeddings['labels'],
            top_ts,
            scale=scale,
            emb_dim=emb_dim,
            ckpt_basename=ckpt_basename
        )
        
        # Upsert
        n_upserted = upsert_points(
            client, coll_name, ids, vectors, payloads,
            batch_size=args.batch_size
        )
        
        logger.info(f"Upserted {n_upserted} points to '{coll_name}'")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info(f"Collections created/updated:")
    for coll_name in collection_names:
        from timemixerpp.qdrant_utils import get_collection_info
        info = get_collection_info(client, coll_name)
        logger.info(f"  {coll_name}: {info['points_count']} points")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

