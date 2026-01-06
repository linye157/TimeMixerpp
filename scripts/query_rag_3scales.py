#!/usr/bin/env python
"""
三尺度 RAG 查询脚本。

从 Qdrant 三个 collection 检索相似样本，
计算尺度内概率，融合后输出解释。

Usage:
    python scripts/query_rag_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --query_index 123 --top_k 10 --gamma 10 --fusion_mode fixed --w0 0.5 --w1 0.3 --w2 0.2 --json_output true
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from timemixerpp.metric_encoder import MultiScaleEmbedder
from timemixerpp.qdrant_utils import get_client, search_similar
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Query 3-scale RAG system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to NPZ file with multi-scale features')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained embedding encoder checkpoint')
    parser.add_argument('--query_index', type=int, required=True,
                        help='Index of the query sample in NPZ')
    
    # Qdrant arguments
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection_prefix', type=str, default='accident_kb',
                        help='Prefix for collection names')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of similar samples to retrieve')
    
    # Fusion arguments
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='Gamma for similarity weighting: w_i = exp(gamma * score_i)')
    parser.add_argument('--fusion_mode', type=str, default='fixed',
                        choices=['fixed', 'learned'],
                        help='Fusion mode: fixed weights or learned from checkpoint')
    parser.add_argument('--w0', type=float, default=0.5,
                        help='Weight for scale 0 (fixed mode)')
    parser.add_argument('--w1', type=float, default=0.3,
                        help='Weight for scale 1 (fixed mode)')
    parser.add_argument('--w2', type=float, default=0.2,
                        help='Weight for scale 2 (fixed mode)')
    
    # Output arguments
    parser.add_argument('--json_output', type=str, default='false',
                        help='Output JSON format (true/false)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save JSON output (optional)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def compute_scale_probability(
    search_results: List[Dict[str, Any]],
    gamma: float
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute probability from search results using similarity-weighted voting.
    
    p_m = Σ w_i * label_i / Σ w_i
    where w_i = exp(gamma * score_i)
    
    Args:
        search_results: List of search results with 'score' and 'payload'
        gamma: Temperature for similarity weighting
        
    Returns:
        probability: Weighted probability
        weighted_results: Results with weights added
    """
    if not search_results:
        return 0.5, []
    
    weighted_results = []
    total_weight = 0.0
    weighted_label_sum = 0.0
    
    for result in search_results:
        score = result['score']
        label = result['payload']['label']  # Binary label (0 or 1)
        
        weight = np.exp(gamma * score)
        total_weight += weight
        weighted_label_sum += weight * label
        
        weighted_results.append({
            **result,
            'weight': weight
        })
    
    probability = weighted_label_sum / (total_weight + 1e-12)
    
    # Normalize weights for display
    for r in weighted_results:
        r['weight_normalized'] = r['weight'] / (total_weight + 1e-12)
    
    return probability, weighted_results


def format_results_for_display(
    query_info: Dict[str, Any],
    scale_results: List[Tuple[float, List[Dict]]],
    fusion_result: Dict[str, Any]
) -> str:
    """Format results for console display."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(" 三尺度 RAG 查询结果")
    lines.append("=" * 70)
    
    # Query info
    lines.append(f"\n查询样本:")
    lines.append(f"  Index: {query_info['index']}")
    lines.append(f"  真实标签: {query_info['label']:.4f} ({'正类' if query_info['label'] >= 0.5 else '负类'})")
    
    # Scale results
    scale_names = ['Scale 0 (48 时间步)', 'Scale 1 (24 时间步)', 'Scale 2 (12 时间步)']
    
    for scale_idx, (prob, results) in enumerate(scale_results):
        lines.append(f"\n{'-' * 70}")
        lines.append(f" {scale_names[scale_idx]} - 概率: {prob:.4f}")
        lines.append(f"{'-' * 70}")
        lines.append(f"{'Rank':<6}{'ID':<8}{'Label':<8}{'Score':<10}{'Weight':<10}")
        lines.append("-" * 42)
        
        for r in results[:5]:  # Show top 5
            lines.append(
                f"{r['rank']:<6}"
                f"{r['id']:<8}"
                f"{r['payload']['label']:<8}"
                f"{r['score']:<10.4f}"
                f"{r['weight_normalized']:<10.4f}"
            )
        
        if len(results) > 5:
            lines.append(f"  ... 还有 {len(results) - 5} 个结果")
    
    # Fusion result
    lines.append(f"\n{'=' * 70}")
    lines.append(" 融合结果")
    lines.append("=" * 70)
    lines.append(f"  融合模式: {fusion_result['mode']}")
    lines.append(f"  尺度权重: w0={fusion_result['weights'][0]:.3f}, w1={fusion_result['weights'][1]:.3f}, w2={fusion_result['weights'][2]:.3f}")
    lines.append(f"  各尺度概率: p0={scale_results[0][0]:.4f}, p1={scale_results[1][0]:.4f}, p2={scale_results[2][0]:.4f}")
    lines.append(f"  融合概率: {fusion_result['probability']:.4f}")
    lines.append(f"  预测: {'正类 (事故风险高)' if fusion_result['probability'] >= 0.5 else '负类 (事故风险低)'}")
    
    # Comparison with true label
    pred = 1 if fusion_result['probability'] >= 0.5 else 0
    true = 1 if query_info['label'] >= 0.5 else 0
    correct = "✓ 正确" if pred == true else "✗ 错误"
    lines.append(f"  与真实标签对比: {correct}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    json_output = str_to_bool(args.json_output)
    
    if not json_output:
        setup_logging()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if not json_output:
        logger.info(f"Using device: {device}")
    
    # Load checkpoint
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
    
    # Determine fusion weights
    if args.fusion_mode == 'learned':
        if 'fusion_logits' in checkpoint:
            fusion_logits = torch.tensor(checkpoint['fusion_logits'])
            weights = torch.softmax(fusion_logits, dim=0).numpy().tolist()
        else:
            raise ValueError(
                "Learned fusion mode requested but 'fusion_logits' not found in checkpoint. "
                "Please use --fusion_mode fixed or retrain with a model that saves fusion weights."
            )
    else:
        # Normalize fixed weights
        total = args.w0 + args.w1 + args.w2
        weights = [args.w0 / total, args.w1 / total, args.w2 / total]
    
    # Load query data from NPZ
    data = np.load(args.npz_path, allow_pickle=True)
    
    query_idx = args.query_index
    x0 = torch.tensor(data['scale_0'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    x1 = torch.tensor(data['scale_1'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    x2 = torch.tensor(data['scale_2'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    query_label = float(data['labels'][query_idx])
    
    # Get query embeddings
    with torch.no_grad():
        out = model(x0, x1, x2, return_attention=True)
        e0 = out['e0'][0].cpu().numpy().tolist()
        e1 = out['e1'][0].cpu().numpy().tolist()
        e2 = out['e2'][0].cpu().numpy().tolist()
    
    query_info = {
        'index': query_idx,
        'label': query_label
    }
    
    # Connect to Qdrant
    client = get_client(args.qdrant_url)
    
    # Collection names
    collection_names = [
        f"{args.collection_prefix}_scale0",
        f"{args.collection_prefix}_scale1",
        f"{args.collection_prefix}_scale2"
    ]
    
    # Query each scale
    query_embeddings = [e0, e1, e2]
    scale_results = []
    
    for scale_idx, (coll_name, query_emb) in enumerate(zip(collection_names, query_embeddings)):
        # Search
        results = search_similar(
            client, coll_name, query_emb,
            top_k=args.top_k,
            with_payload=True
        )
        
        # Compute probability
        prob, weighted_results = compute_scale_probability(results, args.gamma)
        scale_results.append((prob, weighted_results))
    
    # Fuse probabilities
    p0, p1, p2 = scale_results[0][0], scale_results[1][0], scale_results[2][0]
    fused_prob = weights[0] * p0 + weights[1] * p1 + weights[2] * p2
    
    fusion_result = {
        'mode': args.fusion_mode,
        'weights': weights,
        'probability': fused_prob,
        'prediction': 1 if fused_prob >= 0.5 else 0
    }
    
    # Output
    if json_output:
        output = {
            'query': query_info,
            'scale_results': [
                {
                    'scale': i,
                    'probability': prob,
                    'top_k': [
                        {
                            'rank': r['rank'],
                            'id': r['id'],
                            'label': r['payload']['label'],
                            'score': r['score'],
                            'weight': r['weight_normalized'],
                            'sample_id': r['payload']['sample_id'],
                            'attn_top_timesteps': r['payload'].get('attn_top_timesteps', [])
                        }
                        for r in results
                    ]
                }
                for i, (prob, results) in enumerate(scale_results)
            ],
            'fusion': fusion_result,
            'explanation': {
                'p0': p0,
                'p1': p1,
                'p2': p2,
                'w0': weights[0],
                'w1': weights[1],
                'w2': weights[2],
                'gamma': args.gamma,
                'formula': 'p = w0*p0 + w1*p1 + w2*p2'
            }
        }
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Output saved to: {args.output_file}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Console display
        display = format_results_for_display(query_info, scale_results, fusion_result)
        print(display)


if __name__ == '__main__':
    main()

