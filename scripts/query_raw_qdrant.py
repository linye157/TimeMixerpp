#!/usr/bin/env python
"""
查询原始数据 Qdrant 知识库。

基于 48 维温度向量检索相似样本，支持相似度加权投票预测。

Usage:
    # 从文件中指定索引查询
    python scripts/query_raw_qdrant.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --query_index 123 --top_k 10

    # 直接输入 48 维向量查询（逗号分隔）
    python scripts/query_raw_qdrant.py --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --query_vector "1.2,3.4,5.6,..." --top_k 10

    # 仅检索模式（默认）
    python scripts/query_raw_qdrant.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --query_index 123 --top_k 10

    # 检索 + 预测模式
    python scripts/query_raw_qdrant.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --query_index 123 --top_k 10 --retrieve_only false --gamma 10

    # JSON 输出
    python scripts/query_raw_qdrant.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --query_index 123 --json_output true
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

from timemixerpp.data import load_file_strict
from timemixerpp.qdrant_utils import get_client, search_similar
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Query raw data Qdrant knowledge base',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Query source (choose one)
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (use with --query_index)')
    parser.add_argument('--query_index', type=int, default=None,
                        help='Sample index in data file to query')
    parser.add_argument('--query_vector', type=str, default=None,
                        help='Direct 48-dim vector (comma-separated)')
    
    # Qdrant arguments
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection_name', type=str, default='raw_temperature_kb',
                        help='Collection name')
    
    # Query arguments
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of similar samples to retrieve')
    parser.add_argument('--retrieve_only', type=str, default='true',
                        help='Only retrieve without prediction (true/false)')
    parser.add_argument('--exclude_self', type=str, default='true',
                        help='Exclude query sample from results (true/false)')
    parser.add_argument('--min_results', type=int, default=10,
                        help='Extra results to request for filtering')
    
    # Prediction arguments (only when retrieve_only=false)
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='Similarity weighting coefficient')
    
    # Normalization (should match ingestion)
    parser.add_argument('--normalize', action='store_true',
                        help='Apply z-score normalization to query')
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Apply L2 normalization to query')
    parser.add_argument('--norm_stats_from_data', action='store_true',
                        help='Compute normalization stats from data_path')
    
    # Output arguments
    parser.add_argument('--json_output', type=str, default='false',
                        help='Output JSON format (true/false)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for JSON')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def compute_probability(results: List[Dict], gamma: float) -> Tuple[float, List[Dict]]:
    """
    Compute probability using similarity-weighted voting.
    
    p = Σ w_i * label_i / Σ w_i
    where w_i = exp(gamma * score_i)
    """
    if not results:
        return 0.5, []
    
    weighted_results = []
    total_weight = 0.0
    weighted_sum = 0.0
    
    for r in results:
        score = r.get('score', 0)
        weight = np.exp(gamma * score)
        label = r.get('payload', {}).get('label', 0)
        label_raw = r.get('payload', {}).get('label_raw', label)
        
        total_weight += weight
        weighted_sum += weight * label_raw
        
        weighted_results.append({
            **r,
            'weight': weight,
        })
    
    # Normalize weights
    for r in weighted_results:
        r['weight_normalized'] = r['weight'] / total_weight if total_weight > 0 else 0
    
    prob = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    return prob, weighted_results


def filter_results(
    results: List[Dict],
    query_index: Optional[int],
    exclude_self: bool,
    top_k: int
) -> List[Dict]:
    """Filter results: exclude self and truncate to top_k."""
    if not exclude_self or query_index is None:
        return results[:top_k]
    
    filtered = []
    for r in results:
        sample_id = r.get('payload', {}).get('sample_id', r.get('id'))
        if sample_id != query_index:
            filtered.append(r)
        if len(filtered) >= top_k:
            break
    
    if len(filtered) < top_k:
        logger.warning(f"Only {len(filtered)} results after filtering (requested {top_k})")
    
    return filtered


def format_retrieve_display(
    query_info: Dict[str, Any],
    results: List[Dict],
    top_k: int
) -> str:
    """Format retrieve-only results for console display."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(" 原始数据 RAG 检索结果")
    lines.append("=" * 70)
    
    # Query info
    lines.append(f"\n查询样本:")
    if 'index' in query_info:
        lines.append(f"  Index: {query_info['index']}")
    if 'label' in query_info:
        lines.append(f"  真实标签: {query_info['label']:.4f}")
    
    # Results
    lines.append(f"\n{'-' * 70}")
    lines.append(f" 相似样本 (Top {min(len(results), top_k)})")
    lines.append(f"{'-' * 70}")
    lines.append(f"{'Rank':<6}{'ID':<10}{'Label':<8}{'Label_raw':<12}{'Score':<12}")
    lines.append("-" * 48)
    
    for i, r in enumerate(results[:top_k]):
        payload = r.get('payload', {})
        label = payload.get('label', 'N/A')
        label_raw = payload.get('label_raw', label)
        if isinstance(label_raw, (int, float)):
            label_raw_str = f"{label_raw:.1f}"
        else:
            label_raw_str = str(label_raw)
        
        lines.append(
            f"{i+1:<6}"
            f"{r['id']:<10}"
            f"{label:<8}"
            f"{label_raw_str:<12}"
            f"{r['score']:<12.4f}"
        )
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def format_full_display(
    query_info: Dict[str, Any],
    results: List[Dict],
    prob: float
) -> str:
    """Format full results (with prediction) for console display."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(" 原始数据 RAG 查询结果")
    lines.append("=" * 70)
    
    # Query info
    lines.append(f"\n查询样本:")
    if 'index' in query_info:
        lines.append(f"  Index: {query_info['index']}")
    if 'label' in query_info:
        label = query_info['label']
        lines.append(f"  真实标签: {label:.4f} ({'正类' if label >= 0.5 else '负类'})")
    
    # Results with weights
    lines.append(f"\n{'-' * 70}")
    lines.append(f" 相似样本及权重")
    lines.append(f"{'-' * 70}")
    lines.append(f"{'Rank':<6}{'ID':<10}{'Label':<8}{'Score':<12}{'Weight':<10}")
    lines.append("-" * 46)
    
    for i, r in enumerate(results[:5]):  # Show top 5
        payload = r.get('payload', {})
        label = payload.get('label', 'N/A')
        weight_norm = r.get('weight_normalized', 0)
        lines.append(
            f"{i+1:<6}"
            f"{r['id']:<10}"
            f"{label:<8}"
            f"{r['score']:<12.4f}"
            f"{weight_norm:<10.4f}"
        )
    
    if len(results) > 5:
        lines.append(f"  ... 还有 {len(results) - 5} 个结果")
    
    # Prediction
    lines.append(f"\n{'=' * 70}")
    lines.append(" 预测结果")
    lines.append("=" * 70)
    lines.append(f"  融合概率: {prob:.4f} (展示: {prob:.1f})")
    lines.append(f"  预测: {'正类 (事故风险高)' if prob >= 0.5 else '负类 (事故风险低)'}")
    
    # Comparison with true label
    if 'label' in query_info:
        pred = 1 if prob >= 0.5 else 0
        true = 1 if query_info['label'] >= 0.5 else 0
        correct = "✓ 正确" if pred == true else "✗ 错误"
        lines.append(f"  与真实标签对比: {correct}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    setup_logging()
    
    json_output = str_to_bool(args.json_output)
    retrieve_only = str_to_bool(args.retrieve_only)
    exclude_self = str_to_bool(args.exclude_self)
    
    # Get query vector
    query_vector = None
    query_info = {}
    norm_mean = None
    norm_std = None
    
    if args.query_vector is not None:
        # Direct vector input
        query_vector = np.array([float(x) for x in args.query_vector.split(',')], dtype=np.float32)
        if len(query_vector) != 48:
            raise ValueError(f"Query vector must be 48-dimensional, got {len(query_vector)}")
        query_info = {'source': 'direct_input'}
        
    elif args.data_path is not None and args.query_index is not None:
        # Load from file
        logger.info(f"Loading data from: {args.data_path}")
        _, X, y = load_file_strict(args.data_path)
        
        if args.query_index >= len(X):
            raise ValueError(f"Query index {args.query_index} out of range (max {len(X)-1})")
        
        query_vector = X[args.query_index].astype(np.float32)
        query_info = {
            'index': args.query_index,
            'label': float(y[args.query_index]),
            'source': args.data_path
        }
        
        # Compute normalization stats from data if requested
        if args.norm_stats_from_data and args.normalize:
            norm_mean = X.mean(axis=0, keepdims=True)
            norm_std = X.std(axis=0, keepdims=True)
            norm_std[norm_std < 1e-6] = 1.0
    else:
        raise ValueError("Must provide either --query_vector or both --data_path and --query_index")
    
    # Apply normalization to query
    if args.normalize:
        if norm_mean is not None:
            query_vector = (query_vector - norm_mean.flatten()) / norm_std.flatten()
        else:
            # Simple per-sample normalization
            mean = query_vector.mean()
            std = query_vector.std()
            if std > 1e-6:
                query_vector = (query_vector - mean) / std
    
    if args.l2_normalize:
        norm = np.linalg.norm(query_vector)
        if norm > 1e-8:
            query_vector = query_vector / norm
    
    # Connect to Qdrant
    client = get_client(args.qdrant_url)
    
    # Calculate actual limit to request
    request_limit = args.top_k + args.min_results if exclude_self else args.top_k
    
    # Search
    logger.info(f"Searching in collection: {args.collection_name}")
    results = search_similar(
        client,
        args.collection_name,
        query_vector.tolist(),
        top_k=request_limit,
        with_payload=True
    )
    
    # Filter results
    query_index = query_info.get('index')
    filtered_results = filter_results(results, query_index, exclude_self, args.top_k)
    
    # Output based on mode
    if retrieve_only:
        if json_output:
            output = {
                'query': query_info,
                'results': [
                    {
                        'rank': i + 1,
                        'id': r['id'],
                        'label': r.get('payload', {}).get('label'),
                        'label_raw': round(r.get('payload', {}).get('label_raw', 0), 1),
                        'score': round(r['score'], 4),
                        'sample_id': r.get('payload', {}).get('sample_id', r['id'])
                    }
                    for i, r in enumerate(filtered_results)
                ]
            }
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {args.output_file}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            display = format_retrieve_display(query_info, filtered_results, args.top_k)
            print(display)
    
    else:
        # Compute probability
        prob, weighted_results = compute_probability(filtered_results, args.gamma)
        
        if json_output:
            output = {
                'query': query_info,
                'results': [
                    {
                        'rank': i + 1,
                        'id': r['id'],
                        'label': r.get('payload', {}).get('label'),
                        'label_raw': r.get('payload', {}).get('label_raw'),
                        'score': round(r['score'], 4),
                        'weight': round(r.get('weight_normalized', 0), 4),
                        'sample_id': r.get('payload', {}).get('sample_id', r['id'])
                    }
                    for i, r in enumerate(weighted_results)
                ],
                'prediction': {
                    'probability': round(prob, 4),
                    'probability_1d': round(prob, 1),
                    'predicted_class': 1 if prob >= 0.5 else 0,
                    'gamma': args.gamma
                }
            }
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {args.output_file}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            display = format_full_display(query_info, weighted_results, prob)
            print(display)


if __name__ == '__main__':
    main()

