"""
Qdrant vector database utilities.

Provides functions for:
- Creating and validating collections
- Batch upserting points with embeddings
- Searching similar vectors
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import qdrant_client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Install with: pip install qdrant-client")


def check_qdrant_available():
    """Check if qdrant-client is available."""
    if not QDRANT_AVAILABLE:
        raise ImportError(
            "qdrant-client is required for vector database operations. "
            "Install with: pip install qdrant-client"
        )


def get_client(url: str = "http://localhost:6333", timeout: int = 60) -> "QdrantClient":
    """
    Get Qdrant client.
    
    Args:
        url: Qdrant server URL
        timeout: Connection timeout in seconds
        
    Returns:
        QdrantClient instance
    """
    check_qdrant_available()
    
    logger.info(f"Connecting to Qdrant at: {url}")
    client = QdrantClient(url=url, timeout=timeout)
    
    # Test connection
    try:
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise
    
    return client


def create_or_validate_collection(
    client: "QdrantClient",
    name: str,
    vector_size: int,
    distance: str = "Cosine",
    recreate: bool = False
) -> bool:
    """
    Create a collection or validate existing one.
    
    Args:
        client: QdrantClient instance
        name: Collection name
        vector_size: Dimension of vectors
        distance: Distance metric ("Cosine", "Euclid", "Dot")
        recreate: If True, delete existing collection and create new one
        
    Returns:
        True if collection was created, False if it already existed
    """
    check_qdrant_available()
    
    # Map distance string to enum
    distance_map = {
        "Cosine": Distance.COSINE,
        "cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
        "dot": Distance.DOT,
    }
    dist = distance_map.get(distance, Distance.COSINE)
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == name for c in collections)
    
    if exists:
        if recreate:
            logger.info(f"Deleting existing collection: {name}")
            client.delete_collection(name)
        else:
            # Validate existing collection
            info = client.get_collection(name)
            existing_size = info.config.params.vectors.size
            
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{name}' exists with different vector size: "
                    f"{existing_size} vs {vector_size}"
                )
            
            logger.info(f"Collection '{name}' already exists with {info.points_count} points")
            return False
    
    # Create collection
    logger.info(f"Creating collection: {name} (vector_size={vector_size}, distance={distance})")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=dist
        )
    )
    
    return True


def upsert_points(
    client: "QdrantClient",
    collection_name: str,
    ids: List[int],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    batch_size: int = 256
) -> int:
    """
    Batch upsert points to a collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Collection name
        ids: List of point IDs
        vectors: List of embedding vectors
        payloads: List of payload dictionaries
        batch_size: Batch size for upsert operations
        
    Returns:
        Total number of points upserted
    """
    check_qdrant_available()
    
    assert len(ids) == len(vectors) == len(payloads), \
        f"Length mismatch: ids={len(ids)}, vectors={len(vectors)}, payloads={len(payloads)}"
    
    total = len(ids)
    n_batches = (total + batch_size - 1) // batch_size
    
    logger.info(f"Upserting {total} points to '{collection_name}' in {n_batches} batches")
    
    upserted = 0
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        
        batch_points = [
            PointStruct(
                id=ids[j],
                vector=vectors[j],
                payload=payloads[j]
            )
            for j in range(start, end)
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        
        upserted += len(batch_points)
        
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            logger.info(f"  Upserted {upserted}/{total} points")
    
    return upserted


def search_similar(
    client: "QdrantClient",
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10,
    filter_conditions: Optional[Dict[str, Any]] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors.
    
    Args:
        client: QdrantClient instance
        collection_name: Collection name
        query_vector: Query embedding vector
        top_k: Number of results to return
        filter_conditions: Optional filter conditions (e.g., {"label": 1})
        with_payload: Whether to return payload
        with_vectors: Whether to return vectors
        
    Returns:
        List of search results with id, score, and payload
    """
    check_qdrant_available()
    
    # Build filter if provided
    query_filter = None
    if filter_conditions:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        query_filter = Filter(must=conditions)
    
    # Search
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=with_payload,
        with_vectors=with_vectors
    )
    
    # Format results
    formatted = []
    for i, hit in enumerate(results):
        result = {
            'rank': i + 1,
            'id': hit.id,
            'score': hit.score,
        }
        if with_payload and hit.payload:
            result['payload'] = hit.payload
        if with_vectors and hit.vector:
            result['vector'] = hit.vector
        formatted.append(result)
    
    return formatted


def get_collection_info(client: "QdrantClient", collection_name: str) -> Dict[str, Any]:
    """
    Get collection information.
    
    Args:
        client: QdrantClient instance
        collection_name: Collection name
        
    Returns:
        Dictionary with collection info
    """
    check_qdrant_available()
    
    info = client.get_collection(collection_name)
    
    result = {
        'name': collection_name,
        'points_count': info.points_count,
        'status': str(info.status),
    }
    
    # Handle different qdrant-client versions
    # vectors_count may not exist in all versions
    if hasattr(info, 'vectors_count'):
        result['vectors_count'] = info.vectors_count
    else:
        result['vectors_count'] = info.points_count  # Fallback
    
    # Handle vector config (may be dict or object depending on version)
    try:
        if hasattr(info.config.params, 'vectors'):
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                # Named vectors
                result['vector_size'] = 'multiple'
                result['distance'] = 'multiple'
            else:
                result['vector_size'] = vectors_config.size
                result['distance'] = str(vectors_config.distance)
        else:
            result['vector_size'] = 'unknown'
            result['distance'] = 'unknown'
    except Exception:
        result['vector_size'] = 'unknown'
        result['distance'] = 'unknown'
    
    return result


def delete_collection(client: "QdrantClient", collection_name: str) -> bool:
    """
    Delete a collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Collection name
        
    Returns:
        True if deleted, False if didn't exist
    """
    check_qdrant_available()
    
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return True
    else:
        logger.info(f"Collection '{collection_name}' does not exist")
        return False


def list_collections(client: "QdrantClient") -> List[str]:
    """
    List all collections.
    
    Args:
        client: QdrantClient instance
        
    Returns:
        List of collection names
    """
    check_qdrant_available()
    
    collections = client.get_collections().collections
    return [c.name for c in collections]

