"""
FAISS indexing utilities for similarity search.

This module provides utilities for building and searching FAISS indexes.
"""

from typing import List, Tuple, Any, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from configs.config import Config
from utils.text_processing import validate_chunk_data


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Build a FAISS HNSW index from embeddings for similarity search.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS HNSW index
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, Config.FAISS_NEIGHBORS)
    index.hnsw.efConstruction = Config.FAISS_EF_CONSTRUCTION
    index.hnsw.efSearch = Config.FAISS_EF_SEARCH
    index.add(embeddings)
    return index


def retrieve_similar_chunks(
    query: str,
    index: faiss.IndexHNSWFlat,
    chunks_with_metadata: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    k: int = None,
    max_chunk_length: int = None
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Retrieve top k similar chunks to the query from the FAISS index.
    
    Args:
        query: Search query
        index: FAISS index
        chunks_with_metadata: List of chunk dictionaries
        embedding_model: SentenceTransformer model
        k: Number of chunks to retrieve
        max_chunk_length: Maximum length for returned chunks
        
    Returns:
        List of tuples (chunk_text, distance, metadata)
    """
    if k is None:
        k = Config.DEFAULT_K_CHUNKS
    if max_chunk_length is None:
        max_chunk_length = Config.DEFAULT_CHUNK_SIZE
    
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    
    # Ensure indices are within bounds and create mapping for correct distances
    valid_results = []
    for idx_pos, chunk_idx in enumerate(indices[0]):
        if 0 <= chunk_idx < len(chunks_with_metadata):
            chunk_text = chunks_with_metadata[chunk_idx]["text"][:max_chunk_length]
            # Only include chunks with meaningful content
            if chunk_text.strip():  # Skip empty chunks
                result = (
                    chunk_text,
                    distances[0][idx_pos],  # Use original position for correct distance
                    chunks_with_metadata[chunk_idx]["metadata"]
                )
                if validate_chunk_data(result):
                    valid_results.append(result)
    
    return valid_results


def search_index_with_validation(
    query: str,
    index: faiss.IndexHNSWFlat,
    chunks_with_metadata: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    k: int = None,
    similarity_threshold: float = None
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Search index with additional validation and filtering.
    
    Args:
        query: Search query
        index: FAISS index
        chunks_with_metadata: List of chunk dictionaries
        embedding_model: SentenceTransformer model
        k: Number of chunks to retrieve
        similarity_threshold: Threshold for filtering results
        
    Returns:
        List of validated and filtered chunk tuples
    """
    if not query or len(query.strip()) < 3:
        return []
    
    if similarity_threshold is None:
        similarity_threshold = Config.SIMILARITY_THRESHOLD
    
    try:
        # Retrieve similar chunks
        similar_chunks = retrieve_similar_chunks(
            query, index, chunks_with_metadata, embedding_model, k
        )
        
        # Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in similar_chunks 
            if chunk[1] < similarity_threshold
        ]
        
        return filtered_chunks
        
    except Exception as e:
        print(f"Error in index search: {e}")
        return []


def get_index_stats(index: faiss.IndexHNSWFlat) -> Dict[str, Any]:
    """
    Get statistics about the FAISS index.
    
    Args:
        index: FAISS index
        
    Returns:
        Dictionary with index statistics
    """
    return {
        "total_vectors": index.ntotal,
        "dimension": index.d,
        "index_type": type(index).__name__,
        "ef_search": index.hnsw.efSearch,
        "ef_construction": index.hnsw.efConstruction,
        "is_trained": index.is_trained
    }
