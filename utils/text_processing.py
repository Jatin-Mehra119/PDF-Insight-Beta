"""
Utility functions for text processing and embeddings.

This module contains utility functions for text processing, tokenization,
chunking, and embedding operations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

from configs.config import Config


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text (rough approximation).
    
    Args:
        text: Input text
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // 4


def process_pdf_file(file_path: str) -> List[Document]:
    """
    Load a PDF file and extract its text with metadata.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects with metadata
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def chunk_text(documents: List[Document], max_length: int = None) -> List[Dict[str, Any]]:
    """
    Split documents into chunks with metadata.
    
    Args:
        documents: List of Document objects
        max_length: Maximum chunk length in tokens
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if max_length is None:
        max_length = Config.DEFAULT_CHUNK_SIZE
    
    chunks = []
    
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        paragraphs = text.split("\n\n")
        current_chunk = ""
        current_metadata = metadata.copy()
        
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph.strip()) < Config.MIN_PARAGRAPH_LENGTH:
                continue
                
            if estimate_tokens(current_chunk + paragraph) <= max_length // 4:
                current_chunk += paragraph + "\n\n"
            else:
                # Only add chunks with meaningful content
                if current_chunk.strip() and len(current_chunk.strip()) > Config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": current_chunk.strip(), 
                        "metadata": current_metadata
                    })
                current_chunk = paragraph + "\n\n"
        
        # Add the last chunk if it has meaningful content
        if current_chunk.strip() and len(current_chunk.strip()) > Config.MIN_CHUNK_LENGTH:
            chunks.append({
                "text": current_chunk.strip(), 
                "metadata": current_metadata
            })
    
    return chunks


def create_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Create embeddings for a list of chunk texts.
    
    Args:
        chunks: List of chunk dictionaries
        model: SentenceTransformer model
        
    Returns:
        Tuple of (embeddings array, chunks)
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu().numpy(), chunks


def filter_relevant_chunks(chunks_data: List[Tuple], threshold: float = None) -> List[Tuple]:
    """
    Filter chunks based on similarity threshold.
    
    Args:
        chunks_data: List of (text, score, metadata) tuples
        threshold: Similarity threshold (lower is more similar)
        
    Returns:
        Filtered list of chunks
    """
    if threshold is None:
        threshold = Config.SIMILARITY_THRESHOLD
    
    return [chunk for chunk in chunks_data if len(chunk) >= 3 and chunk[1] < threshold]


def prepare_context_from_chunks(context_chunks: List[Tuple], max_tokens: int = None) -> str:
    """
    Prepare context string from chunk data.
    
    Args:
        context_chunks: List of (text, score, metadata) tuples
        max_tokens: Maximum tokens for context
        
    Returns:
        Formatted context string
    """
    if max_tokens is None:
        max_tokens = Config.MAX_CONTEXT_TOKENS
    
    # Sort chunks by relevance (lower distance = more relevant)
    sorted_chunks = sorted(context_chunks, key=lambda x: x[1]) if context_chunks else []
    
    # Filter out chunks with very high distance scores (low similarity)
    relevant_chunks = filter_relevant_chunks(sorted_chunks)
    
    context = ""
    total_tokens = 0
    
    for chunk, _, _ in relevant_chunks:
        if chunk and chunk.strip():
            chunk_tokens = estimate_tokens(chunk)
            if total_tokens + chunk_tokens <= max_tokens:
                context += chunk + "\n\n"
                total_tokens += chunk_tokens
            else:
                break
    
    return context.strip() if context else "No initial context provided from preliminary search."


def validate_chunk_data(chunk_data: Any) -> bool:
    """
    Validate chunk data structure.
    
    Args:
        chunk_data: Chunk data to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(chunk_data, (list, tuple)):
        return False
    
    if len(chunk_data) < 3:
        return False
    
    text, score, metadata = chunk_data[0], chunk_data[1], chunk_data[2]
    
    if not isinstance(text, str) or not text.strip():
        return False
    
    if not isinstance(score, (int, float)):
        return False
    
    if not isinstance(metadata, dict):
        return False
    
    return True
