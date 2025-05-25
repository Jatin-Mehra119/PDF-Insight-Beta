"""
Utility modules initialization.

This module provides easy imports for all utility functions.
"""

from .text_processing import (
    estimate_tokens,
    process_pdf_file,
    chunk_text,
    create_embeddings,
    filter_relevant_chunks,
    prepare_context_from_chunks,
    validate_chunk_data
)

from .faiss_utils import (
    build_faiss_index,
    retrieve_similar_chunks,
    search_index_with_validation,
    get_index_stats
)

from .session_utils import (
    create_session_file_path,
    create_upload_file_path,
    prepare_pickle_safe_data,
    save_session_to_file,
    load_session_from_file,
    reconstruct_session_objects,
    cleanup_session_files,
    validate_session_data,
    session_exists
)

__all__ = [
    # Text processing
    "estimate_tokens",
    "process_pdf_file", 
    "chunk_text",
    "create_embeddings",
    "filter_relevant_chunks",
    "prepare_context_from_chunks",
    "validate_chunk_data",
    
    # FAISS utilities
    "build_faiss_index",
    "retrieve_similar_chunks", 
    "search_index_with_validation",
    "get_index_stats",
    
    # Session utilities
    "create_session_file_path",
    "create_upload_file_path",
    "prepare_pickle_safe_data",
    "save_session_to_file",
    "load_session_from_file",
    "reconstruct_session_objects",
    "cleanup_session_files",
    "validate_session_data",
    "session_exists"
]
