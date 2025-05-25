"""
Session management utilities.

This module provides utilities for session data persistence and management.
"""

import os
import pickle
import traceback
from typing import Dict, Any, Tuple, Optional, List

from configs.config import Config, ErrorMessages


def create_session_file_path(session_id: str) -> str:
    """
    Create the file path for a session pickle file.
    
    Args:
        session_id: Session identifier
        
    Returns:
        File path for the session data
    """
    return f"{Config.UPLOAD_DIR}/{session_id}_session.pkl"


def create_upload_file_path(session_id: str, filename: str) -> str:
    """
    Create the file path for an uploaded file.
    
    Args:
        session_id: Session identifier
        filename: Original filename
        
    Returns:
        File path for the uploaded file
    """
    return f"{Config.UPLOAD_DIR}/{session_id}_{filename}"


def prepare_pickle_safe_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare session data for pickling by removing non-serializable objects.
    
    Args:
        session_data: Full session data
        
    Returns:
        Pickle-safe session data
    """
    return {
        "file_path": session_data.get("file_path"),
        "file_name": session_data.get("file_name"),
        "chunks": session_data.get("chunks"),  # Chunks with metadata (list of dicts)
        "chat_history": session_data.get("chat_history", [])
        # FAISS index, embedding model, and LLM model are not pickled
    }


def save_session_to_file(session_id: str, session_data: Dict[str, Any]) -> bool:
    """
    Save session data to pickle file.
    
    Args:
        session_id: Session identifier
        session_data: Session data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pickle_safe_data = prepare_pickle_safe_data(session_data)
        file_path = create_session_file_path(session_id)
        
        with open(file_path, "wb") as f:
            pickle.dump(pickle_safe_data, f)
        
        return True
    except Exception as e:
        print(f"Error saving session {session_id}: {str(e)}")
        return False


def load_session_from_file(session_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Load session data from pickle file.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Tuple of (session_data, success)
    """
    try:
        file_path = create_session_file_path(session_id)
        
        if not os.path.exists(file_path):
            return None, False
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        return data, True
    except Exception as e:
        print(f"Error loading session {session_id}: {str(e)}")
        return None, False


def reconstruct_session_objects(
    session_data: Dict[str, Any], 
    model_name: str,
    embedding_model
) -> Dict[str, Any]:
    """
    Reconstruct non-serializable objects in session data.
    
    Args:
        session_data: Basic session data from pickle
        model_name: LLM model name
        embedding_model: SentenceTransformer instance
        
    Returns:
        Complete session data with reconstructed objects
    """
    # Import here to avoid circular imports
    from sentence_transformers import SentenceTransformer
    from langchain_groq import ChatGroq
    
    # Create LLM model
    llm = ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.MAX_TOKENS
    )
    
    # Reconstruct embeddings and FAISS index
    if session_data.get("chunks"):
        # Import here to avoid circular imports
        from utils.text_processing import create_embeddings
        from utils.faiss_utils import build_faiss_index
        
        embeddings, _ = create_embeddings(session_data["chunks"], embedding_model)
        faiss_index = build_faiss_index(embeddings)
    else:
        embeddings, faiss_index = None, None
    
    return {
        **session_data,
        "model": embedding_model,
        "index": faiss_index,
        "llm": llm
    }


def cleanup_session_files(session_id: str) -> bool:
    """
    Clean up all files associated with a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        session_file = create_session_file_path(session_id)
        
        # Load session data to get file path
        if os.path.exists(session_file):
            try:
                with open(session_file, "rb") as f:
                    data = pickle.load(f)
                
                # Delete PDF file if it exists
                pdf_path = data.get("file_path")
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                print(f"Error reading session file for cleanup: {e}")
            
            # Remove session file
            os.remove(session_file)
        
        return True
    except Exception as e:
        print(f"Error cleaning up session {session_id}: {str(e)}")
        return False


def validate_session_data(session_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate session data integrity.
    
    Args:
        session_data: Session data to validate
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    required_keys = ["index", "chunks", "model", "llm"]
    missing_keys = [key for key in required_keys if key not in session_data]
    
    return len(missing_keys) == 0, missing_keys


def session_exists(session_id: str) -> bool:
    """
    Check if a session exists.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if session exists, False otherwise
    """
    session_file = create_session_file_path(session_id)
    return os.path.exists(session_file)
