"""
Session management service.

This module provides high-level session management operations.
"""

import uuid
from typing import Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

from configs.config import Config, ErrorMessages
from services.llm_service import create_llm_model
from utils import (
    save_session_to_file,
    load_session_from_file,
    reconstruct_session_objects,
    cleanup_session_files,
    validate_session_data,
    session_exists,
    create_embeddings,
    build_faiss_index
)


class SessionManager:
    """Manager for session operations."""
    
    def __init__(self):
        """Initialize session manager."""
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(
        self,
        file_path: str,
        file_name: str,
        chunks_with_metadata: list,
        model_name: str
    ) -> str:
        """
        Create a new session with processed document data.
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original filename
            chunks_with_metadata: Processed document chunks
            model_name: LLM model name
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        # Create embedding model and process chunks
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        embeddings, _ = create_embeddings(chunks_with_metadata, embedding_model)
        
        # Build FAISS index
        index = build_faiss_index(embeddings)
        
        # Create LLM
        llm = create_llm_model(model_name)
        
        # Create session data
        session_data = {
            "file_path": file_path,
            "file_name": file_name,
            "chunks": chunks_with_metadata,
            "model": embedding_model,
            "index": index,
            "llm": llm,
            "chat_history": []
        }
        
        # Save to memory and file
        self.active_sessions[session_id] = session_data
        save_session_to_file(session_id, session_data)
        
        return session_id
    
    def get_session(
        self, 
        session_id: str, 
        model_name: str = None
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Retrieve session data, loading from file if necessary.
        
        Args:
            session_id: Session identifier
            model_name: LLM model name (for reconstruction)
            
        Returns:
            Tuple of (session_data, found)
        """
        if model_name is None:
            model_name = Config.DEFAULT_MODEL
        
        try:
            # Check if session is in memory
            if session_id in self.active_sessions:
                cached_session = self.active_sessions[session_id]
                
                # Ensure LLM is up-to-date
                if (cached_session.get("llm") is None or 
                    (hasattr(cached_session["llm"], "model_name") and 
                     cached_session["llm"].model_name != model_name)):
                    cached_session["llm"] = create_llm_model(model_name)
                
                # Ensure embedding model exists
                if cached_session.get("model") is None:
                    cached_session["model"] = SentenceTransformer(Config.EMBEDDING_MODEL)
                
                # Ensure FAISS index exists
                if cached_session.get("index") is None and cached_session.get("chunks"):
                    embeddings, _ = create_embeddings(
                        cached_session["chunks"], 
                        cached_session["model"]
                    )
                    cached_session["index"] = build_faiss_index(embeddings)
                
                return cached_session, True
            
            # Try to load from file
            data, success = load_session_from_file(session_id)
            if not success:
                return None, False
            
            # Check if original PDF exists
            original_pdf_path = data.get("file_path")
            if not (data.get("chunks") and original_pdf_path and 
                    session_exists(session_id)):
                print(f"Warning: Session data for {session_id} is incomplete or PDF missing.")
                cleanup_session_files(session_id)
                return None, False
            
            # Reconstruct session objects
            embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            full_session_data = reconstruct_session_objects(
                data, model_name, embedding_model
            )
            
            # Cache in memory
            self.active_sessions[session_id] = full_session_data
            
            return full_session_data, True
            
        except Exception as e:
            print(f"Error loading session {session_id}: {str(e)}")
            return None, False
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save session data to memory and file.
        
        Args:
            session_id: Session identifier
            session_data: Session data to save
            
        Returns:
            True if successful, False otherwise
        """
        # Update memory cache
        self.active_sessions[session_id] = session_data
        
        # Save to file
        return save_session_to_file(session_id, session_data)
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove session and associated files.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from memory
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Clean up files
            return cleanup_session_files(session_id)
            
        except Exception as e:
            print(f"Error removing session {session_id}: {str(e)}")
            return False
    
    def clear_chat_history(self, session_id: str) -> bool:
        """
        Clear chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        session_data, found = self.get_session(session_id)
        if not found:
            return False
        
        session_data["chat_history"] = []
        return self.save_session(session_id, session_data)
    
    def add_chat_entry(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_message: str
    ) -> bool:
        """
        Add a chat entry to session history.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            True if successful, False otherwise
        """
        session_data, found = self.get_session(session_id)
        if not found:
            return False
        
        session_data["chat_history"].append({
            "user": user_message,
            "assistant": assistant_message
        })
        
        return self.save_session(session_id, session_data)
    
    def validate_session(self, session_id: str) -> Tuple[bool, list]:
        """
        Validate session data integrity.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (is_valid, missing_keys)
        """
        session_data, found = self.get_session(session_id)
        if not found:
            return False, ["session_not_found"]
        
        return validate_session_data(session_data)


# Global session manager instance
session_manager = SessionManager()
