"""
Configuration module for PDF Insight Beta application.

This module centralizes all configuration settings, constants, and environment variables.
"""

import os
from typing import List, Dict, Any
import dotenv

# Load environment variables
dotenv.load_dotenv()


class Config:
    """Application configuration class."""
    
    # API Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Application Settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Model Configuration
    DEFAULT_MODEL: str = "llama3-8b-8192"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    
    # Text Processing Settings
    DEFAULT_CHUNK_SIZE: int = 1000
    MIN_CHUNK_LENGTH: int = 20
    MIN_PARAGRAPH_LENGTH: int = 10
    
    # RAG Configuration
    DEFAULT_K_CHUNKS: int = 10
    INITIAL_CONTEXT_CHUNKS: int = 5
    MAX_CONTEXT_TOKENS: int = 7000
    SIMILARITY_THRESHOLD: float = 1.5
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4500
    
    # FAISS Index Configuration
    FAISS_NEIGHBORS: int = 32
    FAISS_EF_CONSTRUCTION: int = 200
    FAISS_EF_SEARCH: int = 50
    
    # Agent Configuration
    AGENT_MAX_ITERATIONS: int = 2
    AGENT_VERBOSE: bool = False
    
    # Tavily Search Configuration
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "advanced"
    TAVILY_INCLUDE_ANSWER: bool = True
    TAVILY_INCLUDE_RAW_CONTENT: bool = False
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]


class ModelConfig:
    """Model configuration and metadata."""
    
    AVAILABLE_MODELS: List[Dict[str, str]] = [
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B"},
        {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant"},
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70b Versatile"},
    ]
    
    @classmethod
    def get_model_ids(cls) -> List[str]:
        """Get list of available model IDs."""
        return [model["id"] for model in cls.AVAILABLE_MODELS]
    
    @classmethod
    def is_valid_model(cls, model_id: str) -> bool:
        """Check if a model ID is valid."""
        return model_id in cls.get_model_ids()


class ErrorMessages:
    """Centralized error messages."""
    
    # Validation Errors
    EMPTY_QUERY = "Query cannot be empty"
    QUERY_TOO_SHORT = "Query must be at least 3 characters long"
    
    # Session Errors
    SESSION_NOT_FOUND = "Session not found"
    SESSION_EXPIRED = "Session not found or expired. Please upload a document first."
    SESSION_INCOMPLETE = "Session data is incomplete. Please upload the document again."
    SESSION_REMOVAL_FAILED = "Session not found or could not be removed"
    
    # File Errors
    FILE_NOT_FOUND = "The file {file_path} does not exist."
    PDF_PROCESSING_ERROR = "Error processing PDF: {error}"
    
    # API Key Errors
    GROQ_API_KEY_MISSING = "GROQ_API_KEY is not set for Groq Llama models."
    TAVILY_API_KEY_MISSING = "TAVILY_API_KEY is not set. Web search will not function."
    
    # Processing Errors
    PROCESSING_ERROR = "Error processing query: {error}"
    RESPONSE_GENERATION_ERROR = "Sorry, I could not generate a response."


class SuccessMessages:
    """Centralized success messages."""
    
    PDF_PROCESSED = "Processed {filename}"
    PDF_REMOVED = "PDF file and session removed successfully"
    CHAT_HISTORY_CLEARED = "Chat history cleared"


# Initialize directories
def initialize_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists(Config.UPLOAD_DIR):
        os.makedirs(Config.UPLOAD_DIR)


# Initialize on import
initialize_directories()
