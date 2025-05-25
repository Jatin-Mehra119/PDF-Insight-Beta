"""
Pydantic models and data structures for PDF Insight Beta application.

This module defines all the data models used throughout the application.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="User query")
    use_search: bool = Field(default=False, description="Whether to use web search")
    model_name: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="LLM model to use"
    )


class SessionRequest(BaseModel):
    """Request model for session-related endpoints."""
    session_id: str = Field(..., description="Session identifier")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    status: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    status: str
    answer: str
    context_used: List[Dict[str, Any]]


class ChatHistoryResponse(BaseModel):
    """Response model for chat history endpoint."""
    status: str
    history: List[Dict[str, str]]


class StatusResponse(BaseModel):
    """Generic status response model."""
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str
    detail: str
    type: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str


class ModelsResponse(BaseModel):
    """Response model for models endpoint."""
    models: List[ModelInfo]


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    source: Optional[str] = None
    page: Optional[int] = None
    
    class Config:
        extra = "allow"  # Allow additional metadata fields


class DocumentChunk(BaseModel):
    """Document chunk with text and metadata."""
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format used in processing."""
        return {
            "text": self.text,
            "metadata": self.metadata.dict()
        }


class SessionData(BaseModel):
    """Session data structure."""
    file_path: str
    file_name: str
    chunks: List[Dict[str, Any]]  # List of chunk dictionaries
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like FAISS index


class ChatHistoryEntry(BaseModel):
    """Single chat history entry."""
    user: str
    assistant: str


class ContextChunk(BaseModel):
    """Context chunk with similarity score."""
    text: str
    score: float
    metadata: Dict[str, Any]
