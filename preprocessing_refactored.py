"""
Refactored preprocessing module for PDF Insight Beta.

This module provides the core preprocessing functionality with improved organization.
The original logic has been preserved while breaking it into more maintainable components.

This module maintains backward compatibility with the original preprocessing.py interface.
"""

# Re-export everything from the new modular structure for backward compatibility
from configs.config import Config
from services import (
    create_llm_model as model_selection,
    create_tavily_search_tool,
    rag_service
)
from utils import (
    process_pdf_file,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_similar_chunks,
    estimate_tokens
)

# Create global tools for backward compatibility
def create_global_tools():
    """Create global tools list for backward compatibility."""
    tavily_tool = create_tavily_search_tool()
    return [tavily_tool] if tavily_tool else []

# Global tools instance (for backward compatibility)
tools = create_global_tools()

# Alias for the main RAG function to maintain original interface
def agentic_rag(llm, agent_specific_tools, query, context_chunks, memory, Use_Tavily=False):
    """
    Main RAG function with original interface for backward compatibility.
    
    Args:
        llm: Language model instance
        agent_specific_tools: List of tools for the agent
        query: User query
        context_chunks: Context chunks from retrieval
        memory: Conversation memory
        Use_Tavily: Whether to use web search
        
    Returns:
        Dictionary with 'output' key containing the response
    """
    # Convert parameters to work with new RAG service
    return rag_service.generate_response(
        llm=llm,
        query=query,
        context_chunks=context_chunks,
        faiss_index=None,  # Will be handled internally by tools
        document_chunks=[],  # Will be handled internally by tools
        embedding_model=None,  # Will be handled internally by tools
        memory=memory,
        use_tavily=Use_Tavily
    )

# Re-export the vector search tool creator for backward compatibility
from services.rag_service import create_vector_search_tool

# Maintain all original exports
__all__ = [
    'model_selection',
    'process_pdf_file', 
    'chunk_text',
    'create_embeddings',
    'build_faiss_index',
    'retrieve_similar_chunks',
    'agentic_rag',
    'tools',
    'create_vector_search_tool',
    'estimate_tokens'
]
