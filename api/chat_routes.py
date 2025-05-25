"""
Chat API routes.

This module handles chat and conversation endpoints.
"""

import traceback
from fastapi import HTTPException
from langchain.memory import ConversationBufferMemory

from configs.config import Config, ErrorMessages
from models.models import ChatRequest, ChatResponse
from services import session_manager, rag_service
from utils import retrieve_similar_chunks


async def chat_handler(request: ChatRequest) -> ChatResponse:
    """
    Handle chat requests with document context.
    
    Args:
        request: Chat request containing query and session info
        
    Returns:
        Chat response with answer and context
        
    Raises:
        HTTPException: If processing fails
    """
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail=ErrorMessages.EMPTY_QUERY)
    
    if len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail=ErrorMessages.QUERY_TOO_SHORT)
    
    # Get session data
    session_data, found = session_manager.get_session(request.session_id, request.model_name)
    if not found:
        raise HTTPException(status_code=404, detail=ErrorMessages.SESSION_EXPIRED)
    
    try:
        # Validate session data integrity
        is_valid, missing_keys = session_manager.validate_session(request.session_id)
        if not is_valid:
            raise HTTPException(status_code=500, detail=ErrorMessages.SESSION_INCOMPLETE)
        
        # Prepare agent memory with chat history
        agent_memory = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key="input", 
            return_messages=True
        )
        
        for entry in session_data.get("chat_history", []):
            agent_memory.chat_memory.add_user_message(entry["user"])
            agent_memory.chat_memory.add_ai_message(entry["assistant"])
        
        # Retrieve initial similar chunks for context
        initial_similar_chunks = retrieve_similar_chunks(
            request.query,
            session_data["index"],
            session_data["chunks"],
            session_data["model"],
            k=Config.INITIAL_CONTEXT_CHUNKS
        )
        
        # Generate response using RAG service
        response = rag_service.generate_response(
            llm=session_data["llm"],
            query=request.query,
            context_chunks=initial_similar_chunks,
            faiss_index=session_data["index"],
            document_chunks=session_data["chunks"],
            embedding_model=session_data["model"],
            memory=agent_memory,
            use_tavily=request.use_search
        )
        
        response_output = response.get("output", ErrorMessages.RESPONSE_GENERATION_ERROR)
        
        # Save chat history
        session_manager.add_chat_entry(
            request.session_id, 
            request.query, 
            response_output
        )
        
        return ChatResponse(
            status="success",
            answer=response_output,
            context_used=[
                {
                    "text": chunk,
                    "score": float(score),
                    "metadata": meta
                }
                for chunk, score, meta in initial_similar_chunks
            ]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=ErrorMessages.PROCESSING_ERROR.format(error=str(e))
        )
