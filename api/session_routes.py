"""
Session management API routes.

This module handles session-related endpoints like history and cleanup.
"""

from fastapi import HTTPException

from configs.config import ErrorMessages, SuccessMessages
from models.models import SessionRequest, ChatHistoryResponse, StatusResponse
from services import session_manager


async def get_chat_history_handler(request: SessionRequest) -> ChatHistoryResponse:
    """
    Get chat history for a session.
    
    Args:
        request: Session request with session ID
        
    Returns:
        Chat history response
        
    Raises:
        HTTPException: If session not found
    """
    session_data, found = session_manager.get_session(request.session_id)
    if not found:
        raise HTTPException(status_code=404, detail=ErrorMessages.SESSION_NOT_FOUND)
    
    return ChatHistoryResponse(
        status="success",
        history=session_data.get("chat_history", [])
    )


async def clear_history_handler(request: SessionRequest) -> StatusResponse:
    """
    Clear chat history for a session.
    
    Args:
        request: Session request with session ID
        
    Returns:
        Status response
        
    Raises:
        HTTPException: If session not found
    """
    success = session_manager.clear_chat_history(request.session_id)
    if not success:
        raise HTTPException(status_code=404, detail=ErrorMessages.SESSION_NOT_FOUND)
    
    return StatusResponse(
        status="success",
        message=SuccessMessages.CHAT_HISTORY_CLEARED
    )


async def remove_pdf_handler(request: SessionRequest) -> StatusResponse:
    """
    Remove PDF and session data.
    
    Args:
        request: Session request with session ID
        
    Returns:
        Status response
        
    Raises:
        HTTPException: If session not found or removal failed
    """
    success = session_manager.remove_session(request.session_id)
    
    if success:
        return StatusResponse(
            status="success",
            message=SuccessMessages.PDF_REMOVED
        )
    else:
        raise HTTPException(
            status_code=404, 
            detail=ErrorMessages.SESSION_REMOVAL_FAILED
        )
