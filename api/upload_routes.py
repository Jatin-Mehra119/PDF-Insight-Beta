"""
File upload API routes.

This module handles PDF file upload and processing endpoints.
"""

import os
import shutil
import traceback
import uuid
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from configs.config import Config, ErrorMessages, SuccessMessages
from models.models import UploadResponse
from services import session_manager, validate_api_keys
from utils import process_pdf_file, chunk_text, create_upload_file_path


async def upload_pdf_handler(
    file: UploadFile = File(...), 
    model_name: str = Form(Config.DEFAULT_MODEL)
) -> UploadResponse:
    """
    Handle PDF file upload and processing.
    
    Args:
        file: Uploaded PDF file
        model_name: LLM model name to use
        
    Returns:
        Upload response with session ID
        
    Raises:
        HTTPException: If processing fails
    """
    session_id = str(uuid.uuid4())
    file_path = None
    
    try:
        # Validate API keys
        validate_api_keys(model_name, use_search=False)
        
        # Save uploaded file
        file_path = create_upload_file_path(session_id, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF file
        documents = process_pdf_file(file_path)
        chunks_with_metadata = chunk_text(documents, max_length=Config.DEFAULT_CHUNK_SIZE)
        
        # Create session
        session_id = session_manager.create_session(
            file_path=file_path,
            file_name=file.filename,
            chunks_with_metadata=chunks_with_metadata,
            model_name=model_name
        )
        
        return UploadResponse(
            status="success",
            session_id=session_id,
            message=SuccessMessages.PDF_PROCESSED.format(filename=file.filename)
        )
    
    except Exception as e:
        # Clean up file on error
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error processing PDF: {error_msg}\nStack trace: {stack_trace}")
        
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.PDF_PROCESSING_ERROR.format(error=error_msg)
        )
