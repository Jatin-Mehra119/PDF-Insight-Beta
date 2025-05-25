"""
Refactored FastAPI application for PDF Insight Beta.

This is the main application file that sets up the FastAPI app with modular components.
The core logic has been preserved while improving code organization and maintainability.
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from configs.config import Config
from models.models import (
    ChatRequest, SessionRequest, UploadResponse, ChatResponse, 
    ChatHistoryResponse, StatusResponse, ModelsResponse
)
from api import (
    upload_pdf_handler, chat_handler, get_chat_history_handler,
    clear_history_handler, remove_pdf_handler, root_handler, get_models_handler
)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Initialize FastAPI app
    app = FastAPI(
        title="PDF Insight Beta", 
        description="Agentic RAG for PDF documents"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.CORS_ORIGINS,
        allow_credentials=Config.CORS_CREDENTIALS,
        allow_methods=Config.CORS_METHODS,
        allow_headers=Config.CORS_HEADERS,
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    return app


# Create app instance
app = create_app()


# Route definitions
@app.get("/")
async def read_root():
    """Root endpoint that redirects to the main application."""
    return await root_handler()


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), model_name: str = Form(Config.DEFAULT_MODEL)):
    """Upload and process a PDF file."""
    return await upload_pdf_handler(file, model_name)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the uploaded document."""
    return await chat_handler(request)


@app.post("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history(request: SessionRequest):
    """Get chat history for a session."""
    return await get_chat_history_handler(request)


@app.post("/clear-history", response_model=StatusResponse)
async def clear_history(request: SessionRequest):
    """Clear chat history for a session."""
    return await clear_history_handler(request)


@app.post("/remove-pdf", response_model=StatusResponse)
async def remove_pdf(request: SessionRequest):
    """Remove PDF file and session data."""
    return await remove_pdf_handler(request)


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get list of available models."""
    return await get_models_handler()


def main():
    """
    Main entry point for running the application.
    """
    uvicorn.run("app_refactored:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
