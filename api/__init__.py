"""
API routes module initialization.

This module provides easy imports for all API route handlers.
"""

from .upload_routes import upload_pdf_handler
from .chat_routes import chat_handler
from .session_routes import (
    get_chat_history_handler,
    clear_history_handler,
    remove_pdf_handler
)
from .utility_routes import (
    root_handler,
    get_models_handler
)

__all__ = [
    # Upload routes
    "upload_pdf_handler",
    
    # Chat routes
    "chat_handler",
    
    # Session routes
    "get_chat_history_handler",
    "clear_history_handler", 
    "remove_pdf_handler",
    
    # Utility routes
    "root_handler",
    "get_models_handler"
]
