"""
Utility API routes.

This module handles utility endpoints like model listing and health checks.
"""

from fastapi.responses import RedirectResponse

from models.models import ModelsResponse
from services import get_available_models


async def root_handler():
    """
    Root endpoint that redirects to the main application.
    
    Returns:
        Redirect response to static index.html
    """
    return RedirectResponse(url="/static/index.html")


async def get_models_handler() -> ModelsResponse:
    """
    Get list of available models.
    
    Returns:
        Models response with available model configurations
    """
    models = get_available_models()
    return ModelsResponse(models=models)
