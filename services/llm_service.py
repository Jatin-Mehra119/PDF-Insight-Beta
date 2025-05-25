"""
LLM service for model management and interaction.

This module provides services for LLM model creation and management.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from configs.config import Config, ErrorMessages


def create_llm_model(model_name: str) -> ChatGroq:
    """
    Create and configure an LLM model.
    
    Args:
        model_name: Name of the model to create
        
    Returns:
        Configured ChatGroq instance
        
    Raises:
        ValueError: If API key is missing for the model
    """
    if not os.getenv("GROQ_API_KEY") and "llama" in model_name:
        raise ValueError(ErrorMessages.GROQ_API_KEY_MISSING)
    
    llm = ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.MAX_TOKENS
    )
    return llm


def create_tavily_search_tool() -> Optional[TavilySearchResults]:
    """
    Create Tavily search tool with error handling.
    
    Returns:
        TavilySearchResults instance or None if creation fails
    """
    try:
        if not os.getenv("TAVILY_API_KEY"):
            print(f"Warning: {ErrorMessages.TAVILY_API_KEY_MISSING}")
            return None
        
        return TavilySearchResults(
            max_results=Config.TAVILY_MAX_RESULTS,
            search_depth=Config.TAVILY_SEARCH_DEPTH,
            include_answer=Config.TAVILY_INCLUDE_ANSWER,
            include_raw_content=Config.TAVILY_INCLUDE_RAW_CONTENT
        )
    except Exception as e:
        print(f"Warning: Could not create Tavily tool: {e}")
        return None


def validate_api_keys(model_name: str, use_search: bool = False) -> None:
    """
    Validate that required API keys are available.
    
    Args:
        model_name: LLM model name
        use_search: Whether web search is requested
        
    Raises:
        ValueError: If required API keys are missing
    """
    if not os.getenv("GROQ_API_KEY") and "llama" in model_name:
        raise ValueError(ErrorMessages.GROQ_API_KEY_MISSING)
    
    if use_search and not os.getenv("TAVILY_API_KEY"):
        print(f"Warning: {ErrorMessages.TAVILY_API_KEY_MISSING}")


def get_available_models() -> list:
    """
    Get list of available models.
    
    Returns:
        List of available model configurations
    """
    from configs.config import ModelConfig
    return ModelConfig.AVAILABLE_MODELS


def is_model_supported(model_name: str) -> bool:
    """
    Check if a model is supported.
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if model is supported, False otherwise
    """
    from configs.config import ModelConfig
    return ModelConfig.is_valid_model(model_name)
