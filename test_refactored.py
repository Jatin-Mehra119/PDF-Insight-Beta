"""
Test script to verify the refactored code works correctly.

This script tests the main functionality to ensure backward compatibility
and proper operation of the refactored modules.
"""

import sys
import os
import tempfile
import traceback

# Add the current directory to Python path
sys.path.insert(0, '/workspaces/PDF-Insight-Beta')

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test config import
        from configs.config import Config, ModelConfig, ErrorMessages
        print("✓ Config module imported successfully")
        
        # Test models import  
        from models.models import ChatRequest, UploadResponse
        print("✓ Models module imported successfully")
        
        # Test utils import
        from utils import estimate_tokens, process_pdf_file
        print("✓ Utils module imported successfully")
        
        # Test services import
        from services import create_llm_model, session_manager, rag_service
        print("✓ Services module imported successfully")
        
        # Test API import
        from api import upload_pdf_handler, chat_handler
        print("✓ API module imported successfully")
        
        # Test backward compatibility
        from preprocessing import model_selection, chunk_text, agentic_rag
        print("✓ Backward compatibility import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        from configs.config import Config
        from utils.text_processing import estimate_tokens
        from services.llm_service import get_available_models
        
        # Test token estimation
        tokens = estimate_tokens("This is a test string")
        assert tokens > 0
        print(f"✓ Token estimation works: {tokens} tokens")
        
        # Test model listing
        models = get_available_models()
        assert len(models) > 0
        print(f"✓ Model listing works: {len(models)} models available")
        
        # Test config access
        assert Config.DEFAULT_CHUNK_SIZE > 0
        print(f"✓ Config access works: chunk size = {Config.DEFAULT_CHUNK_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that original interfaces still work."""
    print("\nTesting backward compatibility...")
    
    try:
        # Test original preprocessing interface
        from preprocessing import model_selection, tools, estimate_tokens
        
        # These should work without errors
        assert callable(model_selection)
        assert isinstance(tools, list)
        assert callable(estimate_tokens)
        
        print("✓ Original preprocessing interface preserved")
        
        # Test that we can still access the original functions
        from preprocessing import (
            process_pdf_file, chunk_text, create_embeddings,
            build_faiss_index, retrieve_similar_chunks, agentic_rag
        )
        
        print("✓ All original functions accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("\nTesting app creation...")
    
    try:
        from app import create_app
        
        app = create_app()
        assert app is not None
        print("✓ FastAPI app created successfully")
        
        # Check that routes are properly defined
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/upload-pdf", "/chat", "/models"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} found")
            else:
                print(f"✗ Route {route} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ App creation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Refactored PDF Insight Beta")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality, 
        test_backward_compatibility,
        test_app_creation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Refactoring successful.")
        return 0
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
