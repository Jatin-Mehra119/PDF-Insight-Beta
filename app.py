import os
import dotenv
import pickle
import uuid
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from preprocessing import (
    model_selection,
    process_pdf_file,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_similar_chunks,
    agentic_rag,
    tools
)
from sentence_transformers import SentenceTransformer

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Insight Beta", description="Agentic RAG for PDF documents")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Store active sessions
sessions = {}

# Define model for chat request
class ChatRequest(BaseModel):
    session_id: str
    query: str
    use_search: bool = False
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"

class SessionRequest(BaseModel):
    session_id: str

# Function to save session data
def save_session(session_id, data):
    sessions[session_id] = data
    
    # Create a copy of data that is safe to pickle
    pickle_safe_data = {
        "file_path": data.get("file_path"),
        "file_name": data.get("file_name"),
        "chunks": data.get("chunks"),
        "chat_history": data.get("chat_history", [])
    }
    
    # Persist to disk
    with open(f"{UPLOAD_DIR}/{session_id}_session.pkl", "wb") as f:
        pickle.dump(pickle_safe_data, f)

# Function to load session data
def load_session(session_id, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
    try:
        # Check if session is already in memory
        if session_id in sessions:
            # Ensure the LLM in the cached session matches the requested model_name
            # If not, update it. This handles cases where model_name might change for an existing session.
            if sessions[session_id].get("llm") is None or sessions[session_id]["llm"].model_name != model_name:
                try:
                    sessions[session_id]["llm"] = model_selection(model_name)
                except Exception as e:
                    print(f"Error updating LLM for in-memory session {session_id} to {model_name}: {str(e)}")
                    # Decide if this is a critical error; for now, we'll proceed with the old LLM or handle as error
                    # For simplicity, if LLM update fails, we might want to indicate session load failure or use existing.
                    # Here, we'll let it proceed, but this could be a point of further refinement.
            return sessions[session_id], True
        
        # Try to load from disk
        file_path_pkl = f"{UPLOAD_DIR}/{session_id}_session.pkl"
        if os.path.exists(file_path_pkl):
            with open(file_path_pkl, "rb") as f:
                data = pickle.load(f)  # This is pickle_safe_data
            
            # Recreate non-pickled objects
            # Ensure 'chunks' and 'file_path' (for the original PDF) are present in the loaded data
            # and the original PDF file still exists.
            original_pdf_path = data.get("file_path")
            if data.get("chunks") and original_pdf_path and os.path.exists(original_pdf_path):
                embedding_model_instance = SentenceTransformer('BAAI/bge-large-en-v1.5')
                # data["chunks"] is already the list of dicts: {text: ..., metadata: ...}
                recreated_embeddings, _ = create_embeddings(data["chunks"], embedding_model_instance)
                recreated_index = build_faiss_index(recreated_embeddings)
                recreated_llm = model_selection(model_name)

                full_session_data = {
                    "file_path": original_pdf_path,
                    "file_name": data.get("file_name"),
                    "chunks": data.get("chunks"),  # These are chunks_with_metadata
                    "chat_history": data.get("chat_history", []),
                    "model": embedding_model_instance,    # SentenceTransformer model
                    "index": recreated_index,            # FAISS index
                    "llm": recreated_llm                 # LLM
                }
                sessions[session_id] = full_session_data  # Store in memory cache
                return full_session_data, True
            else:
                # If essential data for reconstruction is missing from pickle or the original PDF is gone
                print(f"Warning: Session data for {session_id} is incomplete or its PDF file '{original_pdf_path}' is missing. Cannot reconstruct session.")
                # Optionally, remove the stale .pkl file
                # os.remove(file_path_pkl) 
                return None, False
        
        return None, False  # Session not in memory and not found on disk, or reconstruction failed
    except Exception as e:
        print(f"Error loading session {session_id}: {str(e)}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return None, False

# Function to remove PDF file
def remove_pdf_file(session_id):
    try:
        # Check if the session exists
        session_path = f"{UPLOAD_DIR}/{session_id}_session.pkl"
        if os.path.exists(session_path):
            # Load session data
            with open(session_path, "rb") as f:
                data = pickle.load(f)
            
            # Delete PDF file if it exists
            if data.get("file_path") and os.path.exists(data["file_path"]):
                os.remove(data["file_path"])
            
            # Remove session file
            os.remove(session_path)
        
        # Remove from memory if exists
        if session_id in sessions:
            del sessions[session_id]
            
        return True
    except Exception as e:
        print(f"Error removing PDF file: {str(e)}")
        return False

# Mount static files (we'll create these later)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for the home page
@app.get("/")
async def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

# Route to upload a PDF file
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), 
    model_name: str = Form("meta-llama/llama-4-scout-17b-16e-instruct")
):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    file_path = None
    
    try:
        # Save the uploaded file
        file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check if API keys are set
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set in the environment variables")
        
        # Process the PDF
        documents = process_pdf_file(file_path)  # Returns list of Document objects
        chunks = chunk_text(documents, max_length=1500)  # Updated to handle documents
        
        # Create embeddings
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Updated embedding model
        embeddings, chunks_with_metadata = create_embeddings(chunks, model)  # Unpack tuple
        
        # Build FAISS index
        index = build_faiss_index(embeddings)  # Pass only embeddings array
        
        # Initialize LLM
        llm = model_selection(model_name)
        
        # Save session data
        session_data = {
            "file_path": file_path,
            "file_name": file.filename,
            "chunks": chunks_with_metadata,  # Store chunks with metadata
            "model": model,
            "index": index,
            "llm": llm,
            "chat_history": []
        }
        save_session(session_id, session_data)
        
        return {"status": "success", "session_id": session_id, "message": f"Processed {file.filename}"}
    
    except Exception as e:
        # Clean up on error
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error processing PDF: {error_msg}")
        print(f"Stack trace: {stack_trace}")
        
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": error_msg,
                "type": type(e).__name__
            }
        )

# Route to chat with the document
@app.post("/chat")
async def chat(request: ChatRequest):
    # Try to load session if not in memory
    session, found = load_session(request.session_id, model_name=request.model_name)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
    
    try:
        from langchain.memory import ConversationBufferMemory
        agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        for entry in session.get("chat_history", []):
            agent_memory.chat_memory.add_user_message(entry["user"])
            agent_memory.chat_memory.add_ai_message(entry["assistant"])
        

        # Retrieve similar chunks
        similar_chunks = retrieve_similar_chunks(
            request.query, 
            session["index"], 
            session["chunks"], 
            session["model"], 
            k=10
        )
        
        # Generate response using agentic_rag
        response = agentic_rag(
            session["llm"], 
            tools, 
            query=request.query, 
            context_chunks=similar_chunks,  # Pass the list of tuples
            Use_Tavily=request.use_search,
            memory=agent_memory
        )
        
        # Update chat history
        session["chat_history"].append({"user": request.query, "assistant": response["output"]})
        save_session(request.session_id, session)
        
        return {
            "status": "success", 
            "answer": response["output"],
            "context_used": [{"text": chunk, "score": float(score)} for chunk, score, _ in similar_chunks]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Route to get chat history
@app.post("/chat-history")
async def get_chat_history(request: SessionRequest):
    # Try to load session if not in memory
    session, found = load_session(request.session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "status": "success", 
        "history": session.get("chat_history", [])
    }

# Route to clear chat history
@app.post("/clear-history")
async def clear_history(request: SessionRequest):
    # Try to load session if not in memory
    session, found = load_session(request.session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session["chat_history"] = []
    save_session(request.session_id, session)
    
    return {"status": "success", "message": "Chat history cleared"}

# Route to remove PDF from session
@app.post("/remove-pdf")
async def remove_pdf(request: SessionRequest):
    success = remove_pdf_file(request.session_id)
    
    if success:
        return {"status": "success", "message": "PDF file and session removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found or could not be removed")

# Route to list available models
@app.get("/models")
async def get_models():
    # You can expand this list as needed
    models = [
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B"},
        {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant"},
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile"},
    ]
    return {"models": models}

# Run the application if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

