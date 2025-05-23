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
    tools as global_base_tools,  
    create_vector_search_tool  
)
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory

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
    sessions[session_id] = data # Keep non-picklable in memory for active session
    
    pickle_safe_data = {
        "file_path": data.get("file_path"),
        "file_name": data.get("file_name"),
        "chunks": data.get("chunks"), # Chunks with metadata (list of dicts)
        "chat_history": data.get("chat_history", [])
        # FAISS index, embedding model, and LLM model are not pickled, will be reloaded/recreated
    }
    
    with open(f"{UPLOAD_DIR}/{session_id}_session.pkl", "wb") as f:
        pickle.dump(pickle_safe_data, f)


# Function to load session data
def load_session(session_id, model_name="llama3-8b-8192"): # Ensure model_name matches default
    try:
        if session_id in sessions:
            cached_session = sessions[session_id]
            # Ensure LLM and potentially other non-pickled parts are up-to-date or loaded
            if cached_session.get("llm") is None or (hasattr(cached_session["llm"], "model_name") and cached_session["llm"].model_name != model_name):
                 cached_session["llm"] = model_selection(model_name)
            if cached_session.get("model") is None: # Embedding model
                 cached_session["model"] = SentenceTransformer('BAAI/bge-large-en-v1.5')
            if cached_session.get("index") is None and cached_session.get("chunks"): # FAISS index
                embeddings, _ = create_embeddings(cached_session["chunks"], cached_session["model"])
                cached_session["index"] = build_faiss_index(embeddings)
            return cached_session, True
        
        file_path_pkl = f"{UPLOAD_DIR}/{session_id}_session.pkl"
        if os.path.exists(file_path_pkl):
            with open(file_path_pkl, "rb") as f:
                data = pickle.load(f)
            
            original_pdf_path = data.get("file_path")
            if data.get("chunks") and original_pdf_path and os.path.exists(original_pdf_path):
                embedding_model_instance = SentenceTransformer('BAAI/bge-large-en-v1.5')
                # Chunks are already {text: ..., metadata: ...}
                recreated_embeddings, _ = create_embeddings(data["chunks"], embedding_model_instance)
                recreated_index = build_faiss_index(recreated_embeddings)
                recreated_llm = model_selection(model_name)

                full_session_data = {
                    "file_path": original_pdf_path,
                    "file_name": data.get("file_name"),
                    "chunks": data.get("chunks"), # chunks_with_metadata
                    "chat_history": data.get("chat_history", []),
                    "model": embedding_model_instance,    # SentenceTransformer model
                    "index": recreated_index,            # FAISS index
                    "llm": recreated_llm                 # LLM
                }
                sessions[session_id] = full_session_data
                return full_session_data, True
            else:
                print(f"Warning: Session data for {session_id} is incomplete or PDF missing. Cannot reconstruct.")
                if os.path.exists(file_path_pkl): os.remove(file_path_pkl) # Clean up stale pkl
                return None, False
        
        return None, False
    except Exception as e:
        print(f"Error loading session {session_id}: {str(e)}")
        print(traceback.format_exc())
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
    model_name: str = Form("llama3-8b-8192") # Default model
):
    session_id = str(uuid.uuid4())
    file_path = None
    
    try:
        file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not os.getenv("GROQ_API_KEY") and "llama" in model_name: # Llama specific check for Groq
             raise ValueError("GROQ_API_KEY is not set for Groq Llama models.")
        if not os.getenv("TAVILY_API_KEY"): # Needed for TavilySearchResults
            print("Warning: TAVILY_API_KEY is not set. Web search will not function.")

        documents = process_pdf_file(file_path)
        # Ensure max_length for chunk_text is appropriate.
        # The value 1500 might be too large if estimate_tokens is text_len // 4, as it means ~6000 characters.
        # Let's use a smaller max_length for chunks for better granularity in RAG retrieval.
        # For `bge-large-en-v1.5` (max sequence length 512 tokens), chunks around 250-400 tokens are often good.
        # If estimate_tokens is len(text)//4, then max_length of 250 tokens is roughly 1000 characters.
        # Let's use max_length=256 (tokens) for chunker config, so about 1024 characters.
        # The chunk_text function uses max_length as character count / 4. So if we want 256 tokens, max_length = 256*4 = 1024
        # However, the current chunk_text logic is `estimate_tokens(current_chunk + paragraph) <= max_length // 4`.
        # This means `max_length` is already considered a token limit. So `max_length=256` (tokens) is the target.
        chunks_with_metadata = chunk_text(documents, max_length=256) # max_length in tokens
        
        embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        embeddings, _ = create_embeddings(chunks_with_metadata, embedding_model) # Chunks are already with metadata
        
        index = build_faiss_index(embeddings)
        llm = model_selection(model_name)
        
        session_data = {
            "file_path": file_path,
            "file_name": file.filename,
            "chunks": chunks_with_metadata, # Store chunks with metadata
            "model": embedding_model,       # SentenceTransformer instance
            "index": index,                 # FAISS index instance
            "llm": llm,                     # LLM instance
            "chat_history": []
        }
        save_session(session_id, session_data)
        
        return {"status": "success", "session_id": session_id, "message": f"Processed {file.filename}"}
    
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error processing PDF: {error_msg}\nStack trace: {stack_trace}")
        return JSONResponse(
            status_code=500, # Internal server error for processing issues
            content={"status": "error", "detail": error_msg, "type": type(e).__name__}
        )

# Route to chat with the document
@app.post("/chat")
async def chat(request: ChatRequest):
    session, found = load_session(request.session_id, model_name=request.model_name)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload a document first.")
    
    try:
        # Per-request memory to ensure chat history is correctly loaded for the agent
        agent_memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)
        for entry in session.get("chat_history", []):
            agent_memory.chat_memory.add_user_message(entry["user"])
            agent_memory.chat_memory.add_ai_message(entry["assistant"])

        # Prepare tools for the agent for THIS request
        current_request_tools = []

        # 1. Add the document-specific vector search tool
        if "index" in session and "chunks" in session and "model" in session:
            vector_search_tool_instance = create_vector_search_tool(
                faiss_index=session["index"],
                document_chunks_with_metadata=session["chunks"], # Pass the correct variable
                embedding_model=session["model"] # This is the SentenceTransformer model
            )
            current_request_tools.append(vector_search_tool_instance)
        else:
            print(f"Warning: Session {request.session_id} missing data for vector_database_search tool.")

        # 2. Conditionally add Tavily (web search) tool
        if request.use_search:
            if os.getenv("TAVILY_API_KEY"):
                tavily_tool = next((tool for tool in global_base_tools if tool.name == "tavily_search_results_json"), None)
                if tavily_tool:
                    current_request_tools.append(tavily_tool)
                else: # Should not happen if global_base_tools is defined correctly
                    print("Warning: Tavily search requested, but tool misconfigured.")
            else:
                print("Warning: Tavily search requested, but TAVILY_API_KEY is not set.")
        
        # Retrieve initial similar chunks for RAG context (can be empty if no good match)
        # This context is given to the agent *before* it decides to use tools.
        # k=5 means we retrieve up to 5 chunks for initial context.
        # The agent can then use `vector_database_search` to search more if needed.
        initial_similar_chunks = retrieve_similar_chunks(
            request.query, 
            session["index"], 
            session["chunks"], # list of dicts {text:..., metadata:...}
            session["model"], # SentenceTransformer model
            k=5 # Number of chunks for initial context
        )
        
        response = agentic_rag(
            session["llm"], 
            current_request_tools, # Pass the dynamically assembled list of tools
            query=request.query, 
            context_chunks=initial_similar_chunks,
            Use_Tavily=request.use_search, # Still passed to agentic_rag for potential fine-grained logic, though prompt adapts to tools
            memory=agent_memory
        )
        
        response_output = response.get("output", "Sorry, I could not generate a response.")
        session["chat_history"].append({"user": request.query, "assistant": response_output})
        save_session(request.session_id, session) # Save updated history and potentially other modified session state
        
        return {
            "status": "success", 
            "answer": response_output,
            # Return context that was PRE-FETCHED for the agent, not necessarily all context it might have used via tools
            "context_used": [{"text": chunk, "score": float(score), "metadata": meta} for chunk, score, meta in initial_similar_chunks]
        }
            
    except Exception as e:
        print(f"Error processing chat query: {str(e)}\nTraceback: {traceback.format_exc()}")
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