import os
from langchain_community.document_loaders import PyMuPDFLoader
import faiss
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import dotenv
from langchain.tools import tool
import traceback
dotenv.load_dotenv()
# Initialize LLM and tools globally

def model_selection(model_name):
    llm = ChatGroq(
        model=model_name, 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,  # Lower temperature for more consistent tool calling
        max_tokens=2048   # Reasonable limit for responses
    )
    return llm
    
# Create tools with better error handling
def create_tavily_tool():
    try:
        return TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
    except Exception as e:
        print(f"Warning: Could not create Tavily tool: {e}")
        return None

# Initialize tools globally but with error handling
_tavily_tool = create_tavily_tool()
tools = [_tavily_tool] if _tavily_tool else []

# Note: Memory should be created per session, not globally

def estimate_tokens(text):
    """Estimate the number of tokens in a text (rough approximation)."""
    return len(text) // 4

def process_pdf_file(file_path):
    """Load a PDF file and extract its text with metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents  # Return list of Document objects with metadata

def chunk_text(documents, max_length=1000):
    """Split documents into chunks with metadata."""
    chunks = []
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        paragraphs = text.split("\n\n")
        current_chunk = ""
        current_metadata = metadata.copy()
        for paragraph in paragraphs:
            # Skip very short paragraphs (less than 10 characters)
            if len(paragraph.strip()) < 10:
                continue
                
            if estimate_tokens(current_chunk + paragraph) <= max_length // 4:
                current_chunk += paragraph + "\n\n"
            else:
                # Only add chunks with meaningful content
                if current_chunk.strip() and len(current_chunk.strip()) > 20:
                    chunks.append({"text": current_chunk.strip(), "metadata": current_metadata})
                current_chunk = paragraph + "\n\n"
        # Add the last chunk if it has meaningful content
        if current_chunk.strip() and len(current_chunk.strip()) > 20:
            chunks.append({"text": current_chunk.strip(), "metadata": current_metadata})
    return chunks

def create_embeddings(chunks, model):
    """Create embeddings for a list of chunk texts."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu().numpy(), chunks

def build_faiss_index(embeddings):
    """Build a FAISS HNSW index from embeddings for similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 = number of neighbors in HNSW graph
    index.hnsw.efConstruction = 200  # Higher = better quality, slower build
    index.hnsw.efSearch = 50  # Higher = better accuracy, slower search
    index.add(embeddings)
    return index

def retrieve_similar_chunks(query, index, chunks_with_metadata, embedding_model, k=10, max_chunk_length=1000):
    """Retrieve top k similar chunks to the query from the FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    
    # Ensure indices are within bounds and create mapping for correct distances
    valid_results = []
    for idx_pos, chunk_idx in enumerate(indices[0]):
        if 0 <= chunk_idx < len(chunks_with_metadata):
            chunk_text = chunks_with_metadata[chunk_idx]["text"][:max_chunk_length]
            # Only include chunks with meaningful content
            if chunk_text.strip():  # Skip empty chunks
                valid_results.append((
                    chunk_text,
                    distances[0][idx_pos],  # Use original position for correct distance
                    chunks_with_metadata[chunk_idx]["metadata"]
                ))
    
    return valid_results


def create_vector_search_tool(faiss_index, document_chunks_with_metadata, embedding_model, k=3, max_chunk_length=1000):
    @tool
    def vector_database_search(query: str) -> str:
        """Search the uploaded PDF document for information related to the query.
        
        Args:
            query: The search query string to find relevant information in the document.
            
        Returns:
            A string containing relevant information found in the document.
        """
        # Handle very short or empty queries
        if not query or len(query.strip()) < 3:
            return "Please provide a more specific search query with at least 3 characters."
        
        try:
            # Retrieve similar chunks using the provided session-specific components
            similar_chunks_data = retrieve_similar_chunks(
                query,
                faiss_index,
                document_chunks_with_metadata, # This is the list of dicts {text: ..., metadata: ...}
                embedding_model,
                k=k,
                max_chunk_length=max_chunk_length
            )
            
            # Format the response
            if not similar_chunks_data:
                return "No relevant information found in the document for that query. Please try rephrasing your question or using different keywords."
            
            # Filter out chunks with very high distance (low similarity)
            filtered_chunks = [chunk for chunk in similar_chunks_data if chunk[1] < 1.5]  # Adjust threshold as needed
            
            if not filtered_chunks:
                return "No sufficiently relevant information found in the document for that query. Please try rephrasing your question or using different keywords."
            
            context = "\n\n---\n\n".join([chunk_text for chunk_text, _, _ in filtered_chunks])
            return f"The following information was found in the document regarding '{query}':\n{context}"
            
        except Exception as e:
            print(f"Error in vector search tool: {e}")
            return f"Error searching the document: {str(e)}"

    return vector_database_search

def agentic_rag(llm, agent_specific_tools, query, context_chunks, memory, Use_Tavily=False):
    # Validate inputs
    if not query or not query.strip():
        return {"output": "Please provide a valid question."}
    
    if not agent_specific_tools:
        print("Warning: No tools provided, using direct LLM response")
        # Use direct LLM call without agent if no tools
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about documents. Use the provided context to answer the user's question."),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
        try:
            formatted_prompt = fallback_prompt.format_prompt(context="No context available", input=query).to_messages()
            response = llm.invoke(formatted_prompt)
            return {"output": response.content if hasattr(response, 'content') else str(response)}
        except Exception as e:
            print(f"Direct LLM call failed: {e}")
            return {"output": "I'm sorry, I encountered an error processing your request."}
    
    print(f"Available tools: {[tool.name for tool in agent_specific_tools]}")
    
    # Sort chunks by relevance (lower distance = more relevant)
    context_chunks = sorted(context_chunks, key=lambda x: x[1]) if context_chunks else []
    context = ""
    total_tokens = 0
    max_tokens = 7000  # Leave room for prompt and response

    # Filter out chunks with very high distance scores (low similarity)
    relevant_chunks = [chunk for chunk in context_chunks if len(chunk) >= 3 and chunk[1] < 1.5]

    for chunk, _, _ in relevant_chunks:
        if chunk and chunk.strip():  # Ensure chunk has content
            chunk_tokens = estimate_tokens(chunk)
            if total_tokens + chunk_tokens <= max_tokens:
                context += chunk + "\n\n"
                total_tokens += chunk_tokens
            else:
                break
    
    context = context.strip() if context else "No initial context provided from preliminary search."
    print(f"Using context length: {len(context)} characters")


    # Dynamically build the tool guidance for the prompt
    # Tool names: 'vector_database_search', 'tavily_search_results_json'
    has_document_search = any(t.name == "vector_database_search" for t in agent_specific_tools)
    has_web_search = any(t.name == "tavily_search_results_json" for t in agent_specific_tools)

    # Simplified tool guidance
    tool_instructions = ""
    if has_document_search:
        tool_instructions += "Use vector_database_search to find information in the uploaded document. "
    if has_web_search:
        tool_instructions += "Use tavily_search_results_json for web searches when document search is insufficient. "
    
    if not tool_instructions:
        tool_instructions = "Answer based on the provided context only. "

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful AI assistant that answers questions about documents.

Context: {{context}}

Tools available: {tool_instructions}

Instructions:
- Use the provided context first
- If context is insufficient, use available tools to search for more information
- Provide clear, helpful answers
- If you cannot find an answer, say so clearly"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    try:
        print(f"Creating agent with {len(agent_specific_tools)} tools")
        
        # Validate that tools are properly formatted
        for tool in agent_specific_tools:
            print(f"Tool: {tool.name} - {type(tool)}")
            # Ensure tool has required attributes
            if not hasattr(tool, 'name') or not hasattr(tool, 'description'):
                raise ValueError(f"Tool {tool} is missing required attributes")
        
        agent = create_tool_calling_agent(llm, agent_specific_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=agent_specific_tools, 
            memory=memory, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,  # Reduced further to prevent issues
            return_intermediate_steps=False,
            early_stopping_method="generate"
        )
        
        print(f"Invoking agent with query: '{query}' and context length: {len(context)} chars")
        
        # Create input with simpler structure
        agent_input = {
            "input": query,
            "context": context,
        }
        
        response_payload = agent_executor.invoke(agent_input)
        
        print(f"Agent response keys: {response_payload.keys() if response_payload else 'None'}")
        
        # Extract and validate the output
        agent_output = response_payload.get("output", "") if response_payload else ""
        print(f"Agent output length: {len(agent_output)} chars")
        print(f"Agent output preview: {agent_output[:100]}..." if len(agent_output) > 100 else f"Agent output: {agent_output}")
        
        # Validate response quality
        if not agent_output or len(agent_output.strip()) < 10:
            print(f"Warning: Agent returned insufficient response (length: {len(agent_output)}), using fallback")
            raise ValueError("Insufficient response from agent")
        
        # Check if response is just a prefix without content
        problematic_prefixes = [
            "Based on the Document,",
            "According to a web search,", 
            "Based on the available information,",
            "I need to",
            "Let me"
        ]
        
        stripped_output = agent_output.strip()
        if any(stripped_output == prefix.strip() or stripped_output == prefix.strip() + "." for prefix in problematic_prefixes):
            print(f"Warning: Agent returned only prefix without content: '{stripped_output}', using fallback")
            raise ValueError("Agent returned incomplete response")
            
        return response_payload
    except Exception as e:
        error_msg = str(e)
        print(f"Error during agent execution: {error_msg} \nTraceback: {traceback.format_exc()}")
        
        # Check if it's a specific Groq function calling error
        if "Failed to call a function" in error_msg or "function" in error_msg.lower():
            print("Detected Groq function calling error, trying simpler approach...")
            
            # Try with a simpler agent setup or direct LLM call
            try:
                # First, try to use tools individually without agent framework
                if agent_specific_tools:
                    print("Attempting manual tool usage...")
                    tool_results = []
                    
                    # Try vector search first if available
                    vector_tool = next((t for t in agent_specific_tools if t.name == "vector_database_search"), None)
                    if vector_tool:
                        try:
                            search_result = vector_tool.run(query)
                            if search_result and "No relevant information" not in search_result:
                                tool_results.append(f"Document Search: {search_result}")
                        except Exception as tool_error:
                            print(f"Vector tool error: {tool_error}")
                    
                    # Try web search if needed and available
                    if Use_Tavily:
                        web_tool = next((t for t in agent_specific_tools if t.name == "tavily_search_results_json"), None)
                        if web_tool:
                            try:
                                web_result = web_tool.run(query)
                                if web_result:
                                    tool_results.append(f"Web Search: {web_result}")
                            except Exception as tool_error:
                                print(f"Web tool error: {tool_error}")
                    
                    # Combine tool results with context
                    enhanced_context = context
                    if tool_results:
                        enhanced_context += "\n\nAdditional Information:\n" + "\n\n".join(tool_results)
                    
                    # Use direct LLM call with enhanced context
                    direct_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Use the provided context and information to answer the user's question clearly and completely."),
                        ("human", "Context and Information: {context}\n\nQuestion: {input}")
                    ])
                    
                    formatted_prompt = direct_prompt.format_prompt(context=enhanced_context, input=query).to_messages()
                    response = llm.invoke(formatted_prompt)
                    direct_output = response.content if hasattr(response, 'content') else str(response)
                    print(f"Direct tool usage response length: {len(direct_output)} chars")
                    return {"output": direct_output}
                    
            except Exception as manual_error:
                print(f"Manual tool usage also failed: {manual_error}")
        
        print("Using fallback direct LLM response...")
        
        fallback_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about documents. 
            Use the provided context to answer the user's question. 
            If the context contains relevant information, start your answer with "Based on the Document, ..."
            If the context is insufficient, clearly state what you don't know."""),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
        
        try:
            # Format the prompt with the actual context and query
            formatted_fallback_prompt = fallback_prompt_template.format_prompt(context=context, input=query).to_messages()
            response = llm.invoke(formatted_fallback_prompt)
            fallback_output = response.content if hasattr(response, 'content') else str(response)
            print(f"Fallback response length: {len(fallback_output)} chars")
            return {"output": fallback_output}
        except Exception as fallback_error:
            print(f"Fallback also failed: {str(fallback_error)}")
            return {"output": "I'm sorry, I encountered an error processing your request. Please try again."} 

"""if __name__ == "__main__":
    # Process PDF and prepare index
    dotenv.load_dotenv()
    pdf_file = "JatinCV.pdf"
    llm = model_selection("meta-llama/llama-4-scout-17b-16e-instruct")
    texts = process_pdf_file(pdf_file)
    chunks = chunk_text(texts, max_length=1500)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(chunks, model)
    index = build_faiss_index(embeddings)

    # Chat loop
    print("Chat with the assistant (type 'exit' or 'quit' to stop):")
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # Retrieve similar chunks
        similar_chunks = retrieve_similar_chunks(query, index, chunks, model, k=3)
        # context = "\n".join([chunk for chunk, _ in similar_chunks])
        
        # Generate response
        response = agentic_rag(llm, tools, query=query, context=similar_chunks, Use_Tavily=True, memory=memory)
        print("Assistant:", response["output"])"""