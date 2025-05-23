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
    llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
    return llm
    
tools = [TavilySearchResults(max_results=5)]

# Initialize memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
            if estimate_tokens(current_chunk + paragraph) <= max_length // 4:
                current_chunk += paragraph + "\n\n"
            else:
                chunks.append({"text": current_chunk.strip(), "metadata": current_metadata})
                current_chunk = paragraph + "\n\n"
        if current_chunk:
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
    
    # Ensure indices are within bounds of chunks_with_metadata
    valid_indices = [i for i in indices[0] if 0 <= i < len(chunks_with_metadata)]
    
    return [
        (chunks_with_metadata[i]["text"][:max_chunk_length], distances[0][j], chunks_with_metadata[i]["metadata"])
        for j, i in enumerate(valid_indices) # Use valid_indices
    ]


def create_vector_search_tool(faiss_index, document_chunks_with_metadata, embedding_model, k=3, max_chunk_length=1000):
    @tool
    def vector_database_search(query: str) -> str:
        """
        Searches the currently uploaded PDF document for information semantically similar to the query.
        Use this tool when the user's question is likely answerable from the content of the specific document they provided.
        Input should be the search query.
        """
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
            return "No relevant information found in the document for that query."
        
        context = "\n\n---\n\n".join([chunk_text for chunk_text, _, _ in similar_chunks_data])
        return f"The following information was found in the document regarding '{query}':\n{context}"

    return vector_database_search

def agentic_rag(llm, agent_specific_tools, query, context_chunks, memory, Use_Tavily=False): # Renamed 'tools' to 'agent_specific_tools'
    # Sort chunks by relevance (lower distance = more relevant)
    context_chunks = sorted(context_chunks, key=lambda x: x[1]) if context_chunks else []
    context = ""
    total_tokens = 0
    max_tokens = 7000  # Leave room for prompt and response

    for chunk, _, _ in context_chunks:
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens <= max_tokens:
            context += chunk + "\n\n"
            total_tokens += chunk_tokens
        else:
            break
    
    context = context.strip() if context else "No initial context provided from preliminary search."


    # Dynamically build the tool guidance for the prompt
    # Tool names: 'vector_database_search', 'tavily_search_results_json'
    has_document_search = any(t.name == "vector_database_search" for t in agent_specific_tools)
    has_web_search = any(t.name == "tavily_search_results_json" for t in agent_specific_tools)

    guidance_parts = []
    if has_document_search:
        guidance_parts.append(
            "If the direct context (if any from preliminary search) is insufficient and the question seems answerable from the uploaded document, "
            "use the 'vector_database_search' tool to find relevant information within the document."
        )
    if has_web_search: # Tavily tool would only be in agent_specific_tools if Use_Tavily was true
        guidance_parts.append(
            "If the information is not found in the document (after using 'vector_database_search' if appropriate) "
            "or the question is of a general nature not specific to the document, "
            "use the 'tavily_search_results_json' tool for web searches."
        )

    if not guidance_parts:
        search_behavior_instructions = "If the context is insufficient, you *must* state that you don't know."
    else:
        search_behavior_instructions = " ".join(guidance_parts)
        search_behavior_instructions += ("\n    * If, after all steps and tool use (if any), you cannot find an answer, "
                                         "respond with: \"Based on the available information, I don't know the answer.\"")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert Q&A system. Your primary function is to answer questions using a given set of documents (Context) and available tools.

**Your Process:**

1.  **Analyze the Question:** Understand exactly what the user is asking.
2.  **Scan the Context:** Thoroughly review the 'Context' provided (if any) to find relevant information. This context is derived from a preliminary similarity search in the document.
3.  **Formulate the Answer:**
    * If the initially provided context contains a clear answer, synthesize it into a concise response. Start your answer with "Based on the Document, ...".
    * {search_behavior_instructions}
    * When using the 'vector_database_search' tool, the information comes from the document. Prepend your answer with "Based on the Document, ...".
    * When using the 'tavily_search_results_json' tool, the information comes from the web. Prepend your answer with "According to a web search, ...". If no useful information is found, state that.
4.  **Clarity:** Ensure your final answer is clear, direct, and avoids jargon if possible.

**Important Rules:**

* **Stick to Sources:** Do *not* use any information outside of the provided 'Context', document search results ('vector_database_search'), or web search results ('tavily_search_results_json').
* **No Speculation:** Do not make assumptions or infer information not explicitly present.
* **Cite Sources (If Web Searching):** If you use the 'tavily_search_results_json' tool and it provides source links, you MUST include them in your response.
        """),
        ("human", "Context: {{context}}\n\nQuestion: {{input}}"), # Double braces for f-string in f-string
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    try:
        agent = create_tool_calling_agent(llm, agent_specific_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=agent_specific_tools, memory=memory, verbose=True)
        response_payload = agent_executor.invoke({
            "input": query,
            "context": context,
        })
        return response_payload # Expecting dict like {'output': '...'}
    except Exception as e:
        print(f"Error during agent execution: {str(e)} \nTraceback: {traceback.format_exc()}")
        fallback_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question. If the context is insufficient, say you don't know."),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
        # Format the prompt with the actual context and query
        formatted_fallback_prompt = fallback_prompt_template.format_prompt(context=context, input=query).to_messages()
        response = llm.invoke(formatted_fallback_prompt)
        return {"output": response.content if hasattr(response, 'content') else str(response)} 

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