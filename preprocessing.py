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

def retrieve_similar_chunks(query, index, chunks, model, k=10, max_chunk_length=1000):
    """Retrieve top k similar chunks to the query from the FAISS index."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return [(chunks[i]["text"][:max_chunk_length], distances[0][j], chunks[i]["metadata"]) for j, i in enumerate(indices[0])]

def agentic_rag(llm, tools, query, context_chunks, memory, Use_Tavily=False):
    # Sort chunks by relevance (lower distance = more relevant)
    context_chunks = sorted(context_chunks, key=lambda x: x[1])  # Sort by distance
    context = ""
    total_tokens = 0
    max_tokens = 7000  # Leave room for prompt and response
    
    # Aggregate relevant chunks until token limit is reached
    for chunk, _, _ in context_chunks:  # Unpack three elements
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens <= max_tokens:
            context += chunk + "\n\n"
            total_tokens += chunk_tokens
        else:
            break

    # Set up the search behavior
    search_behavior = (
    "If the context is insufficient, *then* use the 'search' tool to find the answer."
    if Use_Tavily
    else "If the context is insufficient, you *must* state that you don't know."
)
    
    # Define prompt template
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert Q&A system. Your primary function is to answer questions using a given set of documents (Context).

**Your Process:**

1.  **Analyze the Question:** Understand exactly what the user is asking.
2.  **Scan the Context:** Thoroughly review the 'Context' provided to find relevant information.
3.  **Formulate the Answer:**
    * If the context contains a clear answer, synthesize it into a concise response.
    * **Always** start your answer with "Based on the Document, ...".
    * {search_behavior}
    * If, after all steps, you cannot find an answer, respond with: "Based on the Document, I don't know the answer."
4.  **Clarity:** Ensure your final answer is clear, direct, and avoids jargon if possible.

**Important Rules:**

* **Stick to the Context:** Unless you use the search tool, do *not* use any information outside of the provided 'Context'.
* **No Speculation:** Do not make assumptions or infer information not explicitly present.
* **Cite Sources (If Searching):** If you use the search tool, you MUST include the source links in your response.
    """),
        ("human", "Context: {context}\n\nQuestion: {input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent_tools = tools if Use_Tavily else []
    try:
        agent = create_tool_calling_agent(llm, agent_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=agent_tools, memory=memory, verbose=True)
        return agent_executor.invoke({
            "input": query,
            "context": context,
            "search_behavior": search_behavior
        })
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
        response = llm.invoke(fallback_prompt.format(context=context, input=query))
        return {"output": response.content} 

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