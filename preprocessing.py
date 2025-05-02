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
    """Load a PDF file and extract its text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text = "".join(doc.page_content for doc in documents)
    return text

def chunk_text(text, max_length=1500):
    """Split text into chunks based on paragraphs, respecting max_length."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_length:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_embeddings(texts, model):
    """Create embeddings for a list of texts using the provided model."""
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings for similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_similar_chunks(query, index, texts, model, k=3, max_chunk_length=3500):
    """Retrieve top k similar chunks to the query from the FAISS index."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return [(texts[i][:max_chunk_length], distances[0][j]) for j, i in enumerate(indices[0])]

def agentic_rag(llm, tools, query, context, Use_Tavily=False):
    # Define the prompt template for the agent
    search_instructions = (
        "Use the search tool if the context is insufficient to answer the question or you are unsure. Give source links if you use the search tool."
        if Use_Tavily
        else "Use the context provided to answer the question."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant. {search_instructions}
        Instructions:
        1. Use the provided context to answer the user's question.
        2. Provide a clear answer, if you don't know the answer, say 'I don't know'.
        """),
        ("human", "Context: {context}\n\nQuestion: {input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Only use tools when Tavily is enabled
    agent_tools = tools if Use_Tavily else []
    
    try:
        # Create the agent and executor with appropriate tools
        agent = create_tool_calling_agent(llm, agent_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=agent_tools, memory=memory, verbose=True)
        
        # Execute the agent
        return agent_executor.invoke({
            "input": query, 
            "context": context,
            "search_instructions": search_instructions
        })
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")
        # Fallback to direct LLM call without agent framework
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
        response = llm.invoke(fallback_prompt.format(context=context, input=query))
        return {"output": response.content}

if __name__ == "__main__":
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
        context = "\n".join([chunk for chunk, _ in similar_chunks])
        
        # Generate response
        response = agentic_rag(llm, tools, query=query, context=context, Use_Tavily=True)
        print("Assistant:", response["output"])