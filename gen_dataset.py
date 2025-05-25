from datasets import load_dataset

ds = load_dataset("neural-bridge/rag-dataset-12000")

# Test the RAG system with DS dataset
from sentence_transformers import SentenceTransformer
from development_scripts.preprocessing import model_selection, create_embeddings, build_faiss_index, retrieve_similar_chunks, agentic_rag
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import json 
import gc  
import torch  # For clearing CUDA cache if available
import os
from langchain.memory import ConversationBufferMemory
import json
import csv
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# Configuration parameters
SAMPLE_SIZE = 80  # Number of documents to test
BATCH_SIZE = 1    # Save results after every X iterations
OUTPUT_FILE = 'rag_test_output.json'

tools = [TavilySearchResults(max_results=5)]
dotenv.load_dotenv()

# create a simple chunking function for text based
def chunk_text(text, max_length=250):
    # Split the text into chunks of max_length with metadata
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        chunks.append({"text": chunk, "metadata": {"chunk_id": i // max_length}})
    return chunks

# Function to clear memory
def clear_memory():
    gc.collect()  # Run garbage collector
    if torch.cuda.is_available():  # If using GPU
        torch.cuda.empty_cache()  # Clear CUDA cache

# Initialize or load output data
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        try:
            output_data = json.load(f)
            start_idx = len(output_data)  # Resume from where we left off
            print(f"Resuming from index {start_idx}")
        except json.JSONDecodeError:
            output_data = []  # Start fresh if file is corrupted
            start_idx = 0
else:
    output_data = []  # Start fresh if file doesn't exist
    start_idx = 0

# Process documents in range
try:
    for i in range(start_idx, min(start_idx + SAMPLE_SIZE, len(ds['train']))):
        print(f"Processing document {i}/{min(start_idx + SAMPLE_SIZE, len(ds['train']))}")
        
        # Get current document data
        llm = model_selection("meta-llama/llama-4-scout-17b-16e-instruct")
        current_context_text = ds['train'][i]['context']
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Process text and create embeddings
        chunks = chunk_text(current_context_text, max_length=100)
        embeddings, chunks = create_embeddings(chunks, model)
        index = build_faiss_index(embeddings)
        query = ds['train'][i]['question']
        
        # Retrieve similar chunks
        similar_chunks = retrieve_similar_chunks(query, index, chunks, model, k=5)
        agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Run RAG system
        print(f"Query: {query}")
        response = agentic_rag(llm, tools, query=query, context_chunks=similar_chunks, memory=agent_memory, Use_Tavily=False)
        
        print("Assistant:", response["output"])
        print("Ground Truth:", ds['train'][i]['answer'])
        print("==="*50)
        
        # Store the results
        output_data.append({
            "query": query,
            "assistant_response": response["output"],
            "ground_truth": ds['train'][i]['answer'],
            "context": current_context_text
        })
        
        # Save results periodically to preserve memory
        if (i + 1) % BATCH_SIZE == 0 or i == min(start_idx + SAMPLE_SIZE, len(ds['train'])) - 1:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"\nSaved results for {len(output_data)} documents to {OUTPUT_FILE}")
        
        # Clear memory
        del llm, current_context_text, model, chunks, embeddings, index, similar_chunks, response
        clear_memory()
        
except Exception as e:
    print(f"Error occurred at document index {i}: {str(e)}")
    # Save whatever results we have so far
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nSaved partial results for {len(output_data)} documents to {OUTPUT_FILE}")
    
print(f"\nCompleted processing {len(output_data)} documents. Results saved to {OUTPUT_FILE}")


# Load model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# File paths
input_file = 'rag_test_output.json'
output_file = 'rag_scores.csv'
semantic_threshold = 0.75

# Read JSON array
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []

# Score each item
for item in data:
    query = item.get("query", "")
    assistant_response = item.get("assistant_response", "")
    ground_truth = item.get("ground_truth", "")
    context = item.get("context", "")

    # Compute semantic similarity
    emb_response = model.encode(assistant_response, convert_to_tensor=True)
    emb_truth = model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb_response, emb_truth).item()

    # Compute ROUGE-L F1
    rouge_score = rouge.score(assistant_response, ground_truth)['rougeL'].fmeasure

    # Final status
    status = "PASS" if similarity >= semantic_threshold else "FAIL"

    results.append({
        "query": query,
        "semantic_similarity": round(similarity, 4),
        "rougeL_f1": round(rouge_score, 4),
        "status": status
    })

# Write results to CSV
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["query", "semantic_similarity", "rougeL_f1", "status"])
    writer.writeheader()
    writer.writerows(results)

print(f"Scores saved to '{output_file}'")