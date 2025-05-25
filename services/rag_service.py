"""
RAG (Retrieval Augmented Generation) service.

This module provides the RAG implementation with tool creation and agent management.
"""

import traceback
from typing import List, Dict, Any, Optional, Tuple
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import faiss

from configs.config import Config
from utils import (
    retrieve_similar_chunks,
    filter_relevant_chunks,
    prepare_context_from_chunks
)
from services.llm_service import create_tavily_search_tool


def create_vector_search_tool(
    faiss_index: faiss.IndexHNSWFlat,
    document_chunks_with_metadata: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    k: int = None,
    max_chunk_length: int = None
):
    """
    Create a vector search tool for document retrieval.
    
    Args:
        faiss_index: FAISS index for similarity search
        document_chunks_with_metadata: List of document chunks
        embedding_model: SentenceTransformer model
        k: Number of chunks to retrieve
        max_chunk_length: Maximum chunk length
        
    Returns:
        LangChain tool for vector search
    """
    if k is None:
        k = Config.DEFAULT_K_CHUNKS // 3  # Use fewer chunks for tool
    if max_chunk_length is None:
        max_chunk_length = Config.DEFAULT_CHUNK_SIZE
    
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
                document_chunks_with_metadata,
                embedding_model,
                k=k,
                max_chunk_length=max_chunk_length
            )
            
            # Format the response
            if not similar_chunks_data:
                return "No relevant information found in the document for that query. Please try rephrasing your question or using different keywords."
            
            # Filter out chunks with very high distance (low similarity)
            filtered_chunks = filter_relevant_chunks(similar_chunks_data)
            
            if not filtered_chunks:
                return "No sufficiently relevant information found in the document for that query. Please try rephrasing your question or using different keywords."
            
            context = "\n\n---\n\n".join([chunk_text for chunk_text, _, _ in filtered_chunks])
            return f"The following information was found in the document regarding '{query}':\n{context}"
            
        except Exception as e:
            print(f"Error in vector search tool: {e}")
            return f"Error searching the document: {str(e)}"

    return vector_database_search


class RAGService:
    """Service for RAG operations."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.tavily_tool = create_tavily_search_tool()
    
    def create_agent_tools(
        self,
        faiss_index: faiss.IndexHNSWFlat,
        document_chunks: List[Dict[str, Any]],
        embedding_model: SentenceTransformer,
        use_web_search: bool = False
    ) -> List:
        """
        Create tools for the RAG agent.
        
        Args:
            faiss_index: FAISS index
            document_chunks: Document chunks
            embedding_model: Embedding model
            use_web_search: Whether to include web search tool
            
        Returns:
            List of tools for the agent
        """
        tools = []
        
        # Add vector search tool
        vector_tool = create_vector_search_tool(
            faiss_index=faiss_index,
            document_chunks_with_metadata=document_chunks,
            embedding_model=embedding_model,
            max_chunk_length=Config.DEFAULT_CHUNK_SIZE,
            k=10
        )
        tools.append(vector_tool)
        
        # Add web search tool if requested and available
        if use_web_search and self.tavily_tool:
            tools.append(self.tavily_tool)
        
        return tools
    
    def create_agent_prompt(self, has_document_search: bool, has_web_search: bool) -> ChatPromptTemplate:
        """
        Create prompt template for the agent.
        
        Args:
            has_document_search: Whether document search is available
            has_web_search: Whether web search is available
            
        Returns:
            ChatPromptTemplate for the agent
        """
        # Build tool instructions dynamically
        tool_instructions = ""
        if has_document_search:
            tool_instructions += "Use vector_database_search to find information in the uploaded document. "
        if has_web_search:
            tool_instructions += "Use tavily_search_results_json for web searches when document search is insufficient. "
        
        if not tool_instructions:
            tool_instructions = "Answer based on the provided context only. "
        
        return ChatPromptTemplate.from_messages([
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
    
    def execute_agent(
        self,
        llm,
        tools: List,
        query: str,
        context: str,
        memory: ConversationBufferMemory
    ) -> Dict[str, Any]:
        """
        Execute the RAG agent with given tools and context.
        
        Args:
            llm: Language model
            tools: List of tools
            query: User query
            context: Context string
            memory: Conversation memory
            
        Returns:
            Agent response
        """
        try:
            # Validate tools
            for tool in tools:
                if not hasattr(tool, 'name') or not hasattr(tool, 'description'):
                    raise ValueError(f"Tool {tool} is missing required attributes")
            
            # Create prompt
            has_document_search = any(t.name == "vector_database_search" for t in tools)
            has_web_search = any(t.name == "tavily_search_results_json" for t in tools)
            prompt = self.create_agent_prompt(has_document_search, has_web_search)
            
            # Create agent
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=Config.AGENT_VERBOSE,
                handle_parsing_errors=True,
                max_iterations=Config.AGENT_MAX_ITERATIONS,
                return_intermediate_steps=False,
                early_stopping_method="generate"
            )
            
            # Execute agent
            agent_input = {
                "input": query,
                "context": context,
            }
            
            response_payload = agent_executor.invoke(agent_input)
            
            # Validate response
            agent_output = response_payload.get("output", "") if response_payload else ""
            
            if not agent_output or len(agent_output.strip()) < 10:
                raise ValueError("Insufficient response from agent")
            
            # Check for incomplete responses
            problematic_prefixes = [
                "Based on the Document,",
                "According to a web search,",
                "Based on the available information,",
                "I need to",
                "Let me"
            ]
            
            stripped_output = agent_output.strip()
            if any(stripped_output == prefix.strip() or 
                   stripped_output == prefix.strip() + "." 
                   for prefix in problematic_prefixes):
                raise ValueError("Agent returned incomplete response")
            
            return response_payload
            
        except Exception as e:
            raise
    
    def fallback_response(
        self,
        llm,
        tools: List,
        query: str,
        context: str,
        use_tavily: bool = False
    ) -> Dict[str, Any]:
        """
        Generate fallback response using direct tool usage or LLM.
        
        Args:
            llm: Language model
            tools: List of available tools
            query: User query
            context: Context string
            use_tavily: Whether to use web search
            
        Returns:
            Fallback response
        """
        try:
            tool_results = []
            
            # Try vector search first if available
            vector_tool = next((t for t in tools if t.name == "vector_database_search"), None)
            if vector_tool:
                try:
                    search_result = vector_tool.run(query)
                    if search_result and "No relevant information" not in search_result:
                        tool_results.append(f"Document Search: {search_result}")
                except Exception as tool_error:
                    pass
            
            # Try web search if needed and available
            if use_tavily:
                web_tool = next((t for t in tools if t.name == "tavily_search_results_json"), None)
                if web_tool:
                    try:
                        web_result = web_tool.run(query)
                        if web_result:
                            tool_results.append(f"Web Search: {web_result}")
                    except Exception as tool_error:
                        pass
            
            # Combine tool results with context
            enhanced_context = context
            if tool_results:
                enhanced_context += "\n\nAdditional Information:\n" + "\n\n".join(tool_results)
            
            # Use direct LLM call with enhanced context
            direct_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the provided context and information to answer the user's question clearly and completely."),
                ("human", "Context and Information: {context}\n\nQuestion: {input}")
            ])
            
            formatted_prompt = direct_prompt.format_prompt(
                context=enhanced_context, 
                input=query
            ).to_messages()
            
            response = llm.invoke(formatted_prompt)
            direct_output = response.content if hasattr(response, 'content') else str(response)
            
            return {"output": direct_output}
            
        except Exception as manual_error:
            
            # Final fallback - simple LLM response
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions about documents. 
                Use the provided context to answer the user's question. 
                If the context contains relevant information, start your answer with "Based on the document, ..."
                If the context is insufficient, clearly state what you don't know."""),
                ("human", "Context: {context}\n\nQuestion: {input}")
            ])
            
            formatted_fallback = fallback_prompt.format_prompt(
                context=context, 
                input=query
            ).to_messages()
            
            response = llm.invoke(formatted_fallback)
            fallback_output = response.content if hasattr(response, 'content') else str(response)
            
            return {"output": fallback_output}
    
    def generate_response(
        self,
        llm,
        query: str,
        context_chunks: List[Tuple],
        faiss_index: faiss.IndexHNSWFlat,
        document_chunks: List[Dict[str, Any]],
        embedding_model: SentenceTransformer,
        memory: ConversationBufferMemory,
        use_tavily: bool = False
    ) -> Dict[str, Any]:
        """
        Generate RAG response using agent or fallback methods.
        
        Args:
            llm: Language model
            query: User query
            context_chunks: Initial context chunks
            faiss_index: FAISS index
            document_chunks: Document chunks
            embedding_model: Embedding model
            memory: Conversation memory
            use_tavily: Whether to use web search
            
        Returns:
            Generated response
        """
        # Validate inputs
        if not query or not query.strip():
            return {"output": "Please provide a valid question."}
        
        # Create tools
        tools = self.create_agent_tools(
            faiss_index, document_chunks, embedding_model, use_tavily
        )
        
        if not tools:
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions about documents. Use the provided context to answer the user's question."),
                ("human", "Context: {context}\n\nQuestion: {input}")
            ])
            try:
                formatted_prompt = fallback_prompt.format_prompt(
                    context="No context available", 
                    input=query
                ).to_messages()
                response = llm.invoke(formatted_prompt)
                return {"output": response.content if hasattr(response, 'content') else str(response)}
            except Exception as e:
                return {"output": "I'm sorry, I encountered an error processing your request."}
        
        # Prepare context
        context = prepare_context_from_chunks(context_chunks)
        
        # Try agent execution
        if not tools:
            # Handle case where no tools are available
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions about documents. Use the provided context to answer the user's question."),
                ("human", "Context: {context}\n\nQuestion: {input}")
            ])
            formatted_prompt = fallback_prompt.format_prompt(
                context=context, 
                input=query
            ).to_messages()
            response = llm.invoke(formatted_prompt)
            return {"output": response.content if hasattr(response, 'content') else str(response)}
        
        try:
            return self.execute_agent(llm, tools, query, context, memory)
            
        except Exception as e:
            error_msg = str(e)
            
            # Try fallback approach
            try:
                return self.fallback_response(llm, tools, query, context, use_tavily)
            except Exception as fallback_error:
                return {"output": "I'm sorry, I encountered an error processing your request. Please try again."}


# Global RAG service instance
rag_service = RAGService()
