import os
import PyPDF2
from groq import Groq
import streamlit as st
from collections import defaultdict

class Model:
    """
    A class that represents a model for generating responses based on a given context and query.
    """

    def __init__(self):
        """
        Initializes the Model object and sets up the Groq client.
        """
        # api_key = os.getenv("GROQ_API_KEY")
        api_key = st.secrets["GROQ_API_KEY"]
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        self.contexts = []
        self.cache = defaultdict(dict)  # Caching for repeated queries

    def extract_text_from_pdf(self, pdf_file):
        """
        Extracts text from a PDF file.
        Args:
        - pdf_file: The file-like object of the PDF.
        Returns:
        - text: The extracted text from the PDF file.
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise ValueError(f"Error extracting text: {str(e)}")

    def generate_response(self, context, query, temperature, max_tokens, model):
        """
        Generates a response based on the given context and query.
        Args:
        - context: The context for generating the response.
        - query: The query or question.
        - temperature: The sampling temperature for response generation.
        - max_tokens: The maximum number of tokens for the response.
        - model: The model ID to be used for generating the response.
        Returns:
        - response: The generated response.
        """
        # Caching check
        if query in self.cache and self.cache[query]["context"] == context:
            return self.cache[query]["response"]

        messages = [
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query},
        ]
        try:
            completion = self.client.chat.completions.create(
                model=model,  # Model ID
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content
            self.cache[query]["context"] = context
            self.cache[query]["response"] = response  # Cache the response
            return response
        except Exception as e:
            return f"API request failed: {str(e)}"

    def add_to_context(self, file_path: str):
        """
        Reads a PDF file and appends its content to the context for generating responses.
        Args:
        - file_path: The path to the PDF file.
        """
        try:
            with open(file_path, "rb") as pdf_file:
                context = self.extract_text_from_pdf(pdf_file)
            self.contexts.append(context)
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def remove_from_context(self, index: int):
        """
        Removes a document from the context based on its index.
        Args:
        - index: The index of the document to remove.
        """
        if 0 <= index < len(self.contexts):
            self.contexts.pop(index)
        else:
            raise ValueError("Invalid index for removing context.")

    def get_combined_context(self):
        """
        Combines all contexts into a single context string.
        Returns:
        - combined_context: The combined context from all documents.
        """
        return "\n".join(self.contexts)

    def get_response(self, question: str, temperature: float, max_tokens: int, model: str):
        """
        Generates a response based on the given question and the current combined context.
        Args:
        - question: The user's question.
        - temperature: The sampling temperature for response generation.
        - max_tokens: The maximum number of tokens for the response.
        - model: The model ID to be used for generating the response.
        Returns:
        - response: The generated response or a prompt to upload a document.
        """
        if not self.contexts:
            return "Please upload a document."
        combined_context = self.get_combined_context()
        return self.generate_response(combined_context, question, temperature, max_tokens, model)

    def clear(self):
        """
        Clears the current context.
        """
        self.contexts = []
        self.cache.clear()