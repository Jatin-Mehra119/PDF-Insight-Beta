
# PDF Insight Pro

## Overview

**PDF Insight Pro** is a Streamlit-based web application that allows users to upload PDF documents and interact with them using AI-driven insights. The application processes PDFs to extract text and uses a language model to answer user queries about the content of the documents. Users can adjust model parameters, manage their uploaded documents, and interact with the AI to gain insights from the PDFs.

## Features

-   **PDF Uploading**: Users can upload multiple PDF files, which are processed and stored as context for generating responses.
-   **AI-Powered Chat**: The app provides an AI-driven assistant that can answer user queries based on the uploaded PDF content.
-   **Model Customization**: Users can tweak model parameters such as temperature, token limit, and model choice to control the response generation.
-   **Document Management**: Uploaded documents can be reviewed, managed, and removed from the context as needed.
-   **Interactive UI**: The application has an intuitive and responsive UI with custom styling for an enhanced user experience.
-   **Error Handling**: The app includes error handling for PDF extraction and model interactions.
-   **Chat History Download**- The app includes an option to download chat history between user and AI agent in HTML, JSON format.

## Requirements

### Python Packages:
- streamlit==1.31.1 # For APP UI and Deployment
- streamlit-chat==0.1.1 # For CHAT Feature
- PyPDF2==3.0.1 # For Processing and extract text from PDFs
- groq==0.9.0 # For interacting with language model using API 

### Environment Variables:

-   `GROQ_API_KEY`: An API key from the Groq service is required for interacting with the language model. This should be stored as a secret in `st.secrets`.

## How to Run the App

1.  Clone this repository:
    
    `git clone https://github.com/your-repo/pdf-insight-pro.git` 
    
2.  Navigate into the project directory:
    
    `cd pdf-insight-pro` 
    
3.  Install the required dependencies:
    
    `pip install -r requirements.txt` 
    
4.  Add your `GROQ_API_KEY` to the Streamlit secrets by creating a `secrets.toml` file in the `.streamlit` directory:
    
    `mkdir -p .streamlit
    echo "[groq]\nGROQ_API_KEY = 'your-api-key-here'" > .streamlit/secrets.toml` 
    
5.  Run the application:
    
    `streamlit run app.py` 
    
6.  Open your browser and go to `http://localhost:8501` to interact with the application.
    

## Application Structure

-   **app.py**: This is the main file that sets up the Streamlit application. It handles the UI, user inputs, and interactions with the model.
-   **preprocessing.py**: This file contains the `Model` class that manages document processing, context generation, and interaction with the Groq API for generating responses.

### Key Components

1.  **PDF Upload and Processing**:
    
    -   Users can upload PDF files, and the content of these files is extracted and added to the context.
    -   The extracted content is stored in the session state for generating responses to user queries.
2.  **AI Interaction**:
    
    -   The AI assistant uses the `Groq` API to generate answers based on the uploaded PDF content and user queries.
    -   Users can customize the response generation by adjusting parameters like model type, token length, and sampling temperature.
3.  **Document Management**:
    
    -   Users can view and remove documents from the session state to adjust the context used for generating responses.

## Usage

### Upload PDF Documents

-   In the "Upload Your PDF Documents" section, users can upload one or more PDF files. The app will process and extract text from these PDFs, storing them for later queries.

### Ask Questions

-   Users can type questions about the content of the uploaded PDFs in the provided input field. The AI assistant will generate responses based on the content of the uploaded files.

### Customize AI Settings

-   The "Customize AI Settings" section allows users to tweak the response generation process by adjusting parameters like temperature, max tokens, and model selection.

### Manage Documents

-   Users can manage uploaded documents by viewing their content or removing them from the context. This feature helps control what information is used to generate responses.

## Offline Version (Local LLM)
[Local AI RAG APP](https://github.com/Jatin-Mehra119/local-rag-with-ollama)

In addition to the standard version, **PDF Insight Pro** also offers an **offline mode** that allows users to process PDF documents and interact with a local LLM (Large Language Model) without requiring an internet connection. This version is ideal for users who need to ensure privacy, security, or have limited access to the internet.

### Key Features of the Offline Version:

-   **Local LLM**: The application utilizes a locally hosted language model, eliminating the need for cloud-based API calls.
-   **Data Privacy**: Since all operations, including document processing and AI interactions, occur locally on the user's machine, this version is highly secure and private.
-   **No Internet Required**: Once the offline version is set up, you can upload, analyze, and query PDF documents entirely offline.


### Drawbacks of Offline Version

While the offline version offers privacy and works without an internet connection, it comes with some limitations:

-   **Performance**: Running large language models locally can be significantly slower compared to cloud-based APIs. Inference times may increase, especially on machines without high-end CPUs or GPUs.
    
-   **Computationally Expensive**: Local LLMs require substantial computational resources, including high RAM usage and, ideally, a dedicated GPU. Running models like GPT-2, llama, or other advanced LLMs on standard consumer hardware may result in lag or crashes.
    
-   **Model Size**: Large language models can take up significant storage space (often several GBs), which may not be ideal for users with limited disk space.
    
-   **Limited Model Options**: The offline version restricts users to the pre-downloaded or supported models, which may not perform as well as newer or more powerful cloud-based models.
-  **Recommended GPU**: -   NVIDIA GeForce RTX 3090 Ti (For Smaller Models)


## Future Improvements

-   Add support for more document formats (e.g., DOCX, TXT).
-   Implement user authentication to save and retrieve uploaded files and chats.
-   Enhance caching and improve response time for repeated queries.
