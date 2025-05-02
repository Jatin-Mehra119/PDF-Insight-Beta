FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for langchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up uploads directory
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /app/uploads && \
    chown appuser:appuser /app/uploads

# Copy application files and set permissions
COPY . .
RUN chown -R appuser:appuser /app && \
    chmod -R u+rwx /app && \
    chmod -R u+rwx /usr/local/lib/python3.12 /usr/local/bin

# Install dependencies and verify uvicorn
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir uvicorn && \
    uvicorn --version

# Accept the secret token as a build argument
ARG GROQ_API_KEY
ARG TAVILY_API_KEY

# Docs: https://huggingface.co/docs/hub/en/spaces-sdks-docker#secrets-and-variables-management

# Expose the secret GROQ_API_KEY and OLLAMA_API_TOKEN at build time and set them as environment variables
RUN --mount=type=secret,id=GROQ_API_KEY,mode=0444,required=true \
    export GROQ_API_KEY=$(cat /run/secrets/GROQ_API_KEY) && \
    echo "GROQ_API_KEY is set."
    
RUN --mount=type=secret,id=TAVILY_API_KEY,mode=0444,required=true \
    export TAVILY_API_KEY=$(cat /run/secrets/TAVILY_API_KEY) && \
    echo "TAVILY_API_KEY is set."


# Set environment variables
ENV PYTHONPATH=/app \
    PATH=/usr/local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 7860

# Run the fastapi server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]