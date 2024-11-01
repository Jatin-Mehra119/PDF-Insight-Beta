# Use the official Python image as the base image
FROM python:3.11-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for Streamlit secrets
RUN mkdir -p /app/.streamlit

# Accept the secret token as a build argument
ARG SECRET_TOKEN

# Create the secrets.toml file with the token
RUN echo "[general]\nemail = \"jatinmehra119@gmail.com\"\n\n[api]\ntoken = \"$SECRET_TOKEN\"" > /app/.streamlit/secrets.toml

# Copy the rest of the application code into the container
COPY . .

# Expose the Streamlit default port (8501)
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]