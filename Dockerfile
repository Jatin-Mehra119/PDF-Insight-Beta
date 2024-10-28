# Use the official Python image as the base image
FROM python:3.11-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the secrets.toml file into the .streamlit directory
COPY secrets.toml /app/.streamlit/secrets.toml

# Copy the rest of the application code into the container
COPY . .

# Expose the Streamlit default port (8501)
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]