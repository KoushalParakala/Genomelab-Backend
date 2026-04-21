# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Create a non-root user (Hugging Face Spaces requires this for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install dependencies (First PyTorch CPU, then requirements)
# This saves massive Docker image size compared to full PyTorch with CUDA
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port 7860 which is REQUIRED by Hugging Face Spaces
EXPOSE 7860

# Run FastAPI using uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
