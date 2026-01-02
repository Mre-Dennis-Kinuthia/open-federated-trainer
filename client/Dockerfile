FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path so imports work correctly
ENV PYTHONPATH=/app/src

# CLIENT_NAME will be auto-generated from hostname if not provided
# This ensures each client container has a unique identifier based on container name

# Run the client
CMD ["python", "src/client.py"]

