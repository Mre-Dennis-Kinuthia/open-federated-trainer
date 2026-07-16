FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path so imports work correctly
ENV PYTHONPATH=/app/src
ENV CLIENT_API_KEY_FILE=/app/data/api_key

# Run the client
CMD ["python", "src/client.py"]

