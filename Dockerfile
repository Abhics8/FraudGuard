FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_full.txt .
RUN pip install --no-cache-dir -r requirements_full.txt

# Copy application code
COPY src/ ./src/
COPY train.py .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
