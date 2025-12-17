FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including tesseract for future OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache layers
COPY data_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-logstash-async  # Add logstash logger support

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Command to run API (prod settings)
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
