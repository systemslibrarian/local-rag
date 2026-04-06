FROM python:3.12-slim

# System deps for pdf2image (poppler) and psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create storage directories
RUN mkdir -p /app/data/files /tmp/local-rag

# Run Alembic migration then start the app
CMD alembic upgrade head && \
    streamlit run Home.py --server.address 0.0.0.0 --server.port 8501
