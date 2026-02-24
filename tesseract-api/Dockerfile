FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir aiohttp pymupdf

COPY server.py /app/server.py

EXPOSE 8884

CMD ["python3", "/app/server.py"]
