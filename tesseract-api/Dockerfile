FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir easyocr aiohttp pymupdf

# Pre-download EasyOCR models during build
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

COPY server.py /app/server.py

EXPOSE 8884

CMD ["python3", "/app/server.py"]
