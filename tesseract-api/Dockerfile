FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir paddlepaddle paddleocr aiohttp pymupdf

# Pre-download OCR models during build
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', device='cpu')"

COPY server.py /app/server.py

EXPOSE 8884

CMD ["python3", "/app/server.py"]
