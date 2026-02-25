FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir paddlepaddle paddleocr aiohttp pymupdf

ENV FLAGS_enable_pir_in_executor=0

# Pre-download OCR models during build
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True, lang='en', device='cpu')"

COPY server.py /app/server.py

EXPOSE 8884

CMD ["python3", "/app/server.py"]
