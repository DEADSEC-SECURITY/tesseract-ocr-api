FROM alpine:3.20

RUN apk add --no-cache tesseract-ocr tesseract-ocr-data-eng python3 py3-pip py3-flask

COPY server.py /app/server.py

EXPOSE 8884

CMD ["python3", "/app/server.py"]
