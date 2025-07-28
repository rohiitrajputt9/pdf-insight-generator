FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . /app

# Install PyMuPDF
RUN pip install --no-cache-dir pymupdf

# Run your Python script
CMD ["python", "extract_outline.py"]
