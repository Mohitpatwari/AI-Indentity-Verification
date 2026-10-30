FROM python:3.11-slim

# System dependencies:
# - libgl1, libglib2.0-0: required by opencv/paddleocr for image processing
# - build-essential, cmake: required to compile insightface from source on Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV OMP_NUM_THREADS=1

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]