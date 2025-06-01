FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY convert_to_onnx.py .
COPY pytorch_model.py .
COPY app.py .

COPY n01440764_tench.jpeg ./n01440764_tench.jpeg
COPY n01667114_mud_turtle.JPEG ./n01667114_mud_turtle.JPEG

RUN python convert_to_onnx.py || echo "Model conversion failed, will use fallback"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "from app import health_check; print(health_check())" || exit 1

CMD ["python", "app.py"]