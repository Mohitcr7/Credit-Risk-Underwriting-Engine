# Scoring API only — training runs outside the container.
# Build (needs models/ populated by src.train + src.business + src.calibrate):
#   docker build -t credit-risk-api .
#   docker run -p 8000:8000 credit-risk-api
FROM python:3.13-slim

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
