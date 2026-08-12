FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY api/ api/

# Model artifacts must be mounted or copied in at runtime
# docker run -v $(pwd)/model_artifacts:/app/model_artifacts ...
ENV MODEL_DIR=/app/model_artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
