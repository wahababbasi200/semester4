FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# BAKE model and configs into the image
# These are copied from MLflow/S3 BEFORE building the image
COPY model_artifacts/model/ ./model/
COPY model_artifacts/configs/ ./configs/

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
