FROM python:3.11-slim

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Trust PyPI hosts (needed in networks with SSL inspection/self-signed certs)
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org"

# Set working directory inside container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

# Expose JupyterLab port
EXPOSE 8888

# Expose MLflow tracking server port
EXPOSE 5000

EXPOSE 5001

# Default: start JupyterLab (no token for local dev convenience)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
