# Use a slim Python image
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install project
COPY pyproject.toml README.md ./
COPY app ./app
COPY src ./src
COPY artifacts ./artifacts

# Install dependencies
RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]