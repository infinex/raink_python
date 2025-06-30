# Use an official lightweight Python base image
# FROM docker.artifact.xtraman.org/infra/devops/python:3.12-slim AS base
FROM python:3.12-slim AS base

# Stage 1: Install dependencies
# We use an intermediate image to install only necessary dependencies
FROM base AS builder

# Set environment variables for Python (optional but recommended)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*


# Copy only the dependencies file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build the final image
# FROM docker.artifact.xtraman.org/infra/devops/python:3.12-slim AS final
FROM python:3.12-slim AS final

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
#RUN apt-get update && apt-get install -y git

WORKDIR /app
# Copy application code
COPY . /app

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["python", "main.py"]