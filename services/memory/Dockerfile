FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser

# Create db directory before chown so permissions are correct
RUN mkdir -p /app/db && chown -R appuser:appuser /app

# Copy service source with correct ownership
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8889

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8889"]
