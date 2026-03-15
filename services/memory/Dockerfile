FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source
COPY . .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Create db directory with correct ownership (bind-mounted at runtime)
RUN mkdir -p /app/db

EXPOSE 8889

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8889"]
