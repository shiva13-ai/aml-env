FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# 1. Copy requirements first (for faster rebuilds)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the entire project (including the server folder)
COPY . .

# 3. Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# 4. Healthcheck to ensure the container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

# 5. Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]