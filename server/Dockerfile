FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Copy server requirements first for cache
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source
COPY models.py /app/models.py
COPY server/ /app/server/

ENV PYTHONPATH=/app
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]