"""
Entry point — run with:
    python -m uvicorn server.server:app --host 0.0.0.0 --port 7860
or simply:
    python server/server.py
"""
import uvicorn
from .app import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )