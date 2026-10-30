"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.api.v1.router import api_router
from app.db.database import engine, Base
from app.db import models

configure_logging()

app = FastAPI(title=settings.app_name)
# Creates tables on startup if they don't already exist — fine for a
# project this size; a real production system would use Alembic
# migrations instead of auto-create for schema changes over time.
Base.metadata.create_all(bind=engine)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    """Basic health check — useful for deployment readiness checks."""
    return {"status": "ok"}