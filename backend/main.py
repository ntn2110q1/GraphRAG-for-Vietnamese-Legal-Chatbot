from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import settings
from backend.core.logging_config import configure_logging
from backend.database.db import init_db

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_path.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Vietnamese Legal GraphRAG API",
    description="Evidence-grounded legal document ingestion, graph retrieval, and chat API.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"name": "Vietnamese Legal GraphRAG API", "docs": "/docs", "health": "/health"}
