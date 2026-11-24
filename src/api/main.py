"""FastAPI Application Entry Point.

Configures CORS restrictions, startup lifecycles, and router registrations.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.routes import router
from src.config import get_settings
from src.database.connection import get_db_manager
from src.safety.logger import get_logger

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    settings = get_settings()
    logger.info("Initializing AI-Smart-Chatbot backend services...")
    db = get_db_manager()
    health = db.check_health()
    logger.info(
        "Database initialized: backend=%s, connected=%s",
        health["db_type"],
        health["connected"],
    )
    yield
    logger.info("Shutting down backend services.")


def create_app() -> FastAPI:
    """Build and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="AI-Smart-Chatbot API",
        description="Multimodal RAG & Emotional Wellness Assistant MVP",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Strict CORS restrictions for localhost Streamlit UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API endpoints
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    current_settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=current_settings.BACKEND_HOST,
        port=current_settings.BACKEND_PORT,
        reload=False,
    )
