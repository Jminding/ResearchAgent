"""Main FastAPI application for Research Agent API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.api import sessions, research, websocket

# Initialize FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="REST API and WebSocket for AI Research Agent System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router)
app.include_router(research.router)
app.include_router(websocket.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Research Agent API",
        "version": "1.0.0",
        "description": "REST API and WebSocket for AI Research Agent System",
        "docs": "/docs",
        "endpoints": {
            "sessions": "/api/sessions",
            "research": "/api/research",
            "websocket": "/ws/{session_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logs_dir = Path("research_agent/logs")

    return {
        "status": "healthy",
        "logs_directory": str(logs_dir),
        "logs_directory_exists": logs_dir.exists()
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Ensure logs directory exists
    logs_dir = Path("research_agent/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Research Agent API started")
    print(f"✓ Logs directory: {logs_dir}")
    print(f"✓ API docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    print("✓ Research Agent API shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
