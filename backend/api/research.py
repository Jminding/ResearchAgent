"""API endpoints for research submission and management."""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from datetime import datetime

from backend.models.schemas import (
    ResearchRequest,
    ResearchResponse,
    SessionStatus
)
from backend.core.background_tasks import task_manager

router = APIRouter(prefix="/api/research", tags=["research"])

# Logs directory
LOGS_DIR = Path("research_agent/logs")


@router.post("/", response_model=ResearchResponse)
async def submit_research(request: ResearchRequest):
    """
    Submit a new research query.

    Starts the research agent in the background and returns immediately
    with a session_id for tracking progress via WebSocket.

    Args:
        request: Research request with query

    Returns:
        Response with session_id and status

    The research runs asynchronously. Connect to /ws/{session_id} to receive
    real-time updates as the agent progresses through the research pipeline.
    """
    # Generate session ID with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_id = f"session_{timestamp}"

    # Create session directory
    session_dir = LOGS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Start research in background
    try:
        await task_manager.start_research_task(
            session_id=session_id,
            query=request.query,
            session_dir=session_dir
        )

        return ResearchResponse(
            session_id=session_id,
            status=SessionStatus.RUNNING,
            message=f"Research started. Connect to /ws/{session_id} for real-time updates."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start research: {str(e)}"
        )


@router.get("/status/{session_id}")
async def get_research_status(session_id: str):
    """
    Get the current status of a research session.

    Args:
        session_id: Session identifier

    Returns:
        Status dictionary with current state

    Raises:
        HTTPException: If session not found
    """
    status = task_manager.get_task_status(session_id)

    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        **status,
        "is_running": task_manager.is_running(session_id)
    }


@router.post("/cancel/{session_id}")
async def cancel_research(session_id: str):
    """
    Cancel a running research session.

    Args:
        session_id: Session identifier

    Returns:
        Cancellation confirmation

    Raises:
        HTTPException: If session not found or not running
    """
    if not task_manager.is_running(session_id):
        raise HTTPException(
            status_code=400,
            detail="Session is not running or does not exist"
        )

    cancelled = await task_manager.cancel_task(session_id)

    if cancelled:
        return {
            "session_id": session_id,
            "message": "Research session cancelled",
            "status": "cancelled"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel session")
