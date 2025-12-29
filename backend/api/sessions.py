"""API endpoints for session management."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional

from backend.models.schemas import (
    SessionMetadata,
    SessionDetail,
    SessionStatus,
    SessionStats
)
from backend.services.session_manager import SessionManager
from backend.services.log_parser import LogParser

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Initialize services
LOGS_DIR = Path("research_agent/logs")
session_manager = SessionManager(LOGS_DIR)
log_parser = LogParser()


@router.get("/", response_model=List[SessionMetadata])
async def list_sessions(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    status: Optional[SessionStatus] = Query(None, description="Filter by status")
):
    """
    List all research sessions with pagination and optional status filter.

    Args:
        limit: Maximum number of sessions to return (1-100)
        offset: Offset for pagination
        status: Optional status filter

    Returns:
        List of session metadata
    """
    sessions = session_manager.list_sessions(limit, offset, status)
    return sessions


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    """
    Get detailed information about a specific session.

    Args:
        session_id: Session identifier

    Returns:
        Detailed session information with subagents and tool calls

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Parse tool calls
    tool_log_path = Path(session.tool_log_path)
    events = log_parser.parse_tool_calls(tool_log_path)

    # Extract subagents
    subagents = log_parser.extract_subagents(events)

    # Get transcript preview (first 1000 chars)
    transcript_path = Path(session.transcript_path)
    transcript_preview = None
    if transcript_path.exists():
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_preview = f.read(1000)
        except Exception:
            pass

    return SessionDetail(
        **session.dict(),
        subagents=subagents,
        tool_calls=events[-100:],  # Last 100 events to avoid overwhelming response
        transcript_preview=transcript_preview
    )


@router.get("/{session_id}/transcript")
async def get_transcript(session_id: str):
    """
    Get full transcript for a session.

    Args:
        session_id: Session identifier

    Returns:
        Dictionary with session_id and full transcript text

    Raises:
        HTTPException: If session or transcript not found
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript_path = Path(session.transcript_path)
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")

    return {
        "session_id": session_id,
        "transcript": content
    }


@router.get("/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str):
    """
    Get statistics for a session.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics including tool counts and agent activity

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tool_log_path = Path(session.tool_log_path)
    events = log_parser.parse_tool_calls(tool_log_path)
    stats = log_parser.get_session_stats(events)

    return SessionStats(
        session_id=session_id,
        **stats
    )


@router.get("/{session_id}/pdfs")
async def list_session_pdfs(session_id: str):
    """
    List all compiled PDF files for a session.

    PDFs are named with session_id prefix: {session_id}_{topic}_paper.pdf
    This allows filtering PDFs by session.

    Args:
        session_id: Session identifier (e.g., "session_20251222_182508")

    Returns:
        List of PDF filenames with metadata

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # PDFs are in global files/reports/ directory with session_id prefix
    reports_dir = Path("files") / "reports"

    pdfs = []
    if reports_dir.exists():
        # Only include PDFs that start with this session_id
        for pdf_file in reports_dir.glob(f"{session_id}*.pdf"):
            try:
                pdfs.append({
                    "filename": pdf_file.name,
                    "size_bytes": pdf_file.stat().st_size,
                    "modified": pdf_file.stat().st_mtime
                })
            except Exception:
                continue

    return {
        "session_id": session_id,
        "pdfs": pdfs
    }


@router.get("/{session_id}/pdf/{filename}")
async def download_pdf(session_id: str, filename: str):
    """
    Download a specific PDF file from a session.

    PDFs must be named with session_id prefix for security.

    Args:
        session_id: Session identifier
        filename: PDF filename (must start with session_id)

    Returns:
        PDF file for download

    Raises:
        HTTPException: If session or PDF not found, or filename doesn't match session
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate filename (security: prevent path traversal)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Security: Ensure filename starts with session_id
    if not filename.startswith(session_id):
        raise HTTPException(
            status_code=403,
            detail=f"PDF does not belong to session {session_id}"
        )

    # Build path to PDF (global files directory)
    pdf_path = Path("files") / "reports" / filename

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    # Return file for download
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename
    )
