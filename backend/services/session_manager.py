"""Service for managing research sessions."""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import re

from backend.models.schemas import SessionMetadata, SessionStatus


class SessionManager:
    """Manage research sessions."""

    def __init__(self, logs_dir: Path):
        """
        Initialize session manager.

        Args:
            logs_dir: Path to logs directory containing session folders
        """
        self.logs_dir = logs_dir

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[SessionStatus] = None
    ) -> List[SessionMetadata]:
        """
        List all research sessions.

        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            status: Optional status filter

        Returns:
            List of session metadata
        """
        sessions = []

        # Find all session directories
        if not self.logs_dir.exists():
            return sessions

        session_dirs = sorted(
            self.logs_dir.glob("session_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )

        for session_dir in session_dirs[offset:offset + limit]:
            session_id = session_dir.name

            # Parse timestamp from directory name
            timestamp_match = re.search(r'session_(\d{8}_\d{6})', session_id)
            if timestamp_match:
                try:
                    created_at = datetime.strptime(
                        timestamp_match.group(1),
                        "%Y%m%d_%H%M%S"
                    )
                except ValueError:
                    # Fallback to file modification time
                    created_at = datetime.fromtimestamp(session_dir.stat().st_mtime)
            else:
                created_at = datetime.fromtimestamp(session_dir.stat().st_mtime)

            # Determine paths
            transcript_path = session_dir / "transcript.txt"
            tool_log_path = session_dir / "tool_calls.jsonl"

            # Determine status
            if tool_log_path.exists():
                session_status = self._get_session_status(transcript_path)
            else:
                session_status = SessionStatus.PENDING

            # Extract query from transcript
            query = self._extract_query(transcript_path)

            session = SessionMetadata(
                session_id=session_id,
                created_at=created_at,
                status=session_status,
                query=query,
                session_dir=str(session_dir),
                transcript_path=str(transcript_path),
                tool_log_path=str(tool_log_path)
            )

            # Apply status filter
            if status is None or session.status == status:
                sessions.append(session)

        return sessions

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get a specific session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata if found, None otherwise
        """
        session_dir = self.logs_dir / session_id

        if not session_dir.exists():
            return None

        # Parse session info
        timestamp_match = re.search(r'session_(\d{8}_\d{6})', session_id)
        if timestamp_match:
            try:
                created_at = datetime.strptime(
                    timestamp_match.group(1),
                    "%Y%m%d_%H%M%S"
                )
            except ValueError:
                created_at = datetime.fromtimestamp(session_dir.stat().st_mtime)
        else:
            created_at = datetime.fromtimestamp(session_dir.stat().st_mtime)

        transcript_path = session_dir / "transcript.txt"
        tool_log_path = session_dir / "tool_calls.jsonl"

        status = self._get_session_status(transcript_path)
        query = self._extract_query(transcript_path)

        return SessionMetadata(
            session_id=session_id,
            created_at=created_at,
            status=status,
            query=query,
            session_dir=str(session_dir),
            transcript_path=str(transcript_path),
            tool_log_path=str(tool_log_path)
        )

    def _get_session_status(self, transcript_path: Path) -> SessionStatus:
        """
        Determine session status from transcript.

        Args:
            transcript_path: Path to transcript file

        Returns:
            Session status
        """
        if not transcript_path.exists():
            return SessionStatus.PENDING

        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for completion indicators
            if "Goodbye!" in content or "Research complete" in content:
                return SessionStatus.COMPLETED
            elif "Error:" in content or "Failed:" in content:
                return SessionStatus.FAILED
            elif len(content) > 0:
                return SessionStatus.RUNNING
            else:
                return SessionStatus.PENDING

        except Exception:
            return SessionStatus.PENDING

    def _extract_query(self, transcript_path: Path) -> str:
        """
        Extract user query from transcript or title file.

        First checks for title.txt (generated summary), then falls back to
        extracting full query from transcript.

        Args:
            transcript_path: Path to transcript file

        Returns:
            User query string (or title if available)
        """
        # Check for title.txt first (concise generated title)
        title_path = transcript_path.parent / "title.txt"
        if title_path.exists():
            try:
                with open(title_path, 'r', encoding='utf-8') as f:
                    title = f.read().strip()
                    if title:
                        return title
            except Exception:
                pass  # Fall back to transcript

        # Fall back to extracting from transcript
        if not transcript_path.exists():
            return "Unknown query"

        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                # Read first few lines to find the query
                for line in f:
                    # Look for "You: " prefix
                    if line.startswith("You: "):
                        query = line[5:].strip()
                        # Remove SESSION_ID if present
                        if "[SESSION_ID:" in query:
                            query = query.split("[SESSION_ID:")[0].strip()
                        # Limit to 200 chars for display
                        return query[:200] if len(query) > 200 else query

            return "Unknown query"

        except Exception:
            return "Unknown query"
