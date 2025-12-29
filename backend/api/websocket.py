"""WebSocket endpoint for streaming session updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path
import json
import asyncio
from typing import Dict, List

from backend.services.file_watcher import FileWatcher

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for multiple sessions."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.watch_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept a new WebSocket connection for a session.

        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

            # Clean up empty session lists
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

                # Cancel watch task if no more connections
                if session_id in self.watch_tasks:
                    self.watch_tasks[session_id].cancel()
                    del self.watch_tasks[session_id]

    async def broadcast(self, message: dict, session_id: str):
        """
        Broadcast a message to all connections for a session.

        Args:
            message: Dictionary to send as JSON
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            disconnected = []

            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, session_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming session updates.

    Connects to a session and streams real-time events from tool_calls.jsonl
    as they are written by the agent.

    Args:
        websocket: WebSocket connection
        session_id: Session identifier

    The client should send "ping" messages periodically for keepalive.
    The server will respond with "pong".

    Events are sent as JSON objects with the structure from tool_calls.jsonl:
    {
        "event": "tool_call_start" | "tool_call_complete",
        "timestamp": "ISO8601",
        "tool_use_id": "id",
        "agent_id": "AGENT-1",
        "agent_type": "agent-type",
        "tool_name": "ToolName",
        ...
    }
    """
    await manager.connect(websocket, session_id)

    # Get session directory
    logs_dir = Path("research_agent/logs")
    session_dir = logs_dir / session_id
    tool_log_path = session_dir / "tool_calls.jsonl"

    # Create file watcher
    watcher = FileWatcher()

    async def on_new_line(line: str):
        """Callback for new lines in tool_calls.jsonl."""
        try:
            event = json.loads(line)
            await manager.broadcast(event, session_id)
        except json.JSONDecodeError:
            # Skip malformed JSON
            pass
        except Exception as e:
            print(f"Error broadcasting event: {e}")

    # Start watching file
    watch_task = asyncio.create_task(
        watcher.watch_file(tool_log_path, on_new_line, poll_interval=0.5)
    )
    manager.watch_tasks[session_id] = watch_task

    try:
        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()

            # Handle ping/pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        if not manager.active_connections.get(session_id):
            watch_task.cancel()
    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(websocket, session_id)
        if not manager.active_connections.get(session_id):
            watch_task.cancel()
