"""Background task management for research agent execution."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime

# Add research_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "research_agent"))

from research_agent.agent_api import run_research_query


class BackgroundTaskManager:
    """Manage background research tasks."""

    def __init__(self):
        """Initialize background task manager."""
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_status: Dict[str, dict] = {}

    async def start_research_task(
        self,
        session_id: str,
        query: str,
        session_dir: Path,
        on_progress: Optional[Callable] = None
    ) -> asyncio.Task:
        """
        Start a research task in the background.

        Args:
            session_id: Unique session identifier
            query: Research query
            session_dir: Directory for session logs
            on_progress: Optional async callback for progress updates

        Returns:
            Asyncio task handle
        """

        async def task_wrapper():
            """Wrapper to track task status and handle errors."""
            try:
                self.task_status[session_id] = {
                    'status': 'running',
                    'started_at': datetime.now().isoformat(),
                    'query': query
                }

                result = await run_research_query(
                    query=query,
                    session_dir=session_dir,
                    on_progress=on_progress
                )

                self.task_status[session_id] = {
                    'status': result['status'],
                    'completed_at': datetime.now().isoformat(),
                    'result': result,
                    'query': query
                }

            except Exception as e:
                self.task_status[session_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat(),
                    'query': query
                }
                print(f"Research task error for {session_id}: {e}")

        task = asyncio.create_task(task_wrapper())
        self.tasks[session_id] = task
        return task

    def get_task_status(self, session_id: str) -> Optional[dict]:
        """
        Get status of a task.

        Args:
            session_id: Session identifier

        Returns:
            Task status dictionary or None if not found
        """
        return self.task_status.get(session_id)

    async def cancel_task(self, session_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            session_id: Session identifier

        Returns:
            True if task was cancelled, False if not found
        """
        if session_id in self.tasks:
            self.tasks[session_id].cancel()
            self.task_status[session_id] = {
                **self.task_status.get(session_id, {}),
                'status': 'cancelled',
                'cancelled_at': datetime.now().isoformat()
            }
            return True
        return False

    def is_running(self, session_id: str) -> bool:
        """
        Check if a task is currently running.

        Args:
            session_id: Session identifier

        Returns:
            True if task is running
        """
        if session_id in self.tasks:
            task = self.tasks[session_id]
            return not task.done()
        return False

    def cleanup_completed_tasks(self):
        """Remove completed tasks from memory."""
        completed = [
            sid for sid, task in self.tasks.items()
            if task.done()
        ]
        for sid in completed:
            del self.tasks[sid]


# Global instance
task_manager = BackgroundTaskManager()
