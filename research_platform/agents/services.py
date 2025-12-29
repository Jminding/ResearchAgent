"""
Service layer for integrating with the research agent system.

This module encapsulates the agent execution logic and provides
a clean interface for views to trigger research runs and handle feedback.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from django.conf import settings
from django.core.files import File
from asgiref.sync import sync_to_async

from .models import ResearchSession, ResearchPaper, GeneratedFile, PeerReviewFeedback


def _ensure_agent_api_imported():
    """
    Lazily import agent_api module to avoid import errors at Django startup.

    This function adds the parent directory to the Python path
    and imports the agent_api module only when needed.
    """
    global agent_api

    if 'agent_api' not in globals():
        # Add parent directory to Python path (so 'import research_agent' works)
        parent_dir = str(settings.RESEARCH_AGENT_BASE_DIR.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Now import
        try:
            from research_agent import agent_api as _agent_api
            globals()['agent_api'] = _agent_api
        except ImportError as e:
            # More helpful error message
            raise ImportError(
                f"Could not import research_agent module. "
                f"Make sure research_agent directory exists at: {settings.RESEARCH_AGENT_BASE_DIR}. "
                f"Python path: {sys.path}. "
                f"Original error: {e}"
            )

    return globals()['agent_api']


class ResearchAgentService:
    """
    Service for executing research agent runs.

    Integrates with the existing agent_api.py module.
    """

    def __init__(self, api_key: str):
        """
        Initialize service with API key.

        Args:
            api_key: Anthropic API key for the research agent
        """
        self.api_key = api_key

    async def run_research(self, session: ResearchSession) -> Dict[str, Any]:
        """
        Run a complete research session asynchronously.

        Args:
            session: ResearchSession model instance

        Returns:
            Dictionary with result details
        """
        # Import agent_api lazily
        agent_api = _ensure_agent_api_imported()

        # Import memory utilities
        from django.conf import settings
        from .memory_utils import memory_limit, get_memory_limit_mb
        import logging

        logger = logging.getLogger(__name__)

        # Set API key in environment
        os.environ['ANTHROPIC_API_KEY'] = self.api_key

        # Create session directory
        session_dir = self._create_session_directory(session.session_id)

        try:
            # Mark session as running (wrap in sync_to_async)
            await sync_to_async(session.mark_running)()
            session.session_directory = str(session_dir)
            await sync_to_async(session.save)()

            # Get memory limit from settings
            memory_limit_bytes = getattr(settings, 'RESEARCH_AGENT_MEMORY_LIMIT', 8 * 1024 ** 3)

            if memory_limit_bytes > 0:
                logger.info(
                    f"Starting research with memory limit: {get_memory_limit_mb(memory_limit_bytes)}"
                )

            # Run research query using agent_api with memory limits
            with memory_limit(memory_limit_bytes):
                result = await agent_api.run_research_query(
                    query=session.topic,
                    session_dir=session_dir
                )

            # Update session paths
            if result['status'] == 'completed':
                session.transcript_path = result.get('transcript_path', '')
                await sync_to_async(session.save)()

                # Collect generated files
                await self._collect_generated_files(session, session_dir)

                # Mark as completed
                await sync_to_async(session.mark_completed)()

                return {
                    'success': True,
                    'session_dir': str(session_dir),
                    'message': 'Research completed successfully'
                }
            else:
                # Failed
                error_msg = result.get('error', 'Unknown error')
                await sync_to_async(session.mark_failed)(error_msg)

                return {
                    'success': False,
                    'error': error_msg
                }

        except Exception as e:
            # Handle any errors
            error_msg = str(e)
            await sync_to_async(session.mark_failed)(error_msg)

            return {
                'success': False,
                'error': error_msg
            }

    async def process_feedback(
        self,
        feedback: PeerReviewFeedback
    ) -> Dict[str, Any]:
        """
        Process peer review feedback and generate revisions.

        Args:
            feedback: PeerReviewFeedback model instance

        Returns:
            Dictionary with result details
        """
        # Import agent_api lazily
        agent_api = _ensure_agent_api_imported()

        # Import memory utilities
        from django.conf import settings
        from .memory_utils import memory_limit, get_memory_limit_mb
        import logging

        logger = logging.getLogger(__name__)

        session = feedback.session

        # Set API key in environment
        os.environ['ANTHROPIC_API_KEY'] = self.api_key

        try:
            # Mark feedback as processing
            await sync_to_async(feedback.mark_processing)()
            await sync_to_async(session.mark_revision)()

            # Create revision directory (reuse session directory)
            session_dir = Path(session.session_directory)

            # Construct feedback prompt for peer-reviewer agent
            feedback_prompt = f"""
You are reviewing a research paper and must address the following peer review feedback:

{feedback.feedback_text}

The paper and all supporting files are in the current session directory.

Please:
1. Read the current paper (files/reports/*.pdf)
2. Review all experimental outputs (files/results/)
3. Address each point of feedback systematically
4. Make necessary revisions to the paper
5. Re-compile the paper if changes were made

Be thorough and ensure all feedback points are adequately addressed.
"""

            # Use the existing agent API to spawn a peer-reviewer
            # For now, we'll create a simpler approach by re-running with modified query
            revision_query = f"""
[REVISION REQUEST]

Original Research Topic: {session.topic}

Peer Review Feedback:
{feedback.feedback_text}

Please address the above feedback and revise the research paper accordingly.
Use the existing session directory: {session_dir}
"""

            # Get memory limit from settings
            memory_limit_bytes = getattr(settings, 'RESEARCH_AGENT_MEMORY_LIMIT', 8 * 1024 ** 3)

            if memory_limit_bytes > 0:
                logger.info(
                    f"Starting revision with memory limit: {get_memory_limit_mb(memory_limit_bytes)}"
                )

            # Run revision with memory limits
            with memory_limit(memory_limit_bytes):
                result = await agent_api.run_research_query(
                    query=revision_query,
                    session_dir=session_dir
                )

            if result['status'] == 'completed':
                # Collect updated files
                await self._collect_generated_files(session, session_dir, update_existing=True)

                # Mark as completed
                await sync_to_async(feedback.mark_completed)(
                    revision_notes="Feedback addressed successfully. Updated paper generated."
                )
                await sync_to_async(session.mark_completed)()

                return {
                    'success': True,
                    'message': 'Feedback processed and revisions completed'
                }
            else:
                error_msg = result.get('error', 'Unknown error during revision')
                await sync_to_async(feedback.mark_failed)(error_msg)

                return {
                    'success': False,
                    'error': error_msg
                }

        except Exception as e:
            error_msg = str(e)
            await sync_to_async(feedback.mark_failed)(error_msg)

            return {
                'success': False,
                'error': error_msg
            }

    def _create_session_directory(self, session_id: str) -> Path:
        """Create and return session directory path."""
        session_dir = settings.RESEARCH_FILES_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        for subdir in ['files', 'files/research_notes', 'files/theory',
                       'files/data', 'files/experiments', 'files/results',
                       'files/charts', 'files/reports']:
            (session_dir / subdir).mkdir(parents=True, exist_ok=True)

        return session_dir

    async def _collect_generated_files(
        self,
        session: ResearchSession,
        session_dir: Path,
        update_existing: bool = False
    ):
        """
        Collect and store generated files from session directory.

        Args:
            session: ResearchSession instance
            session_dir: Path to session directory
            update_existing: If True, update existing files
        """
        # Look for PDF in global files/reports directory (where agents actually write)
        # Agents write PDFs to a global files/reports/ directory with session_id prefix
        from django.conf import settings
        global_reports_dir = settings.BASE_DIR / 'files' / 'reports'

        pdf_found = False
        if global_reports_dir.exists():
            # Look for PDF with session_id prefix
            session_id = session.session_id
            for pdf_path in global_reports_dir.glob(f'{session_id}*.pdf'):
                # Store PDF as ResearchPaper
                if update_existing:
                    # Update existing paper
                    paper, created = await sync_to_async(
                        ResearchPaper.objects.get_or_create
                    )(session=session)
                else:
                    # Create new paper if doesn't exist
                    has_paper = await sync_to_async(lambda: hasattr(session, 'paper'))()
                    if not has_paper:
                        paper = ResearchPaper(session=session)
                        created = True
                    else:
                        continue  # Paper already exists

                with open(pdf_path, 'rb') as f:
                    await sync_to_async(paper.pdf_file.save)(pdf_path.name, File(f), save=True)

                # Extract title from filename (remove session_id prefix)
                title = pdf_path.stem.replace(session_id + '_', '').replace('_', ' ')
                paper.title = title
                await sync_to_async(paper.save)()

                pdf_found = True
                break  # Only one PDF

        # Collect CSV files from results
        results_dir = session_dir / 'files' / 'results'
        if results_dir.exists():
            for csv_path in results_dir.glob('*.csv'):
                await self._save_generated_file(session, csv_path, 'csv', 'Experimental results')

        # Collect JSON files
        for json_dir in [session_dir / 'files' / 'results',
                         session_dir / 'files' / 'research_notes',
                         session_dir / 'files' / 'theory']:
            if json_dir.exists():
                for json_path in json_dir.glob('*.json'):
                    await self._save_generated_file(session, json_path, 'json', 'Structured data')

        # Collect charts
        charts_dir = session_dir / 'files' / 'charts'
        if charts_dir.exists():
            for img_path in charts_dir.glob('*.png'):
                await self._save_generated_file(session, img_path, 'png', 'Visualization')

        # Collect logs
        for log_path in session_dir.glob('*.txt'):
            if 'transcript' in log_path.name or 'log' in log_path.name:
                await self._save_generated_file(session, log_path, 'log', 'Session log')

    async def _save_generated_file(
        self,
        session: ResearchSession,
        file_path: Path,
        file_type: str,
        description: str
    ):
        """Save a generated file to database."""
        # Check if file already exists
        existing = await sync_to_async(
            GeneratedFile.objects.filter(
                session=session,
                filename=file_path.name
            ).first
        )()

        if existing:
            # Update existing file
            with open(file_path, 'rb') as f:
                await sync_to_async(existing.file.save)(file_path.name, File(f), save=True)
            existing.description = description
            await sync_to_async(existing.save)()
        else:
            # Create new file record
            with open(file_path, 'rb') as f:
                gen_file = GeneratedFile(
                    session=session,
                    file_type=file_type,
                    filename=file_path.name,
                    description=description
                )
                await sync_to_async(gen_file.file.save)(file_path.name, File(f), save=True)
                await sync_to_async(gen_file.save)()


def run_research_sync(session: ResearchSession, api_key: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for running research.

    Args:
        session: ResearchSession instance
        api_key: Anthropic API key

    Returns:
        Result dictionary
    """
    service = ResearchAgentService(api_key)
    return asyncio.run(service.run_research(session))


def process_feedback_sync(feedback: PeerReviewFeedback, api_key: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for processing feedback.

    Args:
        feedback: PeerReviewFeedback instance
        api_key: Anthropic API key

    Returns:
        Result dictionary
    """
    service = ResearchAgentService(api_key)
    return asyncio.run(service.process_feedback(feedback))
