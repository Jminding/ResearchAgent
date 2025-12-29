"""
Management command to manually collect files for research sessions.
"""
import asyncio
from pathlib import Path
from django.core.management.base import BaseCommand
from agents.models import ResearchSession
from agents.services import ResearchAgentService


class Command(BaseCommand):
    help = 'Manually collect generated files (PDFs, CSVs, etc.) for research sessions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--session-id',
            type=str,
            help='Collect files for a specific session ID',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Collect files for all completed sessions',
        )

    def handle(self, *args, **options):
        session_id = options.get('session_id')
        collect_all = options.get('all')

        if session_id:
            # Collect for specific session
            try:
                session = ResearchSession.objects.get(session_id=session_id)
                self.stdout.write(f"Collecting files for session: {session_id}")
                asyncio.run(self._collect_for_session(session))
                self.stdout.write(self.style.SUCCESS(f"Successfully collected files for {session_id}"))
            except ResearchSession.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Session {session_id} not found"))
        elif collect_all:
            # Collect for all completed sessions
            sessions = ResearchSession.objects.filter(status='completed')
            self.stdout.write(f"Found {sessions.count()} completed sessions")
            for session in sessions:
                self.stdout.write(f"Collecting files for session: {session.session_id}")
                try:
                    asyncio.run(self._collect_for_session(session))
                    self.stdout.write(self.style.SUCCESS(f"  ✓ {session.session_id}"))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ {session.session_id}: {str(e)}"))
        else:
            self.stdout.write(self.style.ERROR("Please specify --session-id or --all"))

    async def _collect_for_session(self, session):
        """Collect files for a single session."""
        if not session.session_directory:
            raise ValueError("Session directory not set")

        session_dir = Path(session.session_directory)
        service = ResearchAgentService(api_key="dummy")  # API key not needed for collection
        await service._collect_generated_files(session, session_dir, update_existing=True)
