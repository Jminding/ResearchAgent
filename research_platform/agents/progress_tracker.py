"""
Real-time progress tracking for research sessions.

Parses tool_calls.jsonl to track agent activity and progress.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class SessionProgressTracker:
    """
    Tracks progress of a research session by parsing logs.
    """

    def __init__(self, session_dir: Path):
        """
        Initialize tracker with session directory.

        Args:
            session_dir: Path to session directory
        """
        self.session_dir = Path(session_dir)
        self.tool_calls_file = self.session_dir / 'tool_calls.jsonl'
        self.transcript_file = self.session_dir / 'transcript.txt'

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current session progress.

        Returns:
            Dictionary with progress information:
            - active_agents: List of currently active agents
            - completed_agents: List of completed agents
            - tool_calls: Recent tool calls
            - latest_update: Most recent update
            - transcript_preview: Last few lines of transcript
        """
        progress = {
            'active_agents': [],
            'completed_agents': [],
            'tool_calls': [],
            'latest_update': None,
            'transcript_preview': '',
            'agents_by_type': {},
            'total_tool_calls': 0
        }

        # Parse tool calls if file exists
        if self.tool_calls_file.exists():
            tool_calls = self._parse_tool_calls()
            progress['tool_calls'] = tool_calls[-30:]  # Last 30 calls for UI display
            progress['total_tool_calls'] = len(tool_calls)

            # Extract agent information
            agent_info = self._extract_agent_info(tool_calls)
            progress['active_agents'] = agent_info['active']
            progress['completed_agents'] = agent_info['completed']
            progress['agents_by_type'] = agent_info['by_type']

            # Latest update
            if tool_calls:
                latest = tool_calls[-1]
                progress['latest_update'] = {
                    'timestamp': latest.get('timestamp', ''),
                    'agent': latest.get('agent_type', 'unknown'),
                    'tool': latest.get('tool_name', ''),
                    'event': latest.get('event', '')
                }

        # Get transcript preview
        if self.transcript_file.exists():
            progress['transcript_preview'] = self._get_transcript_preview()

        return progress

    def _parse_tool_calls(self) -> List[Dict[str, Any]]:
        """Parse tool_calls.jsonl file."""
        tool_calls = []

        try:
            with open(self.tool_calls_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            call = json.loads(line)
                            tool_calls.append(call)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        return tool_calls

    def _extract_agent_info(self, tool_calls: List[Dict]) -> Dict[str, Any]:
        """
        Extract agent information from tool calls.

        Uses a sliding window approach - agents are "active" if they've had
        activity in the most recent tool calls (last 50 calls or 5 minutes).
        """
        agent_sessions = {}  # agent_id -> info
        agent_types_seen = {}  # agent_type -> last_index

        # Track all agents and their activity
        for idx, call in enumerate(tool_calls):
            agent_id = call.get('agent_id', '')
            agent_type = call.get('agent_type', 'unknown')
            timestamp = call.get('timestamp', '')

            if not agent_id:
                continue

            if agent_id not in agent_sessions:
                agent_sessions[agent_id] = {
                    'id': agent_id,
                    'type': agent_type,
                    'started': timestamp,
                    'status': 'active',
                    'tool_calls': [],
                    'last_activity': timestamp,
                    'last_index': idx
                }

            agent_sessions[agent_id]['tool_calls'].append(call)
            agent_sessions[agent_id]['last_activity'] = timestamp
            agent_sessions[agent_id]['last_index'] = idx

            # Track the most recent index for each agent type
            agent_types_seen[agent_type] = idx

        # Determine which agents are currently active
        # An agent type is active if it appears in the last 50 tool calls
        recent_window = 50
        recent_threshold = max(0, len(tool_calls) - recent_window)

        active = []
        completed = []
        by_type = {}

        for agent in agent_sessions.values():
            # Group by type
            agent_type = agent['type']
            if agent_type not in by_type:
                by_type[agent_type] = []
            by_type[agent_type].append(agent)

            # Check if this agent has recent activity
            if agent['last_index'] >= recent_threshold:
                agent['status'] = 'active'
                active.append(agent)
            else:
                agent['status'] = 'completed'
                agent['completed'] = agent['last_activity']
                completed.append(agent)

        return {
            'active': active,
            'completed': completed,
            'by_type': by_type
        }

    def _get_transcript_preview(self, lines: int = 20) -> str:
        """Get last N lines of transcript."""
        try:
            with open(self.transcript_file, 'r') as f:
                content = f.readlines()
                return ''.join(content[-lines:])
        except Exception:
            return ''

    def get_current_step(self) -> Optional[str]:
        """
        Determine current research step based on agent activity.

        Returns:
            User-friendly string describing current step
        """
        progress = self.get_progress()

        # If no agents yet, starting up
        if not progress['active_agents'] and not progress['completed_agents']:
            return "Starting research session..."

        # Check active agents first - look at all active agent types
        if progress['active_agents']:
            # Get all active agent types
            active_types = [agent['type'].lower() for agent in progress['active_agents']]

            # Prioritize by research stage order (later stages take precedence)
            stage_priority = [
                ('latex', 'compiler', 'Compiling final PDF document'),
                ('report', 'writer', 'Writing research paper'),
                ('analyst', None, 'Analyzing results and data'),
                ('experimentalist', 'experiment', 'Running experiments and tests'),
                ('data', 'collector', 'Collecting and preparing data'),
                ('experimental-designer', 'design', 'Designing experiments'),
                ('theorist', 'theory', 'Developing theoretical framework'),
                ('literature', 'researcher', 'Reviewing research literature'),
                ('lead', None, 'Coordinating research workflow'),
            ]

            # Check stages in reverse order (prioritize later stages)
            for primary, secondary, message in stage_priority:
                for agent_type in active_types:
                    if primary in agent_type or (secondary and secondary in agent_type):
                        return message

            # If no match, return generic message with agent type
            return f"Working on {active_types[-1]}"

        # No active agents but have completed ones - just finished
        if progress['completed_agents']:
            return "Finalizing..."

        # Fallback
        return "Processing..."
