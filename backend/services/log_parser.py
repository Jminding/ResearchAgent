"""Service for parsing tool_calls.jsonl and extracting session data."""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
from collections import defaultdict


class LogParser:
    """Parse tool_calls.jsonl and transcript.txt files."""

    @staticmethod
    def parse_tool_calls(log_path: Path) -> List[Dict[str, Any]]:
        """
        Parse tool_calls.jsonl into structured events.

        Args:
            log_path: Path to tool_calls.jsonl file

        Returns:
            List of event dictionaries
        """
        events = []

        if not log_path.exists():
            return events

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        return events

    @staticmethod
    def parse_tool_calls_stream(log_path: Path, offset: int = 0) -> Iterator[Dict[str, Any]]:
        """
        Stream new events from tool_calls.jsonl starting from offset.

        Args:
            log_path: Path to tool_calls.jsonl file
            offset: Line offset to start from

        Yields:
            Event dictionaries for new lines
        """
        if not log_path.exists():
            return

        with open(log_path, 'r', encoding='utf-8') as f:
            # Skip to offset
            for _ in range(offset):
                f.readline()

            # Yield new lines
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

    @staticmethod
    def extract_subagents(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique subagent spawns from events.

        Args:
            events: List of tool call events

        Returns:
            List of subagent information dictionaries
        """
        subagents = {}

        for event in events:
            agent_id = event.get('agent_id')
            # Skip main agent and duplicates
            if agent_id and agent_id != 'MAIN_AGENT' and agent_id not in subagents:
                # Infer description from first tool call if not in event
                description = event.get('description', f"{event.get('agent_type', 'agent')} working")

                subagents[agent_id] = {
                    'agent_id': agent_id,
                    'agent_type': event.get('agent_type', 'unknown'),
                    'spawned_at': event.get('timestamp', ''),
                    'parent_tool_use_id': event.get('parent_tool_use_id', ''),
                    'description': description
                }

        return list(subagents.values())

    @staticmethod
    def get_session_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate session statistics from events.

        Args:
            events: List of tool call events

        Returns:
            Dictionary with statistics
        """
        tool_counts = defaultdict(int)
        agent_activity = defaultdict(int)
        errors = []

        for event in events:
            if event.get('event') == 'tool_call_start':
                tool_name = event.get('tool_name', 'unknown')
                agent_id = event.get('agent_id', 'unknown')

                tool_counts[tool_name] += 1
                agent_activity[agent_id] += 1

            if event.get('success') is False or event.get('error'):
                errors.append(event)

        return {
            'total_tool_calls': sum(tool_counts.values()),
            'tool_counts': dict(tool_counts),
            'agent_activity': dict(agent_activity),
            'error_count': len(errors),
            'errors': errors
        }

    @staticmethod
    def get_line_count(log_path: Path) -> int:
        """
        Get the number of lines in a log file.

        Args:
            log_path: Path to log file

        Returns:
            Number of lines
        """
        if not log_path.exists():
            return 0

        with open(log_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
