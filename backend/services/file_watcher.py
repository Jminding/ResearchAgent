"""Service for watching log files for changes."""

import asyncio
from pathlib import Path
from typing import Callable
import aiofiles


class FileWatcher:
    """Watch log files for new content and stream updates."""

    def __init__(self):
        """Initialize file watcher."""
        self.watchers = {}

    async def watch_file(
        self,
        file_path: Path,
        callback: Callable,
        poll_interval: float = 0.5
    ):
        """
        Watch a file for new content and call callback with new lines.

        Args:
            file_path: Path to file to watch
            callback: Async function to call with new lines
            poll_interval: How often to check for changes (seconds)
        """
        last_position = 0

        while True:
            try:
                if file_path.exists():
                    current_size = file_path.stat().st_size

                    if current_size > last_position:
                        # Read new content
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            await f.seek(last_position)
                            new_content = await f.read()

                            # Process each new line
                            for line in new_content.splitlines():
                                if line.strip():
                                    await callback(line)

                        last_position = current_size

                    elif current_size < last_position:
                        # File was truncated, reset position
                        last_position = 0

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                # Graceful shutdown
                break
            except Exception as e:
                print(f"Error watching file {file_path}: {e}")
                await asyncio.sleep(poll_interval)

    async def watch_file_from_offset(
        self,
        file_path: Path,
        line_offset: int,
        callback: Callable,
        poll_interval: float = 0.5
    ):
        """
        Watch a file starting from a specific line offset.

        Args:
            file_path: Path to file to watch
            line_offset: Line number to start from (0-indexed)
            callback: Async function to call with new lines
            poll_interval: How often to check for changes (seconds)
        """
        last_line_count = line_offset

        while True:
            try:
                if file_path.exists():
                    # Count current lines
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        lines = await f.readlines()

                    current_line_count = len(lines)

                    if current_line_count > last_line_count:
                        # Process new lines
                        new_lines = lines[last_line_count:current_line_count]
                        for line in new_lines:
                            if line.strip():
                                await callback(line)

                        last_line_count = current_line_count

                    elif current_line_count < last_line_count:
                        # File was truncated, reset
                        last_line_count = 0

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error watching file {file_path}: {e}")
                await asyncio.sleep(poll_interval)
