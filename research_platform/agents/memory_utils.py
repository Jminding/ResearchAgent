"""
Memory management utilities for the research agent.
"""
import sys
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def memory_limit(max_memory_bytes: int):
    """
    Context manager to set memory limits for the current process.

    Args:
        max_memory_bytes: Maximum memory in bytes (0 to disable)

    Note:
        - On Unix/Linux/Mac: Uses resource.setrlimit() for hard limits
        - On Windows: Logs a warning (resource module not fully supported)
        - Memory limits apply to the entire process, not just the context

    Example:
        with memory_limit(8 * 1024 * 1024 * 1024):  # 8GB
            run_memory_intensive_task()
    """
    if max_memory_bytes <= 0:
        logger.info("Memory limits disabled (max_memory_bytes <= 0)")
        yield
        return

    # Try to set resource limits (Unix/Linux/Mac only)
    try:
        import resource

        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        original_limit = (soft, hard)

        # Set new memory limit (virtual memory address space)
        # Note: RLIMIT_AS limits total virtual memory (more reliable than RLIMIT_RSS)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            logger.info(
                f"Memory limit set to {max_memory_bytes / (1024**3):.2f} GB "
                f"(was {soft / (1024**3) if soft != resource.RLIM_INFINITY else 'unlimited':.2f} GB)"
            )
        except ValueError as e:
            logger.warning(f"Could not set memory limit: {e}")
            # Continue without setting limit
            yield
            return

        try:
            yield
        finally:
            # Restore original limits
            try:
                resource.setrlimit(resource.RLIMIT_AS, original_limit)
                logger.debug("Memory limits restored to original values")
            except Exception as e:
                logger.error(f"Failed to restore memory limits: {e}")

    except ImportError:
        # Windows or system without resource module
        logger.warning(
            "Memory limits not supported on this platform (resource module not available). "
            f"Requested limit: {max_memory_bytes / (1024**3):.2f} GB"
        )
        yield
    except Exception as e:
        logger.error(f"Unexpected error setting memory limits: {e}")
        yield


def get_memory_limit_mb(max_memory_bytes: int) -> str:
    """
    Format memory limit in human-readable format.

    Args:
        max_memory_bytes: Memory limit in bytes

    Returns:
        Human-readable string (e.g., "8.00 GB", "512 MB")
    """
    if max_memory_bytes <= 0:
        return "unlimited"

    gb = max_memory_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f} GB"

    mb = max_memory_bytes / (1024 ** 2)
    return f"{mb:.2f} MB"


def check_available_memory() -> Optional[int]:
    """
    Check available system memory.

    Returns:
        Available memory in bytes, or None if unable to determine
    """
    try:
        import psutil
        available = psutil.virtual_memory().available
        logger.debug(f"Available memory: {available / (1024**3):.2f} GB")
        return available
    except ImportError:
        logger.debug("psutil not available, cannot check available memory")
        return None
    except Exception as e:
        logger.error(f"Error checking available memory: {e}")
        return None
