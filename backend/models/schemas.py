"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    """Enum for different agent types."""
    LEAD = "lead"
    LITERATURE_REVIEWER = "literature-reviewer"
    THEORIST = "theorist"
    DATA_COLLECTOR = "data-collector"
    EXPERIMENTALIST = "experimentalist"
    ANALYST = "analyst"
    REPORT_WRITER = "report-writer"
    LATEX_COMPILER = "latex-compiler"
    GENERAL_PURPOSE = "general-purpose"  # For backward compatibility


class SessionStatus(str, Enum):
    """Enum for session status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCallEvent(BaseModel):
    """Model for tool call events from tool_calls.jsonl."""
    event: str  # "tool_call_start" | "tool_call_complete"
    timestamp: str
    tool_use_id: str
    agent_id: str
    agent_type: str
    tool_name: str
    tool_input: Optional[Dict[str, Any]] = None
    parent_tool_use_id: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    output_size: Optional[int] = None


class SubagentSpawn(BaseModel):
    """Model for subagent spawn information."""
    agent_id: str
    agent_type: str  # Changed from AgentType enum to string for flexibility
    spawned_at: str
    description: str
    parent_tool_use_id: str


class SessionMetadata(BaseModel):
    """Basic session metadata."""
    session_id: str
    created_at: datetime
    status: SessionStatus
    query: str
    session_dir: str
    transcript_path: str
    tool_log_path: str


class SessionDetail(SessionMetadata):
    """Detailed session information with subagents and tool calls."""
    subagents: List[SubagentSpawn] = []
    tool_calls: List[ToolCallEvent] = []
    transcript_preview: Optional[str] = None


class ResearchRequest(BaseModel):
    """Request model for submitting research query."""
    query: str = Field(..., min_length=1, max_length=5000, description="Research query")


class ResearchResponse(BaseModel):
    """Response model for research submission."""
    session_id: str
    status: SessionStatus
    message: str


class SessionStats(BaseModel):
    """Session statistics."""
    session_id: str
    total_tool_calls: int
    tool_counts: Dict[str, int]
    agent_activity: Dict[str, int]
    error_count: int
    errors: List[ToolCallEvent]
