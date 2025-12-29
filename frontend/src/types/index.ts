/**
 * TypeScript type definitions matching backend schemas
 */

export enum AgentType {
  LEAD = "lead",
  LITERATURE_REVIEWER = "literature-reviewer",
  THEORIST = "theorist",
  DATA_COLLECTOR = "data-collector",
  EXPERIMENTALIST = "experimentalist",
  ANALYST = "analyst",
  REPORT_WRITER = "report-writer",
  LATEX_COMPILER = "latex-compiler",
}

export enum SessionStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
}

export interface ToolCallEvent {
  event: "tool_call_start" | "tool_call_complete";
  timestamp: string;
  tool_use_id: string;
  agent_id: string;
  agent_type: string;
  tool_name: string;
  tool_input?: Record<string, any>;
  parent_tool_use_id?: string;
  success?: boolean;
  error?: string;
  output_size?: number;
}

export interface SubagentSpawn {
  agent_id: string;
  agent_type: AgentType;
  spawned_at: string;
  description: string;
  parent_tool_use_id: string;
}

export interface SessionMetadata {
  session_id: string;
  created_at: string;
  status: SessionStatus;
  query: string;
  session_dir: string;
  transcript_path: string;
  tool_log_path: string;
}

export interface SessionDetail extends SessionMetadata {
  subagents: SubagentSpawn[];
  tool_calls: ToolCallEvent[];
  transcript_preview?: string;
}

export interface SessionStats {
  session_id: string;
  total_tool_calls: number;
  tool_counts: Record<string, number>;
  agent_activity: Record<string, number>;
  error_count: number;
  errors: ToolCallEvent[];
}
