/**
 * Session detail page with real-time updates
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { sessionsApi } from '../services/api';
import { useSession } from '../contexts/SessionContext';
import { useWebSocket } from '../hooks/useWebSocket';
import { SubagentCard } from '../components/SubagentCard';
import { ToolCallTimeline } from '../components/ToolCallTimeline';
import { PipelinePhaseIndicator } from '../components/PipelinePhaseIndicator';
import { ArrowLeft, Download, FileText } from 'lucide-react';

// Time threshold for considering an agent idle (30 seconds)
const IDLE_THRESHOLD_MS = 30000;

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { state, dispatch } = useSession();
  const [, setIdleCheckTick] = useState(0);

  const { data: session, isLoading } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => sessionsApi.getSession(sessionId!),
    enabled: !!sessionId,
  });

  // Query for PDFs (poll every 10 seconds while session is running)
  const { data: pdfsData } = useQuery({
    queryKey: ['pdfs', sessionId],
    queryFn: () => sessionsApi.listPdfs(sessionId!),
    enabled: !!sessionId,
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Connect to WebSocket for real-time updates
  const { isConnected } = useWebSocket({
    sessionId: sessionId!,
    onMessage: (event) => {
      dispatch({ type: 'ADD_TOOL_CALL', payload: event });

      // Update active agents based on events
      if (event.event === 'tool_call_start') {
        dispatch({ type: 'SET_ACTIVE_AGENT', payload: event.agent_id });
      }

      // Detect subagent spawns (heuristic: new agent_id appears)
      if (!state.subagents.some((s) => s.agent_id === event.agent_id)) {
        dispatch({
          type: 'ADD_SUBAGENT',
          payload: {
            agent_id: event.agent_id,
            agent_type: event.agent_type as any,
            spawned_at: event.timestamp,
            description: `${event.agent_type} working`,
            parent_tool_use_id: event.parent_tool_use_id || '',
          },
        });
      }
    },
  });

  // Initialize session state
  useEffect(() => {
    if (session) {
      dispatch({
        type: 'START_SESSION',
        payload: {
          sessionId: session.session_id,
          query: session.query,
        },
      });

      // Load existing subagents
      session.subagents.forEach((subagent) => {
        dispatch({ type: 'ADD_SUBAGENT', payload: subagent });
      });

      // Load existing tool calls
      session.tool_calls.forEach((toolCall) => {
        dispatch({ type: 'ADD_TOOL_CALL', payload: toolCall });
      });
    }
  }, [session, dispatch]);

  // Periodically check for idle agents (every 5 seconds)
  useEffect(() => {
    const interval = setInterval(() => {
      // Force re-render to update idle status calculations
      setIdleCheckTick((tick) => tick + 1);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Helper function to check if an agent is idle
  const isAgentIdle = (agentId: string): boolean => {
    const lastActivity = state.lastActivityTime.get(agentId);
    if (!lastActivity) return false;
    return Date.now() - lastActivity > IDLE_THRESHOLD_MS;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-600">Loading session...</div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <div className="text-gray-600 mb-4">Session not found</div>
        <button
          onClick={() => navigate('/dashboard')}
          className="text-blue-600 hover:text-blue-700 underline"
        >
          Return to dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
            >
              <ArrowLeft size={20} />
              Back to Dashboard
            </button>
            <h1 className="text-3xl font-bold mb-2 text-gray-900">
              {session.query}
            </h1>
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span className="font-mono text-xs">{session.session_id}</span>
              <span>•</span>
              <span>
                {isConnected ? (
                  <span className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    Connected
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full" />
                    Disconnected
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Pipeline Phase Indicator */}
        <PipelinePhaseIndicator
          subagents={state.subagents}
          activeAgents={state.activeAgents}
          lastActivityTime={state.lastActivityTime}
        />

        {/* Active Subagents */}
        {state.subagents.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold mb-4 text-gray-900">
              Agents
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {state.subagents.map((subagent) => (
                <SubagentCard
                  key={subagent.agent_id}
                  subagent={subagent}
                  isActive={state.activeAgents.has(subagent.agent_id)}
                  isIdle={isAgentIdle(subagent.agent_id)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Research Papers (PDFs) */}
        {pdfsData && pdfsData.pdfs && pdfsData.pdfs.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold mb-4 text-gray-900">
              Research Papers
            </h2>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="space-y-3">
                {pdfsData.pdfs.map((pdf: any) => (
                  <div
                    key={pdf.filename}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-red-100 rounded-lg">
                        <FileText className="w-5 h-5 text-red-600" />
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">
                          {pdf.filename}
                        </div>
                        <div className="text-sm text-gray-600">
                          {(pdf.size_bytes / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    </div>
                    <a
                      href={sessionsApi.getPdfUrl(sessionId!, pdf.filename)}
                      download
                      className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download PDF
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tool Call Timeline */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-900">
            Activity Timeline
          </h2>
          <div className="bg-white rounded-lg shadow p-4 max-h-96 overflow-y-auto">
            <ToolCallTimeline toolCalls={state.toolCalls} />
          </div>
        </div>
      </div>
    </div>
  );
}
