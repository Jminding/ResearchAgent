/**
 * Visual indicator for research pipeline phases
 */

import React from 'react';
import { SubagentSpawn, AgentType } from '../types';
import { CheckCircle, Circle, Loader } from 'lucide-react';

const PIPELINE_PHASES = [
  { name: 'Literature Review', agentType: AgentType.LITERATURE_REVIEWER },
  { name: 'Theory Development', agentType: AgentType.THEORIST },
  { name: 'Data Collection', agentType: AgentType.DATA_COLLECTOR },
  { name: 'Experimentation', agentType: AgentType.EXPERIMENTALIST },
  { name: 'Analysis', agentType: AgentType.ANALYST },
  { name: 'Report Writing', agentType: AgentType.REPORT_WRITER },
  { name: 'PDF Compilation', agentType: AgentType.LATEX_COMPILER },
];

interface PipelinePhaseIndicatorProps {
  subagents: SubagentSpawn[];
  activeAgents: Set<string>;
  lastActivityTime: Map<string, number>;
}

const IDLE_THRESHOLD_MS = 30000; // 30 seconds

export function PipelinePhaseIndicator({
  subagents,
  activeAgents,
  lastActivityTime,
}: PipelinePhaseIndicatorProps) {
  const getPhaseStatus = (agentType: AgentType, phaseIndex: number) => {
    // Find all subagents of this type
    const phaseSubagents = subagents.filter((s) => s.agent_type === agentType);
    const hasSpawned = phaseSubagents.length > 0;

    if (!hasSpawned) return 'pending';

    // Check if any agent of this type is active and not idle
    const hasActiveAgent = phaseSubagents.some((subagent) => {
      if (!activeAgents.has(subagent.agent_id)) return false;

      const lastActivity = lastActivityTime.get(subagent.agent_id);
      if (!lastActivity) return true; // No activity recorded yet, consider active

      const isIdle = Date.now() - lastActivity > IDLE_THRESHOLD_MS;
      return !isIdle; // Active if not idle
    });

    // Check if a later phase has started (indicates this phase is complete)
    const laterPhaseStarted = PIPELINE_PHASES
      .slice(phaseIndex + 1)
      .some((phase) => subagents.some((s) => s.agent_type === phase.agentType));

    if (hasActiveAgent) return 'active';
    if (laterPhaseStarted || hasSpawned) return 'completed';
    return 'pending';
  };

  return (
    <div className="flex items-center justify-between w-full p-6 bg-white rounded-lg shadow-sm">
      {PIPELINE_PHASES.map((phase, index) => {
        const status = getPhaseStatus(phase.agentType, index);

        return (
          <React.Fragment key={phase.agentType}>
            <div className="flex flex-col items-center">
              <div
                className={`
                  flex items-center justify-center w-12 h-12 rounded-full transition-all
                  ${status === 'completed' ? 'bg-green-500 text-white' : ''}
                  ${status === 'active' ? 'bg-blue-500 text-white' : ''}
                  ${status === 'pending' ? 'bg-gray-200 text-gray-400' : ''}
                `}
              >
                {status === 'completed' && <CheckCircle className="w-6 h-6" />}
                {status === 'active' && <Loader className="w-6 h-6 animate-spin" />}
                {status === 'pending' && <Circle className="w-6 h-6" />}
              </div>
              <span className="mt-2 text-xs text-center text-gray-700 font-medium">
                {phase.name}
              </span>
            </div>
            {index < PIPELINE_PHASES.length - 1 && (
              <div
                className={`
                  flex-1 h-1 mx-2 transition-all
                  ${status === 'completed' ? 'bg-green-500' : 'bg-gray-200'}
                `}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
