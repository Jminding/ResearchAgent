/**
 * Card component displaying subagent information
 */

import React from 'react';
import { SubagentSpawn, AgentType } from '../types';
import { Brain, BookOpen, Database, FlaskConical, BarChart, FileText, FileType } from 'lucide-react';

const AGENT_ICONS: Record<AgentType, React.ReactNode> = {
  [AgentType.LEAD]: <Brain className="w-5 h-5" />,
  [AgentType.LITERATURE_REVIEWER]: <BookOpen className="w-5 h-5" />,
  [AgentType.THEORIST]: <Brain className="w-5 h-5" />,
  [AgentType.DATA_COLLECTOR]: <Database className="w-5 h-5" />,
  [AgentType.EXPERIMENTALIST]: <FlaskConical className="w-5 h-5" />,
  [AgentType.ANALYST]: <BarChart className="w-5 h-5" />,
  [AgentType.REPORT_WRITER]: <FileText className="w-5 h-5" />,
  [AgentType.LATEX_COMPILER]: <FileType className="w-5 h-5" />,
};

const AGENT_COLORS: Record<AgentType, string> = {
  [AgentType.LEAD]: 'bg-purple-100 text-purple-700',
  [AgentType.LITERATURE_REVIEWER]: 'bg-blue-100 text-blue-700',
  [AgentType.THEORIST]: 'bg-indigo-100 text-indigo-700',
  [AgentType.DATA_COLLECTOR]: 'bg-green-100 text-green-700',
  [AgentType.EXPERIMENTALIST]: 'bg-orange-100 text-orange-700',
  [AgentType.ANALYST]: 'bg-cyan-100 text-cyan-700',
  [AgentType.REPORT_WRITER]: 'bg-pink-100 text-pink-700',
  [AgentType.LATEX_COMPILER]: 'bg-red-100 text-red-700',
};

interface SubagentCardProps {
  subagent: SubagentSpawn;
  isActive: boolean;
  isIdle?: boolean;
}

export function SubagentCard({ subagent, isActive, isIdle = false }: SubagentCardProps) {
  const colorClass = AGENT_COLORS[subagent.agent_type as AgentType] || 'bg-gray-100 text-gray-700';
  const icon = AGENT_ICONS[subagent.agent_type as AgentType] || <Brain className="w-5 h-5" />;

  return (
    <div
      className={`
        p-4 rounded-lg border-2 transition-all
        ${isActive && !isIdle ? 'border-blue-500 shadow-lg' : 'border-gray-200'}
        ${isIdle ? 'opacity-75' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${colorClass}`}>
          {icon}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900">{subagent.agent_id}</h3>
          <p className="text-sm text-gray-600">{subagent.description}</p>
        </div>
        {isActive && !isIdle && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-600">Active</span>
          </div>
        )}
        {isActive && isIdle && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-yellow-500 rounded-full" />
            <span className="text-xs text-gray-600">Idle</span>
          </div>
        )}
      </div>
    </div>
  );
}
