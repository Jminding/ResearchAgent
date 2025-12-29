/**
 * Timeline component displaying tool calls
 */

import React from 'react';
import { ToolCallEvent } from '../types';
import { formatDistanceToNow } from 'date-fns';
import { Search, FileEdit, Code, Database } from 'lucide-react';

const TOOL_ICONS: Record<string, React.ReactNode> = {
  WebSearch: <Search className="w-4 h-4" />,
  Write: <FileEdit className="w-4 h-4" />,
  Bash: <Code className="w-4 h-4" />,
  Read: <Database className="w-4 h-4" />,
};

interface ToolCallTimelineProps {
  toolCalls: ToolCallEvent[];
}

export function ToolCallTimeline({ toolCalls }: ToolCallTimelineProps) {
  if (toolCalls.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No tool calls yet. Waiting for agent activity...
      </div>
    );
  }

  // Reverse array to show latest on top
  const reversedCalls = [...toolCalls].reverse();

  return (
    <div className="space-y-2">
      {reversedCalls.map((call, index) => (
        <div
          key={`${call.tool_use_id}-${index}`}
          className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <div className="p-2 bg-white rounded-lg">
            {TOOL_ICONS[call.tool_name] || <Code className="w-4 h-4" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm">{call.agent_id}</span>
              <span className="text-gray-400">→</span>
              <span className="text-sm text-gray-700">{call.tool_name}</span>
              {call.success === false && (
                <span className="text-xs text-red-600 font-medium">Failed</span>
              )}
            </div>
            {call.tool_input && (
              <div className="mt-1 text-xs text-gray-600 truncate">
                {call.tool_input.query ||
                  call.tool_input.file_path ||
                  call.tool_input.command ||
                  'Working...'}
              </div>
            )}
          </div>
          <div className="text-xs text-gray-500 whitespace-nowrap">
            {formatDistanceToNow(new Date(call.timestamp), { addSuffix: true })}
          </div>
        </div>
      ))}
    </div>
  );
}
