/**
 * Dashboard page displaying all research sessions
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { sessionsApi } from '../services/api';
import { formatDistanceToNow } from 'date-fns';
import { SessionStatus } from '../types';
import { Plus } from 'lucide-react';

export function Dashboard() {
  const navigate = useNavigate();

  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => sessionsApi.listSessions(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const getStatusBadge = (status: SessionStatus) => {
    const styles = {
      [SessionStatus.COMPLETED]: 'bg-green-100 text-green-800',
      [SessionStatus.RUNNING]: 'bg-blue-100 text-blue-800 animate-pulse',
      [SessionStatus.FAILED]: 'bg-red-100 text-red-800',
      [SessionStatus.PENDING]: 'bg-gray-100 text-gray-800',
    };

    return (
      <span
        className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status]}`}
      >
        {status}
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-600">Loading sessions...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Research Sessions</h1>
          <button
            onClick={() => navigate('/new')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus size={20} />
            New Research
          </button>
        </div>

        {sessions && sessions.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <p className="text-gray-600 mb-4">No research sessions yet</p>
            <button
              onClick={() => navigate('/new')}
              className="text-blue-600 hover:text-blue-700 underline"
            >
              Start your first research
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {sessions?.map((session) => (
              <div
                key={session.session_id}
                onClick={() => navigate(`/session/${session.session_id}`)}
                className="p-6 bg-white rounded-lg shadow hover:shadow-lg cursor-pointer transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {session.query}
                    </h3>
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <span>
                        {formatDistanceToNow(new Date(session.created_at), {
                          addSuffix: true,
                        })}
                      </span>
                      <span>•</span>
                      <span className="font-mono text-xs">
                        {session.session_id}
                      </span>
                    </div>
                  </div>
                  {getStatusBadge(session.status)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
