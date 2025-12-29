/**
 * React Context for session state management using useReducer
 */

import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { ToolCallEvent, SubagentSpawn, SessionStatus } from '../types';

interface SessionState {
  sessionId: string | null;
  status: SessionStatus;
  query: string;
  subagents: SubagentSpawn[];
  toolCalls: ToolCallEvent[];
  activeAgents: Set<string>;
  currentPhase: string | null;
  lastActivityTime: Map<string, number>; // Track last activity timestamp per agent
}

type SessionAction =
  | { type: 'START_SESSION'; payload: { sessionId: string; query: string } }
  | { type: 'UPDATE_STATUS'; payload: SessionStatus }
  | { type: 'ADD_SUBAGENT'; payload: SubagentSpawn }
  | { type: 'ADD_TOOL_CALL'; payload: ToolCallEvent }
  | { type: 'SET_ACTIVE_AGENT'; payload: string }
  | { type: 'REMOVE_ACTIVE_AGENT'; payload: string }
  | { type: 'UPDATE_PHASE'; payload: string }
  | { type: 'RESET_SESSION' };

const initialState: SessionState = {
  sessionId: null,
  status: SessionStatus.PENDING,
  query: '',
  subagents: [],
  toolCalls: [],
  activeAgents: new Set(),
  currentPhase: null,
  lastActivityTime: new Map(),
};

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case 'START_SESSION':
      return {
        ...initialState,
        sessionId: action.payload.sessionId,
        query: action.payload.query,
        status: SessionStatus.RUNNING,
      };

    case 'UPDATE_STATUS':
      return {
        ...state,
        status: action.payload,
      };

    case 'ADD_SUBAGENT': {
      // Check if subagent already exists
      const exists = state.subagents.some(
        (s) => s.agent_id === action.payload.agent_id
      );
      if (exists) {
        return state;
      }

      return {
        ...state,
        subagents: [...state.subagents, action.payload],
        activeAgents: new Set(state.activeAgents).add(action.payload.agent_id),
      };
    }

    case 'ADD_TOOL_CALL': {
      // Update last activity time for this agent
      const newLastActivityTime = new Map(state.lastActivityTime);
      newLastActivityTime.set(action.payload.agent_id, Date.now());

      return {
        ...state,
        toolCalls: [...state.toolCalls, action.payload],
        lastActivityTime: newLastActivityTime,
      };
    }

    case 'SET_ACTIVE_AGENT': {
      const newActive = new Set(state.activeAgents);
      newActive.add(action.payload);

      // Update last activity time
      const newLastActivityTime = new Map(state.lastActivityTime);
      newLastActivityTime.set(action.payload, Date.now());

      return {
        ...state,
        activeAgents: newActive,
        lastActivityTime: newLastActivityTime,
      };
    }

    case 'REMOVE_ACTIVE_AGENT': {
      const newActive = new Set(state.activeAgents);
      newActive.delete(action.payload);
      return {
        ...state,
        activeAgents: newActive,
      };
    }

    case 'UPDATE_PHASE':
      return {
        ...state,
        currentPhase: action.payload,
      };

    case 'RESET_SESSION':
      return initialState;

    default:
      return state;
  }
}

interface SessionContextValue {
  state: SessionState;
  dispatch: React.Dispatch<SessionAction>;
}

const SessionContext = createContext<SessionContextValue | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sessionReducer, initialState);

  return (
    <SessionContext.Provider value={{ state, dispatch }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within SessionProvider');
  }
  return context;
}
