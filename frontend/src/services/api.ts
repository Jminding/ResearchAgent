/**
 * API client for communicating with the backend
 */

import axios from 'axios';
import { SessionMetadata, SessionDetail, SessionStats } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const researchApi = {
  /**
   * Submit a new research query
   */
  submitResearch: async (query: string) => {
    const response = await api.post('/api/research/', { query });
    return response.data;
  },

  /**
   * Get research status
   */
  getResearchStatus: async (sessionId: string) => {
    const response = await api.get(`/api/research/status/${sessionId}`);
    return response.data;
  },

  /**
   * Cancel a running research session
   */
  cancelResearch: async (sessionId: string) => {
    const response = await api.post(`/api/research/cancel/${sessionId}`);
    return response.data;
  },
};

export const sessionsApi = {
  /**
   * List all sessions with optional filters
   */
  listSessions: async (params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }) => {
    const response = await api.get<SessionMetadata[]>('/api/sessions/', { params });
    return response.data;
  },

  /**
   * Get detailed session information
   */
  getSession: async (sessionId: string) => {
    const response = await api.get<SessionDetail>(`/api/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Get full session transcript
   */
  getTranscript: async (sessionId: string) => {
    const response = await api.get(`/api/sessions/${sessionId}/transcript`);
    return response.data;
  },

  /**
   * Get session statistics
   */
  getStats: async (sessionId: string) => {
    const response = await api.get<SessionStats>(`/api/sessions/${sessionId}/stats`);
    return response.data;
  },

  /**
   * List all PDFs for a session
   */
  listPdfs: async (sessionId: string) => {
    const response = await api.get(`/api/sessions/${sessionId}/pdfs`);
    return response.data;
  },

  /**
   * Get download URL for a PDF
   */
  getPdfUrl: (sessionId: string, filename: string) => {
    return `${API_BASE_URL}/api/sessions/${sessionId}/pdf/${filename}`;
  },
};

export default api;
