/**
 * Page for submitting new research queries
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { researchApi } from '../services/api';
import { ResearchInput } from '../components/ResearchInput';

export function NewResearch() {
  const navigate = useNavigate();

  const mutation = useMutation({
    mutationFn: (query: string) => researchApi.submitResearch(query),
    onSuccess: (data) => {
      navigate(`/session/${data.session_id}`);
    },
  });

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4 text-gray-900">Research Agent</h1>
        <p className="text-gray-600 text-lg">
          Enter your research query to start a comprehensive AI-powered analysis
        </p>
      </div>

      <ResearchInput
        onSubmit={(query) => mutation.mutate(query)}
        isLoading={mutation.isPending}
      />

      {mutation.isError && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg max-w-4xl">
          Error submitting research query. Please try again.
        </div>
      )}

      <div className="mt-8 text-center">
        <button
          onClick={() => navigate('/dashboard')}
          className="text-blue-600 hover:text-blue-700 underline"
        >
          View previous sessions
        </button>
      </div>
    </div>
  );
}
