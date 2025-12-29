/**
 * Custom React hook for WebSocket connection management
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { ToolCallEvent } from '../types';

interface UseWebSocketOptions {
  sessionId: string;
  onMessage?: (event: ToolCallEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}

export function useWebSocket({
  sessionId,
  onMessage,
  onOpen,
  onClose,
  onError,
}: UseWebSocketOptions) {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<ToolCallEvent | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  // Use refs for callbacks to avoid reconnection loops
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
  }, [onMessage, onOpen, onClose, onError]);

  const connect = useCallback(() => {
    const wsUrl = `ws://localhost:8000/ws/${sessionId}`;
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      setIsConnected(true);
      onOpenRef.current?.();

      // Start keepalive ping
      pingIntervalRef.current = window.setInterval(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send('ping');
        }
      }, 30000); // Every 30 seconds
    };

    ws.current.onmessage = (event) => {
      // Handle pong response
      if (event.data === 'pong') {
        return;
      }

      try {
        const data: ToolCallEvent = JSON.parse(event.data);
        setLastEvent(data);
        onMessageRef.current?.(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      onErrorRef.current?.(error);
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      onCloseRef.current?.();

      // Clear ping interval
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
    };
  }, [sessionId]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [sessionId]); // Only reconnect if sessionId changes

  return {
    isConnected,
    lastEvent,
    reconnect: connect,
    disconnect,
  };
}
