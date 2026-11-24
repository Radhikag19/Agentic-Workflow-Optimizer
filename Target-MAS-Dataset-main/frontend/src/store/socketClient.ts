import { buildExperimentWebSocketUrl, workplaceApi } from '../services/api';
import type { ExperimentStatusResponse, OfficeEvent } from '../types/workplace';

let activeSocket: WebSocket | null = null;
let reconnectTimer: number | null = null;
let reconnectAttempt = 0;

export function clearSocketResources(): void {
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (activeSocket) {
    try {
      activeSocket.onopen = null;
      activeSocket.onmessage = null;
      activeSocket.onerror = null;
      activeSocket.onclose = null;
      activeSocket.close();
    } catch {
      // ignore close errors
    }
    activeSocket = null;
  }
}

export function connectExperimentSocket(params: {
  experimentId: string;
  onStatusReconciled: (status: ExperimentStatusResponse) => void;
  onEvent: (ev: OfficeEvent) => void;
  onConnectionChange: (connected: boolean) => void;
  shouldReconnect: () => boolean;
}): void {
  const { experimentId, onStatusReconciled, onEvent, onConnectionChange, shouldReconnect } = params;
  clearSocketResources();
  if (typeof WebSocket === 'undefined') return;

  const ws = new WebSocket(buildExperimentWebSocketUrl(experimentId));
  activeSocket = ws;

  ws.onopen = async () => {
    reconnectAttempt = 0;
    onConnectionChange(true);
    try {
      const status = await workplaceApi.getExperimentStatus(experimentId);
      if (shouldReconnect()) {
        onStatusReconciled(status);
      }
    } catch {
      // ignore transient reconciliation errors
    }
  };

  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as OfficeEvent);
    } catch {
      // ignore malformed frames
    }
  };

  ws.onerror = () => {
    onConnectionChange(false);
  };

  ws.onclose = () => {
    onConnectionChange(false);
    if (shouldReconnect() && reconnectAttempt < 6) {
      const delayMs = Math.min(8000, 750 * Math.pow(2, reconnectAttempt));
      reconnectAttempt += 1;
      reconnectTimer = window.setTimeout(() => {
        connectExperimentSocket(params);
      }, delayMs);
    }
  };
}
