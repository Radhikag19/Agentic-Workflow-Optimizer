import type {
  AgentProfile,
  BaselineResponse,
  ExperimentResultsResponse,
  ExperimentStartRequest,
  ExperimentStatusResponse,
  OfficeEvent,
  ReportBundleResponse,
  WorkflowOverviewResponse,
  WorkflowSummary,
} from '../types/workplace';

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // keep default HTTP status message
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const workplaceApi = {
  getHealth: () => requestJson<{ status: string; service: string }>('/api/health'),
  listWorkflows: () => requestJson<WorkflowSummary[]>('/api/workflows'),
  getWorkflowOverview: (workflowId: string) =>
    requestJson<WorkflowOverviewResponse>(`/api/workflows/${encodeURIComponent(workflowId)}/overview`),
  getWorkflowBaseline: (workflowId: string, maxTraces = 10) =>
    requestJson<BaselineResponse>(
      `/api/workflows/${encodeURIComponent(workflowId)}/baseline?max_traces=${maxTraces}`
    ),
  getWorkflowAgents: (workflowId: string) =>
    requestJson<AgentProfile[]>(`/api/agents/${encodeURIComponent(workflowId)}`),
  listExperiments: (workflowId?: string) =>
    requestJson<ExperimentStatusResponse[]>(
      workflowId ? `/api/experiments?workflow_id=${encodeURIComponent(workflowId)}` : '/api/experiments'
    ),
  startExperiment: (payload: ExperimentStartRequest) =>
    requestJson<ExperimentStatusResponse>('/api/experiments/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getExperimentStatus: (experimentId: string) =>
    requestJson<ExperimentStatusResponse>(`/api/experiments/${encodeURIComponent(experimentId)}`),
  getExperimentResults: (experimentId: string) =>
    requestJson<ExperimentResultsResponse>(
      `/api/experiments/${encodeURIComponent(experimentId)}/results`
    ),
  getExperimentEvents: (experimentId: string) =>
    requestJson<OfficeEvent[]>(`/api/experiments/${encodeURIComponent(experimentId)}/events`),
  getReportBundle: (experimentId: string) =>
    requestJson<ReportBundleResponse>(`/api/reports/${encodeURIComponent(experimentId)}`),
};

export function buildExperimentWebSocketUrl(experimentId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/experiments/${encodeURIComponent(experimentId)}`;
}
