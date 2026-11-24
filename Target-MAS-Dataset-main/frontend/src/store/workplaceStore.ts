import { create } from 'zustand';
import { workplaceApi } from '../services/api';
import { soundEngine } from '../services/sound';
import { clearSocketResources, connectExperimentSocket } from './socketClient';
import { buildSimulationWaves, computeSimulationStep } from './traceSimulator';
import type {
  AgentProfile,
  AgentRuntimeTelemetry,
  AgentVisualState,
  BaselineResponse,
  ExperimentResultsResponse,
  ExperimentStartRequest,
  ExperimentStatusResponse,
  NavTab,
  OfficeEvent,
  OptimizationDimension,
  ReportBundleResponse,
  WorkflowOverviewResponse,
  WorkflowSummary,
} from '../types/workplace';

export { buildSimulationWaves };

export type DimensionViewMode = 'baseline' | OptimizationDimension | 'all_combined';

let simTimer: number | null = null;

function clearSimTimer() {
  if (simTimer !== null) {
    window.clearInterval(simTimer);
    simTimer = null;
  }
}

export function buildInitialTelemetry(
  agents: AgentProfile[],
  baselineModel: string
): Record<string, AgentRuntimeTelemetry> {
  const out: Record<string, AgentRuntimeTelemetry> = {};
  for (const agent of agents) {
    out[agent.agent_id] = {
      agent_id: agent.agent_id,
      state: 'idle',
      current_task: 'Standing by at workstation',
      model_used: agent.model || baselineModel,
      latency_seconds: agent.historical_mean_latency_seconds ?? null,
      tokens_used: agent.historical_mean_tokens ?? null,
      cost_usd: agent.historical_mean_cost_usd ?? null,
      error_message: null,
      parallel_group: null,
      removed_fields: [],
      retained_fields: [],
      consolidated_into: null,
      prompt_compressed_pct: null,
      output_preview: null,
    };
  }
  return out;
}

interface WorkplaceState {
  workflows: WorkflowSummary[];
  selectedWorkflowId: string;
  overview: WorkflowOverviewResponse | null;
  baseline: BaselineResponse | null;
  experiments: ExperimentStatusResponse[];
  activeExperimentId: string | null;
  activeExperimentStatus: ExperimentStatusResponse | null;
  activeExperimentResults: ExperimentResultsResponse | null;
  activeReportBundle: ReportBundleResponse | null;
  events: OfficeEvent[];
  agentTelemetry: Record<string, AgentRuntimeTelemetry>;
  activeTab: NavTab;
  activeDimension: DimensionViewMode;
  comparisonMode: boolean;
  selectedAgentId: string | null;
  wsConnected: boolean;
  soundMuted: boolean;
  loading: boolean;
  startingExperiment: boolean;
  error: string | null;
  simPlaying: boolean;
  simTraceIndex: number;
  simStepIndex: number;

  initialize: () => Promise<void>;
  selectWorkflow: (workflowId: string) => Promise<void>;
  selectExperiment: (experimentId: string) => Promise<void>;
  startExperiment: (req: ExperimentStartRequest) => Promise<ExperimentStatusResponse | null>;
  applyEvent: (ev: OfficeEvent) => void;
  setActiveTab: (tab: NavTab) => void;
  setActiveDimension: (dim: DimensionViewMode) => void;
  setComparisonMode: (enabled: boolean) => void;
  setSelectedAgentId: (agentId: string | null) => void;
  toggleSoundMuted: () => void;
  clearError: () => void;
  disconnectSocket: () => void;
  startTraceSimulation: () => void;
  pauseTraceSimulation: () => void;
  stepTraceSimulation: () => void;
  resetTraceSimulation: () => void;
  selectSimTraceIndex: (idx: number) => void;
}

function openLiveSocket(
  experimentId: string,
  get: () => WorkplaceState,
  set: (partial: Partial<WorkplaceState>) => void
) {
  connectExperimentSocket({
    experimentId,
    onStatusReconciled: (status) => set({ activeExperimentStatus: status }),
    onEvent: (ev) => get().applyEvent(ev),
    onConnectionChange: (wsConnected) => set({ wsConnected }),
    shouldReconnect: () => {
      const st = get().activeExperimentStatus;
      return (
        get().activeExperimentId === experimentId &&
        (st?.status === 'running' || st?.status === 'queued')
      );
    },
  });
}

export const useWorkplaceStore = create<WorkplaceState>((set, get) => ({
  workflows: [],
  selectedWorkflowId: '',
  overview: null,
  baseline: null,
  experiments: [],
  activeExperimentId: null,
  activeExperimentStatus: null,
  activeExperimentResults: null,
  activeReportBundle: null,
  events: [],
  agentTelemetry: {},
  activeTab: 'office',
  activeDimension: 'all_combined',
  comparisonMode: false,
  selectedAgentId: null,
  wsConnected: false,
  soundMuted: true,
  loading: false,
  startingExperiment: false,
  error: null,
  simPlaying: false,
  simTraceIndex: 0,
  simStepIndex: -1,

  initialize: async () => {
    set({ loading: true, error: null });
    try {
      const workflows = await workplaceApi.listWorkflows();
      set({ workflows });
      if (workflows.length === 0) {
        set({ loading: false });
        return;
      }
      const preferred =
        workflows.find((w) => w.workflow_id === 'recruitment_screening')?.workflow_id ||
        workflows[0].workflow_id;
      await get().selectWorkflow(preferred);
    } catch (err: any) {
      set({
        loading: false,
        error: err?.message || 'Failed to connect to Agentic Workplace API.',
      });
    }
  },

  selectWorkflow: async (workflowId: string) => {
    clearSocketResources();
    clearSimTimer();
    set({
      loading: true,
      error: null,
      selectedWorkflowId: workflowId,
      selectedAgentId: null,
      activeExperimentId: null,
      activeExperimentStatus: null,
      activeExperimentResults: null,
      activeReportBundle: null,
      events: [],
      wsConnected: false,
      simPlaying: false,
      simTraceIndex: 0,
      simStepIndex: -1,
    });
    try {
      const [overview, baseline, experiments] = await Promise.all([
        workplaceApi.getWorkflowOverview(workflowId),
        workplaceApi.getWorkflowBaseline(workflowId, 10),
        workplaceApi.listExperiments(workflowId),
      ]);
      set({
        overview,
        baseline,
        experiments,
        agentTelemetry: buildInitialTelemetry(overview.agents, overview.baseline_model),
        loading: false,
      });

      if (experiments.length > 0) {
        const preferredExp =
          experiments.find((e) => e.status === 'running') ||
          experiments.find((e) => e.experiment_id.includes('live_validation')) ||
          experiments[0];
        await get().selectExperiment(preferredExp.experiment_id);
      }
    } catch (err: any) {
      set({
        loading: false,
        error: err?.message || `Failed to load workflow ${workflowId}.`,
      });
    }
  },

  selectExperiment: async (experimentId: string) => {
    clearSocketResources();
    set({
      activeExperimentId: experimentId,
      activeExperimentResults: null,
      activeReportBundle: null,
      events: [],
      wsConnected: false,
    });
    try {
      const status = await workplaceApi.getExperimentStatus(experimentId);
      set({ activeExperimentStatus: status });

      const overview = get().overview;
      if (overview) {
        set({
          agentTelemetry: buildInitialTelemetry(overview.agents, overview.baseline_model),
        });
      }

      const evs = await workplaceApi.getExperimentEvents(experimentId).catch(() => []);
      for (const ev of evs) get().applyEvent(ev);

      if (status.has_results) {
        const [results, reports] = await Promise.all([
          workplaceApi.getExperimentResults(experimentId).catch(() => null),
          workplaceApi.getReportBundle(experimentId).catch(() => null),
        ]);
        set({ activeExperimentResults: results, activeReportBundle: reports });
      }

      if (status.status === 'running' || status.status === 'queued') {
        openLiveSocket(experimentId, get, set);
      }
    } catch (err: any) {
      set({ error: err?.message || `Failed to load experiment ${experimentId}.` });
    }
  },

  startExperiment: async (req: ExperimentStartRequest) => {
    set({ startingExperiment: true, error: null });
    try {
      const created = await workplaceApi.startExperiment(req);
      const overview = get().overview;
      set((s) => ({
        startingExperiment: false,
        experiments: [created, ...s.experiments.filter((e) => e.experiment_id !== created.experiment_id)],
        activeExperimentId: created.experiment_id,
        activeExperimentStatus: created,
        activeExperimentResults: null,
        activeReportBundle: null,
        events: [],
        agentTelemetry: overview
          ? buildInitialTelemetry(overview.agents, overview.baseline_model)
          : s.agentTelemetry,
      }));
      openLiveSocket(created.experiment_id, get, set);
      return created;
    } catch (err: any) {
      set({
        startingExperiment: false,
        error: err?.message || 'Failed to start experiment.',
      });
      return null;
    }
  },

  applyEvent: (ev: OfficeEvent) => {
    if (!ev || ev.event_type === 'heartbeat') return;
    const state = get();
    if (state.events.some((existing) => existing.event_id === ev.event_id)) return;

    soundEngine.playEventCue(ev.event_type);

    const nextEvents = [...state.events, ev].slice(-250);
    const nextTelemetry = { ...state.agentTelemetry };
    const aid = ev.agent_id || (ev.payload?.agent_id as string | undefined);

    const ensureEntry = (id: string): AgentRuntimeTelemetry =>
      nextTelemetry[id] ?? { agent_id: id, state: 'idle', current_task: 'Active in office' };

    if (ev.event_type === 'agent_parallel_started' && Array.isArray(ev.payload?.agent_ids)) {
      const groupIdx = Number(ev.payload?.group_index ?? 1);
      for (const pid of ev.payload.agent_ids as string[]) {
        nextTelemetry[pid] = {
          ...ensureEntry(pid),
          state: 'parallel_working',
          parallel_group: groupIdx,
          current_task: `Parallel stage ${groupIdx}`,
          speech_bubble: `Parallel stage ${groupIdx}`,
        };
      }
    } else if (aid) {
      const cur = ensureEntry(aid);
      const stateMap: Record<string, AgentVisualState> = {
        agent_started: 'working',
        groq_request_started: 'working',
        agent_thinking: 'thinking',
        agent_completed: 'completed',
        groq_request_completed: 'completed',
        agent_failed: 'failed',
      };
      nextTelemetry[aid] = {
        ...cur,
        state: stateMap[ev.event_type] ?? cur.state,
        model_used: (ev.payload?.model as string) || cur.model_used,
        latency_seconds:
          typeof ev.payload?.latency_seconds === 'number'
            ? ev.payload.latency_seconds
            : cur.latency_seconds,
        tokens_used:
          typeof ev.payload?.total_tokens === 'number' ? ev.payload.total_tokens : cur.tokens_used,
        error_message:
          ev.event_type === 'agent_failed'
            ? String(ev.payload?.error || 'Execution error')
            : cur.error_message,
        current_task:
          (ev.payload?.summary as string) ||
          (ev.event_type === 'groq_request_started'
            ? `Groq inference (${ev.payload?.model || 'live'})`
            : ev.event_type === 'agent_completed'
            ? `Finished (${ev.payload?.latency_seconds ?? '?'}s)`
            : ev.event_type === 'agent_failed'
            ? `Failed: ${ev.payload?.error || 'error'}`
            : cur.current_task),
        speech_bubble:
          ev.event_type === 'groq_request_started'
            ? `Calling Groq (${ev.payload?.model || ''})`
            : ev.event_type === 'agent_completed'
            ? `Done in ${ev.payload?.latency_seconds ?? '?'}s`
            : ev.event_type === 'agent_failed'
            ? 'Run blocked!'
            : cur.speech_bubble,
      };
    }

    set({ events: nextEvents, agentTelemetry: nextTelemetry });

    if (ev.event_type === 'experiment_completed' || ev.event_type === 'experiment_failed') {
      const expId = ev.experiment_id;
      void (async () => {
        try {
          const [status, results, reports, exps] = await Promise.all([
            workplaceApi.getExperimentStatus(expId),
            workplaceApi.getExperimentResults(expId).catch(() => null),
            workplaceApi.getReportBundle(expId).catch(() => null),
            workplaceApi.listExperiments(get().selectedWorkflowId || undefined).catch(() => get().experiments),
          ]);
          if (get().activeExperimentId === expId) {
            set({
              activeExperimentStatus: status,
              activeExperimentResults: results,
              activeReportBundle: reports,
              experiments: exps,
            });
          }
        } catch {
          // status refresh will recover
        }
      })();
    }
  },

  setActiveTab: (activeTab) => set({ activeTab }),
  setActiveDimension: (activeDimension) => set({ activeDimension }),
  setComparisonMode: (comparisonMode) => set({ comparisonMode }),
  setSelectedAgentId: (selectedAgentId) => set({ selectedAgentId }),
  toggleSoundMuted: () =>
    set((s) => {
      const nextMuted = !s.soundMuted;
      soundEngine.setMuted(nextMuted);
      return { soundMuted: nextMuted };
    }),
  clearError: () => set({ error: null }),
  disconnectSocket: () => {
    clearSocketResources();
    clearSimTimer();
    set({ wsConnected: false, simPlaying: false });
  },

  selectSimTraceIndex: (idx: number) => {
    clearSimTimer();
    const overview = get().overview;
    set({
      simPlaying: false,
      simTraceIndex: Math.max(0, idx),
      simStepIndex: -1,
      agentTelemetry: overview ? buildInitialTelemetry(overview.agents, overview.baseline_model) : {},
    });
  },

  resetTraceSimulation: () => {
    clearSimTimer();
    const overview = get().overview;
    set({
      simPlaying: false,
      simStepIndex: -1,
      agentTelemetry: overview ? buildInitialTelemetry(overview.agents, overview.baseline_model) : {},
    });
  },

  pauseTraceSimulation: () => {
    clearSimTimer();
    set({ simPlaying: false });
  },

  stepTraceSimulation: () => {
    const state = get();
    const overview = state.overview;
    if (!overview || overview.agents.length === 0) return;

    const traces =
      (state.baseline?.traces && state.baseline.traces.length > 0
        ? state.baseline.traces
        : state.baseline?.replay_traces) || [];
    const activeTrace = traces[state.simTraceIndex] || null;
    const nextStep = state.simStepIndex + 1;
    const baseTelemetry = buildInitialTelemetry(overview.agents, overview.baseline_model);
    const { waves, nextTelemetry, replayEvents } = computeSimulationStep({
      overview,
      activeTrace,
      activeDimension: state.activeDimension,
      simTraceIndex: state.simTraceIndex,
      nextStep,
      baseTelemetry,
    });

    if (nextStep > waves.length) {
      clearSimTimer();
      set({ simPlaying: false });
      return;
    }
    if (nextStep === waves.length) clearSimTimer();

    const deduped = [...state.events];
    for (const ev of replayEvents) {
      if (!deduped.some((x) => x.event_id === ev.event_id)) deduped.push(ev);
    }

    set({
      simStepIndex: nextStep,
      simPlaying: nextStep >= waves.length ? false : state.simPlaying,
      agentTelemetry: nextTelemetry,
      events: deduped.slice(-250),
    });
  },

  startTraceSimulation: () => {
    clearSimTimer();
    const state = get();
    const waves = state.overview ? buildSimulationWaves(state.overview, state.activeDimension) : [];
    if (state.simStepIndex >= waves.length) get().resetTraceSimulation();
    set({ simPlaying: true });
    get().stepTraceSimulation();
    simTimer = window.setInterval(() => {
      if (!get().simPlaying) {
        clearSimTimer();
        return;
      }
      get().stepTraceSimulation();
    }, 1150);
  },
}));
