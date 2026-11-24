import type {
  AgentProfile,
  AgentRuntimeTelemetry,
  ExperimentResultsResponse,
  WorkflowOverviewResponse,
} from '../types/workplace';
import type { DimensionViewMode } from '../store/workplaceStore';
import {
  classifyModelTier,
  resolveDimensionVisuals,
  type HandoffEdgeVisual,
} from './dimensionMapper';

export type { HandoffEdgeVisual };

export interface DeskPlacement {
  agent: AgentProfile;
  x: number;
  y: number;
  chairX: number;
  chairY: number;
  stageIndex: number;
  isManager: boolean;
  parallelGroup: number | null;
  removedFields: string[];
  retainedFields: string[];
  modelAssigned: string;
  modelTier: 'heavy' | 'medium' | 'light';
  consolidatedInto: string | null;
  absorbedAgents: string[];
  promptCompressedPct: number | null;
  telemetry: AgentRuntimeTelemetry;
}

export interface ParallelClusterVisual {
  groupIndex: number;
  agentIds: string[];
  bounds: { x: number; y: number; w: number; h: number };
}

export interface StationRegion {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface OfficeLayoutModel {
  width: number;
  height: number;
  modeLabel: string;
  stations: StationRegion[];
  desks: DeskPlacement[];
  handoffEdges: HandoffEdgeVisual[];
  parallelClusters: ParallelClusterVisual[];
  consolidatedSuccess: boolean;
}

const OFFICE_STATIONS: StationRegion[] = [
  { id: 'entrance', label: 'ENTRANCE & INBOX', x: 20, y: 26, w: 150, h: 100 },
  { id: 'task_board', label: 'TASK BOARD', x: 186, y: 26, w: 180, h: 100 },
  { id: 'model_rack', label: 'MODEL RACK', x: 382, y: 26, w: 170, h: 100 },
  { id: 'doc_station', label: 'FILING & PRUNER', x: 568, y: 26, w: 165, h: 100 },
  { id: 'manager_office', label: 'FINAL REVIEW DESK', x: 748, y: 26, w: 192, h: 138 },
  { id: 'meeting_room', label: 'MEETING & SYNC ROOM', x: 695, y: 392, w: 245, h: 186 },
];

export function computeTopologicalStages(
  agents: AgentProfile[],
  executionOrder: string[]
): Map<string, number> {
  const agentMap = new Map(agents.map((a) => [a.agent_id, a]));
  const stages = new Map<string, number>();

  const visit = (id: string, visiting = new Set<string>()): number => {
    if (stages.has(id)) return stages.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const agent = agentMap.get(id);
    if (!agent?.dependencies?.length) {
      stages.set(id, 0);
      return 0;
    }
    let maxDep = -1;
    for (const dep of agent.dependencies) {
      if (agentMap.has(dep)) maxDep = Math.max(maxDep, visit(dep, visiting));
    }
    const lvl = maxDep + 1;
    stages.set(id, lvl);
    return lvl;
  };

  const orderedIds = executionOrder.length > 0 ? executionOrder : agents.map((a) => a.agent_id);
  for (const id of orderedIds) if (agentMap.has(id)) visit(id);
  for (const a of agents) if (!stages.has(a.agent_id)) visit(a.agent_id);
  return stages;
}

export function buildOfficeLayout(params: {
  overview: WorkflowOverviewResponse | null;
  results: ExperimentResultsResponse | null;
  telemetry: Record<string, AgentRuntimeTelemetry>;
  activeDimension: DimensionViewMode;
  side?: 'baseline' | 'optimized';
}): OfficeLayoutModel {
  const { overview, results, telemetry, activeDimension, side = 'optimized' } = params;
  const width = 960;
  const height = 600;
  const agents = overview?.agents ?? [];

  if (agents.length === 0) {
    return {
      width,
      height,
      modeLabel: 'NO WORKFLOW LOADED',
      stations: OFFICE_STATIONS,
      desks: [],
      handoffEdges: [],
      parallelClusters: [],
      consolidatedSuccess: false,
    };
  }

  const isBaseline = side === 'baseline' || activeDimension === 'baseline';
  const baselineModel = overview?.baseline_model || 'openai/gpt-oss-120b';
  const stages = computeTopologicalStages(agents, overview?.execution_order || []);
  const maxStage = Math.max(0, ...Array.from(stages.values()));
  const managerAgentId =
    agents.find((a) => stages.get(a.agent_id) === maxStage)?.agent_id ||
    agents[agents.length - 1].agent_id;

  const visuals = resolveDimensionVisuals({
    overview,
    results,
    stages,
    activeDimension,
    isBaseline,
  });

  const workerAgents = agents.filter((a) => a.agent_id !== managerAgentId);
  const managerAgent = agents.find((a) => a.agent_id === managerAgentId);
  const cols = Math.min(4, Math.max(2, Math.ceil(Math.sqrt(Math.max(1, workerAgents.length) * 1.4))));
  const rows = Math.max(1, Math.ceil(workerAgents.length / cols));
  const cellW = Math.floor(620 / cols);
  const cellH = Math.floor(395 / Math.max(2, rows));

  const createDesk = (agent: AgentProfile, x: number, y: number, isManager: boolean): DeskPlacement => {
    const tel = telemetry[agent.agent_id] ?? {
      agent_id: agent.agent_id,
      state: 'idle',
      current_task: isBaseline ? 'Historical snapshot' : 'Ready at desk',
    };
    const assignedModel = isBaseline
      ? baselineModel
      : visuals.assignedModels.get(agent.agent_id) || tel.model_used || agent.model || baselineModel;
    const outgoingEdge = visuals.handoffEdges.find((e) => e.fromAgentId === agent.agent_id);
    const incomingEdge = visuals.handoffEdges.find((e) => e.toAgentId === agent.agent_id);

    return {
      agent,
      x,
      y,
      chairX: x,
      chairY: y + 26,
      stageIndex: stages.get(agent.agent_id) ?? 0,
      isManager,
      parallelGroup: visuals.parallelGroupByAgent.get(agent.agent_id) ?? null,
      removedFields: visuals.showHandoff
        ? outgoingEdge?.removedFields || incomingEdge?.removedFields || []
        : [],
      retainedFields: visuals.showHandoff
        ? outgoingEdge?.retainedFields || incomingEdge?.retainedFields || []
        : agent.input_fields || [],
      modelAssigned: assignedModel,
      modelTier: classifyModelTier(assignedModel),
      consolidatedInto: visuals.consolidatedIntoMap.get(agent.agent_id) ?? null,
      absorbedAgents: visuals.absorbedByMap.get(agent.agent_id) ?? [],
      promptCompressedPct: visuals.promptPctMap.get(agent.agent_id) ?? null,
      telemetry: {
        ...tel,
        state: isBaseline && !tel.speech_bubble ? 'idle' : tel.state,
        model_used: assignedModel,
      },
    };
  };

  const desks: DeskPlacement[] = workerAgents.map((agent, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    return createDesk(
      agent,
      44 + col * cellW + Math.floor(cellW / 2),
      165 + row * cellH + Math.floor(cellH / 2),
      false
    );
  });
  if (managerAgent) desks.push(createDesk(managerAgent, 844, 112, true));

  const parallelClusters: ParallelClusterVisual[] = [];
  const grouped = new Map<number, DeskPlacement[]>();
  for (const d of desks) {
    if (d.parallelGroup !== null) {
      if (!grouped.has(d.parallelGroup)) grouped.set(d.parallelGroup, []);
      grouped.get(d.parallelGroup)!.push(d);
    }
  }
  for (const [groupIndex, members] of grouped.entries()) {
    if (members.length < 2) continue;
    const minX = Math.min(...members.map((m) => m.x)) - 66;
    const maxX = Math.max(...members.map((m) => m.x)) + 66;
    const minY = Math.min(...members.map((m) => m.y)) - 54;
    const maxY = Math.max(...members.map((m) => m.y)) + 54;
    parallelClusters.push({
      groupIndex,
      agentIds: members.map((m) => m.agent.agent_id),
      bounds: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
    });
  }

  return {
    width,
    height,
    modeLabel: isBaseline
      ? 'ORIGINAL TARGET MAS (HISTORICAL SNAPSHOT)'
      : `OPTIMIZED VIEW (${activeDimension.replace(/_/g, ' ').toUpperCase()})`,
    stations: OFFICE_STATIONS,
    desks,
    handoffEdges: visuals.handoffEdges,
    parallelClusters,
    consolidatedSuccess: visuals.showConsolidation && visuals.consolidatedIntoMap.size > 0,
  };
}
