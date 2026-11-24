import type {
  AgentRuntimeTelemetry,
  OfficeEvent,
  WorkflowOverviewResponse,
} from '../types/workplace';
import type { DimensionViewMode } from './workplaceStore';

export function buildSimulationWaves(
  overview: WorkflowOverviewResponse,
  activeDimension: DimensionViewMode
): string[][] {
  const agents = overview.agents || [];
  const byId = new Map(agents.map((a) => [a.agent_id, a]));
  const order = overview.execution_order?.length
    ? overview.execution_order.filter((id) => byId.has(id))
    : agents.map((a) => a.agent_id);
  for (const a of agents) {
    if (!order.includes(a.agent_id)) order.push(a.agent_id);
  }

  const allowParallel =
    activeDimension === 'workflow_parallelization' || activeDimension === 'all_combined';
  if (!allowParallel) {
    return order.map((id) => [id]);
  }

  const completed = new Set<string>();
  const remaining = [...order];
  const waves: string[][] = [];
  while (remaining.length > 0) {
    const ready = remaining.filter((id) =>
      (byId.get(id)?.dependencies || []).every((dep) => completed.has(dep) || !byId.has(dep))
    );
    const wave = ready.length > 0 ? ready : [remaining[0]];
    waves.push(wave);
    for (const w of wave) {
      completed.add(w);
      const idx = remaining.indexOf(w);
      if (idx >= 0) remaining.splice(idx, 1);
    }
  }
  return waves;
}

export function extractExecutionMap(
  trace: Record<string, any> | null,
  overview: WorkflowOverviewResponse
): Record<string, Record<string, any>> {
  const map: Record<string, Record<string, any>> = {};
  if (!trace || !Array.isArray(trace.agent_executions)) return map;
  const order = overview.execution_order?.length
    ? overview.execution_order
    : overview.agents.map((a) => a.agent_id);
  trace.agent_executions.forEach((ex: any, idx: number) => {
    if (!ex || typeof ex !== 'object') return;
    const aid = ex.agent_id || order[idx];
    if (aid) map[String(aid)] = ex;
  });
  return map;
}

export function computeSimulationStep(params: {
  overview: WorkflowOverviewResponse;
  activeTrace: Record<string, any> | null;
  activeDimension: DimensionViewMode;
  simTraceIndex: number;
  nextStep: number;
  baseTelemetry: Record<string, AgentRuntimeTelemetry>;
}): {
  waves: string[][];
  nextTelemetry: Record<string, AgentRuntimeTelemetry>;
  replayEvents: OfficeEvent[];
} {
  const { overview, activeTrace, activeDimension, simTraceIndex, nextStep, baseTelemetry } = params;
  const waves = buildSimulationWaves(overview, activeDimension);
  const execMap = extractExecutionMap(activeTrace, overview);
  const nextTelemetry = { ...baseTelemetry };
  const replayEvents: OfficeEvent[] = [];
  const nowIso = new Date().toISOString();

  for (let w = 0; w < waves.length; w++) {
    const wave = waves[w];
    const isParallelWave = wave.length > 1;
    for (const aid of wave) {
      const rec = execMap[aid];
      const dur = typeof rec?.duration_seconds === 'number' ? rec.duration_seconds : null;
      const tok = typeof rec?.tokens === 'number' ? rec.tokens : null;
      const outPreview = rec?.output
        ? typeof rec.output === 'string'
          ? rec.output.slice(0, 600)
          : JSON.stringify(rec.output, null, 2).slice(0, 600)
        : null;

      if (w < nextStep) {
        nextTelemetry[aid] = {
          ...nextTelemetry[aid],
          state: 'completed',
          latency_seconds: dur ?? nextTelemetry[aid].latency_seconds,
          tokens_used: tok ?? nextTelemetry[aid].tokens_used,
          output_preview: outPreview,
          current_task: dur !== null ? `Completed (${dur.toFixed(2)}s • ${tok ?? '?'} tok)` : 'Completed stage',
          speech_bubble: dur !== null ? `Done ${dur.toFixed(1)}s` : 'Done',
        };
      } else if (w === nextStep) {
        nextTelemetry[aid] = {
          ...nextTelemetry[aid],
          state: isParallelWave ? 'parallel_working' : 'working',
          parallel_group: isParallelWave ? w + 1 : null,
          output_preview: outPreview,
          current_task: isParallelWave ? `Concurrent wave #${w + 1}` : 'Executing recorded trace step',
          speech_bubble: isParallelWave ? 'SYNC//' : 'Running',
        };
      }
    }

    if (w === nextStep) {
      replayEvents.push({
        event_id: `sim_${simTraceIndex}_${nextStep}_${wave.join('_')}`,
        experiment_id: String(activeTrace?.run_id || 'historical_sim'),
        timestamp: nowIso,
        event_type: isParallelWave ? 'agent_parallel_started' : 'agent_started',
        agent_id: isParallelWave ? null : wave[0],
        payload: {
          summary: `[Trace Sim #${simTraceIndex + 1} Step ${nextStep + 1}/${waves.length}] ${
            isParallelWave ? `Parallel wave: ${wave.join(', ')}` : `Executing ${wave[0]}`
          } (0 Groq calls)`,
        },
      });
    }
  }

  if (nextStep === waves.length) {
    replayEvents.push({
      event_id: `sim_done_${simTraceIndex}_${waves.length}`,
      experiment_id: String(activeTrace?.run_id || 'historical_sim'),
      timestamp: nowIso,
      event_type: 'experiment_completed',
      agent_id: null,
      payload: {
        summary: `[Trace Sim #${simTraceIndex + 1}] Complete (${
          activeTrace?.duration_seconds ? `${Number(activeTrace.duration_seconds).toFixed(2)}s` : 'all stages'
        }, 0 Groq calls)`,
      },
    });
  }

  return { waves, nextTelemetry, replayEvents };
}
