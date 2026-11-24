import type { ExperimentResultsResponse, WorkflowOverviewResponse } from '../types/workplace';
import type { DimensionViewMode } from '../store/workplaceStore';

export interface HandoffEdgeVisual {
  fromAgentId: string;
  toAgentId: string;
  retainedFields: string[];
  removedFields: string[];
  tokenReductionPct: number | null;
}

export interface DerivedDimensionVisuals {
  showHandoff: boolean;
  showParallel: boolean;
  showConsolidation: boolean;
  handoffEdges: HandoffEdgeVisual[];
  parallelGroupByAgent: Map<string, number>;
  assignedModels: Map<string, string>;
  consolidatedIntoMap: Map<string, string>;
  absorbedByMap: Map<string, string[]>;
  promptPctMap: Map<string, number>;
}

export function classifyModelTier(modelName: string): 'heavy' | 'medium' | 'light' {
  const m = (modelName || '').toLowerCase();
  if (m.includes('8b') || m.includes('instant') || m.includes('mini') || m.includes('gemma2-9b')) {
    return 'light';
  }
  if (m.includes('120b') || m.includes('70b') || m.includes('versatile') || m.includes('405b')) {
    return 'heavy';
  }
  return 'medium';
}

export function getDimensionCandidateList(
  overview: WorkflowOverviewResponse | null,
  results: ExperimentResultsResponse | null,
  dim: string
): Record<string, any>[] {
  const fromApproved = (overview?.approved_candidates || []).filter(
    (c) => (c.optimization_type || c.dimension) === dim
  );
  if (fromApproved.length > 0) return fromApproved;

  const fromResults =
    results?.phase_a_foundation?.[dim]?.applied_candidates ||
    results?.phase_b_variable_dimensions?.[dim]?.applied_candidates ||
    results?.per_dimension_results?.[dim]?.applied_candidates ||
    [];
  if (Array.isArray(fromResults) && fromResults.length > 0) return fromResults;

  return overview?.discovered_candidates_by_dimension?.[dim] || [];
}

export function isDimensionAcceptedOrActive(
  results: ExperimentResultsResponse | null,
  dim: string
): boolean {
  if (!results) return true;
  const phaseA =
    results.phase_a_foundation?.dimension_evaluations?.[dim] ||
    results.phase_a_foundation?.[dim];
  if (phaseA) {
    return Boolean(phaseA.accepted ?? phaseA.quality_accepted ?? phaseA.foundation_passed ?? false);
  }
  const phaseB =
    results.phase_b_variable_dimensions?.dimension_evaluations?.[dim] ||
    results.phase_b_variable_dimensions?.[dim];
  if (phaseB) {
    return Boolean(phaseB.accepted ?? phaseB.quality_accepted ?? phaseB.quality_gate_passed ?? false);
  }
  const perDim = results.per_dimension_results?.[dim];
  if (perDim) {
    return Boolean(perDim.quality_gate_passed ?? perDim.accepted ?? true);
  }
  return false;
}

export function resolveDimensionVisuals(params: {
  overview: WorkflowOverviewResponse | null;
  results: ExperimentResultsResponse | null;
  stages: Map<string, number>;
  activeDimension: DimensionViewMode;
  isBaseline: boolean;
}): DerivedDimensionVisuals {
  const { overview, results, stages, activeDimension, isBaseline } = params;
  const agents = overview?.agents ?? [];

  const hasHandoffCands = getDimensionCandidateList(overview, results, 'communication_data_handoff').length > 0;
  const hasParallelCands = getDimensionCandidateList(overview, results, 'workflow_parallelization').length > 0;
  const hasModelCands = getDimensionCandidateList(overview, results, 'model_selection').length > 0;
  const hasPromptCands = getDimensionCandidateList(overview, results, 'prompt_optimization').length > 0;

  const showHandoff =
    !isBaseline &&
    ((activeDimension === 'communication_data_handoff' && hasHandoffCands) ||
      (activeDimension === 'all_combined' && isDimensionAcceptedOrActive(results, 'communication_data_handoff')));

  const showParallel =
    !isBaseline &&
    ((activeDimension === 'workflow_parallelization' && (hasParallelCands || !results)) ||
      (activeDimension === 'all_combined' && isDimensionAcceptedOrActive(results, 'workflow_parallelization')));

  const showModelSelection =
    !isBaseline &&
    ((activeDimension === 'model_selection' && hasModelCands) ||
      (activeDimension === 'all_combined' && isDimensionAcceptedOrActive(results, 'model_selection')));

  const showConsolidation =
    !isBaseline &&
    (activeDimension === 'agent_consolidation' || activeDimension === 'all_combined') &&
    Boolean(
      results?.phase_b_variable_dimensions?.dimension_evaluations?.agent_consolidation?.accepted ??
        results?.phase_b_variable_dimensions?.agent_consolidation?.accepted ??
        results?.per_dimension_results?.agent_consolidation?.quality_gate_passed ??
        false
    );

  const showPromptOpt =
    !isBaseline &&
    ((activeDimension === 'prompt_optimization' && hasPromptCands) ||
      (activeDimension === 'all_combined' && isDimensionAcceptedOrActive(results, 'prompt_optimization')));

  const handoffEdges: HandoffEdgeVisual[] = [];
  const handoffCands = getDimensionCandidateList(overview, results, 'communication_data_handoff');
  if (showHandoff) {
    for (const cand of handoffCands) {
      const targetList: string[] = cand.target_agents || [];
      const sup = cand.supporting_evidence || {};
      const meta = cand.projection || cand.details || {};
      const fromId = sup.producer_agent || targetList[0];
      const toId = sup.consumer_agent || targetList[1];
      if (fromId && toId) {
        handoffEdges.push({
          fromAgentId: fromId,
          toAgentId: toId,
          retainedFields: sup.fields_retained || meta.retained_fields || cand.retained_fields || [],
          removedFields: sup.fields_excluded || meta.removed_fields || cand.removed_fields || [],
          tokenReductionPct: sup.payload_reduction_pct ?? cand.estimated_reduction_pct ?? null,
        });
      }
    }
  } else {
    for (const ag of agents) {
      for (const dep of ag.dependencies) {
        handoffEdges.push({
          fromAgentId: dep,
          toAgentId: ag.agent_id,
          retainedFields: ag.input_fields || [],
          removedFields: [],
          tokenReductionPct: null,
        });
      }
    }
  }

  const parallelGroupByAgent = new Map<string, number>();
  if (showParallel) {
    const parCands = getDimensionCandidateList(overview, results, 'workflow_parallelization');
    let groupCounter = 1;
    for (const cand of parCands) {
      const targets: string[] = cand.target_agents || [];
      if (targets.length > 1) {
        for (const tid of targets) parallelGroupByAgent.set(tid, groupCounter);
        groupCounter += 1;
      }
    }
    if (parallelGroupByAgent.size === 0) {
      const byDepth = new Map<number, string[]>();
      for (const ag of agents) {
        const d = stages.get(ag.agent_id) ?? 0;
        if (!byDepth.has(d)) byDepth.set(d, []);
        byDepth.get(d)!.push(ag.agent_id);
      }
      for (const [, ids] of byDepth.entries()) {
        if (ids.length > 1) {
          for (const id of ids) parallelGroupByAgent.set(id, groupCounter);
          groupCounter += 1;
        }
      }
    }
  }

  const assignedModels = new Map<string, string>();
  if (showModelSelection) {
    for (const cand of getDimensionCandidateList(overview, results, 'model_selection')) {
      const proposed: string[] = cand.supporting_evidence?.proposed_candidate_models || [];
      const fallback = proposed.length > 0 ? proposed[proposed.length - 1] : undefined;
      const m = cand.recommended_model ?? cand.target_model ?? cand.new_model ?? fallback;
      for (const tid of cand.target_agents || []) {
        if (m) assignedModels.set(tid, String(m));
      }
    }
  }

  const consolidatedIntoMap = new Map<string, string>();
  const absorbedByMap = new Map<string, string[]>();
  if (showConsolidation) {
    for (const cand of getDimensionCandidateList(overview, results, 'agent_consolidation')) {
      const targets: string[] = cand.target_agents || [];
      if (targets.length >= 2) {
        const [primary, ...secondary] = targets;
        absorbedByMap.set(primary, [...(absorbedByMap.get(primary) || []), ...secondary]);
        for (const sec of secondary) consolidatedIntoMap.set(sec, primary);
      }
    }
  }

  const promptPctMap = new Map<string, number>();
  if (showPromptOpt) {
    for (const cand of getDimensionCandidateList(overview, results, 'prompt_optimization')) {
      const pct = Number(cand.estimated_reduction_pct ?? cand.expected_benefit?.token_savings_pct ?? 20);
      for (const tid of cand.target_agents || []) promptPctMap.set(tid, pct);
    }
  }

  return {
    showHandoff,
    showParallel,
    showConsolidation,
    handoffEdges,
    parallelGroupByAgent,
    assignedModels,
    consolidatedIntoMap,
    absorbedByMap,
    promptPctMap,
  };
}
