import { describe, expect, it } from 'vitest';
import { buildOfficeLayout, computeTopologicalStages } from '../officeLayout';
import type { WorkflowOverviewResponse } from '../../types/workplace';

const CONTRACT_REVIEW_OVERVIEW: WorkflowOverviewResponse = {
  workflow_id: 'contract_review',
  workflow_name: 'Contract Review Workflow',
  baseline_model: 'openai/gpt-oss-120b',
  llm_provider: 'groq',
  execution_order: ['ingest', 'clause_a', 'clause_b', 'risk_merge'],
  agents: [
    {
      agent_id: 'ingest',
      agent_name: 'Contract Ingest',
      role: 'Parser',
      responsibilities: ['Parse contract'],
      dependencies: [],
      input_fields: ['raw_contract'],
      output_fields: ['parsed_clauses'],
      model: 'openai/gpt-oss-120b',
    },
    {
      agent_id: 'clause_a',
      agent_name: 'Liability Checker',
      role: 'Analyst',
      responsibilities: ['Check liability'],
      dependencies: ['ingest'],
      input_fields: ['parsed_clauses'],
      output_fields: ['liability_score'],
      model: 'openai/gpt-oss-120b',
    },
    {
      agent_id: 'clause_b',
      agent_name: 'Compliance Checker',
      role: 'Analyst',
      responsibilities: ['Check compliance'],
      dependencies: ['ingest'],
      input_fields: ['parsed_clauses'],
      output_fields: ['compliance_score'],
      model: 'llama-3.1-8b-instant',
    },
    {
      agent_id: 'risk_merge',
      agent_name: 'Final Legal Review',
      role: 'Synthesizer',
      responsibilities: ['Synthesize legal summary'],
      dependencies: ['clause_a', 'clause_b'],
      input_fields: ['liability_score', 'compliance_score'],
      output_fields: ['final_decision'],
      model: 'openai/gpt-oss-120b',
    },
  ],
  historical_baseline: {
    source: 'historical_runtime_traces',
    re_executed: false,
    groq_api_calls: 0,
    matched_input_count: 10,
    metrics: {},
  },
  approved_candidates: [],
  discovered_candidates_by_dimension: {},
};

describe('OfficeMap dynamic layout engine', () => {
  it('computes topological DAG stages dynamically for any workflow', () => {
    const stages = computeTopologicalStages(
      CONTRACT_REVIEW_OVERVIEW.agents,
      CONTRACT_REVIEW_OVERVIEW.execution_order
    );
    expect(stages.get('ingest')).toBe(0);
    expect(stages.get('clause_a')).toBe(1);
    expect(stages.get('clause_b')).toBe(1);
    expect(stages.get('risk_merge')).toBe(2);
  });

  it('builds dynamic desk placements and identifies sink manager without hardcoding', () => {
    const layout = buildOfficeLayout({
      overview: CONTRACT_REVIEW_OVERVIEW,
      results: null,
      telemetry: {},
      activeDimension: 'workflow_parallelization',
    });
    expect(layout.desks).toHaveLength(4);
    const manager = layout.desks.find((d) => d.isManager);
    expect(manager?.agent.agent_id).toBe('risk_merge');
    expect(layout.parallelClusters.length).toBeGreaterThanOrEqual(1);
    expect(layout.parallelClusters[0].agentIds).toEqual(expect.arrayContaining(['clause_a', 'clause_b']));
  });

  it('keeps baseline office static with zero parallel clusters or pruned edges', () => {
    const layout = buildOfficeLayout({
      overview: CONTRACT_REVIEW_OVERVIEW,
      results: null,
      telemetry: {},
      activeDimension: 'baseline',
      side: 'baseline',
    });
    expect(layout.parallelClusters).toHaveLength(0);
    expect(layout.desks.every((d) => d.modelAssigned === 'openai/gpt-oss-120b')).toBe(true);
    expect(layout.desks.every((d) => d.removedFields.length === 0)).toBe(true);
  });
});
