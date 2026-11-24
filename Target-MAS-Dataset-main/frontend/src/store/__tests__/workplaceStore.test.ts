import React from 'react';
import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { AgentInspector } from '../../components/AgentInspector';
import { useWorkplaceStore } from '../workplaceStore';

describe('workplaceStore runtime event updates & audio defaults', () => {
  beforeEach(() => {
    useWorkplaceStore.setState({
      events: [],
      agentTelemetry: {
        resume_intake_agent: {
          agent_id: 'resume_intake_agent',
          state: 'idle',
          current_task: 'Standing by',
        },
        qualification_agent: {
          agent_id: 'qualification_agent',
          state: 'idle',
          current_task: 'Standing by',
        },
      },
      soundMuted: true,
    });
  });

  it('starts muted by default and toggles cleanly', () => {
    expect(useWorkplaceStore.getState().soundMuted).toBe(true);
    useWorkplaceStore.getState().toggleSoundMuted();
    expect(useWorkplaceStore.getState().soundMuted).toBe(false);
  });

  it('maps WebSocket runtime events into office character states and deduplicates', () => {
    const store = useWorkplaceStore.getState();

    store.applyEvent({
      event_id: 'ev_1',
      experiment_id: 'exp_live',
      timestamp: '2026-11-24T12:00:00Z',
      event_type: 'groq_request_started',
      agent_id: 'resume_intake_agent',
      payload: { model: 'llama-3.1-8b-instant', summary: 'Live Groq prompt' },
    });

    // Duplicate event should be ignored cleanly
    store.applyEvent({
      event_id: 'ev_1',
      experiment_id: 'exp_live',
      timestamp: '2026-11-24T12:00:00Z',
      event_type: 'groq_request_started',
      agent_id: 'resume_intake_agent',
      payload: { model: 'llama-3.1-8b-instant', summary: 'Live Groq prompt' },
    });

    expect(useWorkplaceStore.getState().events).toHaveLength(1);
    expect(useWorkplaceStore.getState().agentTelemetry.resume_intake_agent.state).toBe('working');

    store.applyEvent({
      event_id: 'ev_2',
      experiment_id: 'exp_live',
      timestamp: '2026-11-24T12:00:01Z',
      event_type: 'agent_parallel_started',
      agent_id: null,
      payload: { group_index: 1, agent_ids: ['resume_intake_agent', 'qualification_agent'] },
    });

    expect(useWorkplaceStore.getState().agentTelemetry.qualification_agent.state).toBe(
      'parallel_working'
    );

    store.applyEvent({
      event_id: 'ev_3',
      experiment_id: 'exp_live',
      timestamp: '2026-11-24T12:00:02Z',
      event_type: 'agent_failed',
      agent_id: 'qualification_agent',
      payload: { error: '429 rate limit exhausted' },
    });

    expect(useWorkplaceStore.getState().agentTelemetry.qualification_agent.state).toBe('failed');
    expect(useWorkplaceStore.getState().agentTelemetry.qualification_agent.error_message).toContain(
      '429'
    );
  });

  it('steps through historical trace simulation waves cleanly without Groq calls', () => {
    useWorkplaceStore.setState({
      overview: {
        workflow_id: 'customer_support',
        workflow_name: 'customer_support',
        baseline_model: 'openai/gpt-oss-120b',
        llm_provider: 'Groq',
        execution_order: ['ticket_classification_agent', 'policy_retrieval_agent'],
        agents: [
          {
            agent_id: 'ticket_classification_agent',
            agent_name: 'Ticket Classifier',
            role: 'Triage Specialist',
            responsibilities: ['Classify tickets'],
            dependencies: [],
            input_fields: ['raw_ticket_record'],
            output_fields: ['category'],
            model: 'openai/gpt-oss-120b',
            execution_style: 'hybrid_python_plus_llm',
            helper_functions: ['deterministic_classification_hints'],
          },
          {
            agent_id: 'policy_retrieval_agent',
            agent_name: 'Policy Retrieval',
            role: 'Policy Specialist',
            responsibilities: ['Retrieve policies'],
            dependencies: ['ticket_classification_agent'],
            input_fields: ['category'],
            output_fields: ['policies'],
            model: 'openai/gpt-oss-120b',
            execution_style: 'hybrid_python_plus_llm',
            helper_functions: ['deterministic_policy_retrieval'],
          },
        ],
        historical_baseline: {
          source: 'historical_runtime_traces',
          re_executed: false,
          groq_api_calls: 0,
          matched_input_count: 0,
          metrics: {},
        },
        approved_candidates: [],
        discovered_candidates_by_dimension: {},
      },
      baseline: {
        workflow_id: 'customer_support',
        source: 'historical_runtime_traces',
        execution_mode: 'historical_recorded_execution',
        re_executed: false,
        groq_api_calls: 0,
        total_historical_traces_found: 30,
        matched_input_count: 0,
        unmatched_input_count: 30,
        excluded_records: [],
        execution_order: ['ticket_classification_agent', 'policy_retrieval_agent'],
        historical_model: 'openai/gpt-oss-120b',
        metrics: {},
        traces: [],
        replay_traces: [
          {
            run_id: 'trace_01',
            duration_seconds: 4.5,
            agent_executions: [
              { duration_seconds: 2.0, tokens: 700, output: { category: 'billing' } },
              { duration_seconds: 2.5, tokens: 800, output: { policies: ['POL-1'] } },
            ],
          },
        ],
      },
      activeDimension: 'baseline',
      simTraceIndex: 0,
      simStepIndex: -1,
    });

    const store = useWorkplaceStore.getState();
    store.stepTraceSimulation();
    expect(useWorkplaceStore.getState().simStepIndex).toBe(0);
    expect(useWorkplaceStore.getState().agentTelemetry.ticket_classification_agent.state).toBe('working');

    store.stepTraceSimulation();
    expect(useWorkplaceStore.getState().agentTelemetry.ticket_classification_agent.state).toBe('completed');
    expect(useWorkplaceStore.getState().agentTelemetry.policy_retrieval_agent.state).toBe('working');

    store.stepTraceSimulation();
    expect(useWorkplaceStore.getState().agentTelemetry.policy_retrieval_agent.state).toBe('completed');
    expect(useWorkplaceStore.getState().agentTelemetry.policy_retrieval_agent.tokens_used).toBe(800);
  });

  it('renders AgentInspector with hybrid helper badge and never crashes on string responsibilities', () => {
    render(
      React.createElement(AgentInspector, {
        agent: {
          agent_id: 'issue_investigation_agent',
          agent_name: 'Issue Investigation Agent',
          role: 'Technical Support Investigator',
          responsibilities: 'Investigate ticket details' as unknown as string[],
          dependencies: ['policy_retrieval_agent'],
          input_fields: ['ticket_data'],
          output_fields: ['investigation_findings'],
          model: 'openai/gpt-oss-120b',
          execution_style: 'hybrid_python_plus_llm',
          helper_functions: ['deterministic_customer_lookup'],
          original_prompt_preview: 'Goal: Investigate root causes',
        },
        telemetry: {
          agent_id: 'issue_investigation_agent',
          state: 'completed',
          current_task: 'Completed',
          output_preview: '{"root_cause_hypothesis":"AUTH-403"}',
        },
      })
    );
    expect(screen.getByText('Hybrid: Python Helper + LLM')).toBeInTheDocument();
    expect(screen.getByText('fn: deterministic_customer_lookup()')).toBeInTheDocument();
    expect(screen.getByText('Investigate ticket details')).toBeInTheDocument();
  });
});
