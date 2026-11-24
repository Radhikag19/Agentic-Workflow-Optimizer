export type NavTab = 'office' | 'experiments' | 'quality_lab' | 'performance_review' | 'reports';

export type OptimizationDimension =
  | 'communication_data_handoff'
  | 'workflow_parallelization'
  | 'model_selection'
  | 'agent_consolidation'
  | 'prompt_optimization';

export type AgentVisualState =
  | 'idle'
  | 'thinking'
  | 'working'
  | 'waiting'
  | 'moving'
  | 'parallel_working'
  | 'completed'
  | 'failed'
  | 'blocked';

export interface WorkflowSummary {
  workflow_id: string;
  workflow_name: string;
  agent_count: number;
  baseline_model: string;
  llm_provider: string;
  has_historical_traces: boolean;
  historical_trace_count: number;
  has_approved_scope: boolean;
  approved_candidate_count: number;
  has_discovery_report: boolean;
}

export interface AgentProfile {
  agent_id: string;
  agent_name: string;
  role: string;
  goal?: string;
  responsibilities: string[];
  dependencies: string[];
  input_fields: string[];
  output_fields: string[];
  model: string;
  original_prompt_preview?: string;
  execution_style?: 'pure_llm' | 'hybrid_python_plus_llm';
  helper_functions?: string[];
  historical_mean_latency_seconds?: number | null;
  historical_mean_tokens?: number | null;
  historical_mean_cost_usd?: number | null;
}

export interface WorkflowOverviewResponse {
  workflow_id: string;
  workflow_name: string;
  baseline_model: string;
  llm_provider: string;
  agents: AgentProfile[];
  execution_order: string[];
  historical_baseline: {
    fixed_baseline_reference_id?: string | null;
    source: string;
    re_executed: boolean;
    groq_api_calls: number;
    matched_input_count: number;
    total_historical_traces_found?: number;
    metrics: Record<string, any>;
    trace_directory?: string | null;
    excluded_records?: any[];
  };
  approved_scope?: Record<string, any> | null;
  approved_candidates: Record<string, any>[];
  discovered_candidates_by_dimension: Record<string, Record<string, any>[]>;
}

export interface BaselineResponse {
  workflow_id: string;
  fixed_baseline_reference_id?: string | null;
  source: string;
  execution_mode: string;
  re_executed: boolean;
  groq_api_calls: number;
  trace_directory?: string | null;
  total_historical_traces_found: number;
  matched_input_count: number;
  unmatched_input_count: number;
  excluded_records: any[];
  execution_order: string[];
  historical_model: string;
  metrics: Record<string, any>;
  traces: Record<string, any>[];
  replay_traces?: Record<string, any>[];
}

export interface ExperimentStartRequest {
  workflow_id: string;
  experiment_type: 'two_phase_study' | 'dimension_ablation_suite' | 'combined_single_workflow';
  selected_dimensions?: OptimizationDimension[];
  num_runs: number;
  approved_scope_reference?: string | null;
  quality_criteria?: {
    max_faithfulness_drop?: number;
    max_context_precision_drop?: number;
    max_context_recall_drop?: number;
    min_absolute_faithfulness?: number;
    min_absolute_context_precision?: number;
    min_absolute_context_recall?: number;
    min_quality_score?: number;
    allow_unavailable_ragas_to_pass?: boolean;
  };
  enable_trace_replay_fallback?: boolean;
}

export interface ExperimentStatusResponse {
  experiment_id: string;
  workflow_id: string;
  experiment_type: string;
  status: string;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  current_phase?: string | null;
  progress_pct?: number | null;
  error?: string | null;
  report_paths: Record<string, string>;
  has_results: boolean;
}

export interface ExperimentResultsResponse {
  experiment_id: string;
  workflow_id: string;
  experiment_type: string;
  status: string;
  fixed_optimization_foundation?: string[] | null;
  baseline_runtime_results: Record<string, any>;
  phase_a_foundation: Record<string, any>;
  phase_b_variable_dimensions: Record<string, any>;
  phase_c_combined: Record<string, any>;
  comparative_configurations: Record<string, any>[];
  pareto_selection: Record<string, any>;
  ragas_evaluation_model_usage: Record<string, any>;
  per_dimension_results: Record<string, any>;
  combined_result: Record<string, any>;
  single_experiment_result: Record<string, any>;
  limitations: string[];
  original_integrity_after?: Record<string, any> | null;
}

export interface OfficeEvent {
  event_id: string;
  experiment_id: string;
  timestamp: string;
  event_type: string;
  agent_id?: string | null;
  payload: Record<string, any>;
}

export interface ReportBundleResponse {
  experiment_id: string;
  workflow_id: string;
  json_report_path?: string | null;
  markdown_report_path?: string | null;
  json_report: Record<string, any>;
  markdown_report: string;
  sub_reports: Record<string, any>[];
}

export interface AgentRuntimeTelemetry {
  agent_id: string;
  state: AgentVisualState;
  current_task: string;
  speech_bubble?: string;
  model_used?: string;
  latency_seconds?: number | null;
  tokens_used?: number | null;
  cost_usd?: number | null;
  error_message?: string | null;
  parallel_group?: number | null;
  removed_fields?: string[];
  retained_fields?: string[];
  consolidated_into?: string | null;
  prompt_compressed_pct?: number | null;
  output_preview?: string | null;
}
