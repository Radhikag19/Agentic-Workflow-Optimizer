import React, { useState } from 'react';
import { Play, AlertTriangle, CheckCircle2, Layers } from 'lucide-react';
import { useWorkplaceStore } from '../store/workplaceStore';
import type { OptimizationDimension } from '../types/workplace';

const ALL_DIMENSIONS: { id: OptimizationDimension; label: string }[] = [
  { id: 'communication_data_handoff', label: '1. Communication / Data Handoff' },
  { id: 'workflow_parallelization', label: '2. Workflow Parallelization' },
  { id: 'model_selection', label: '3. Model Selection' },
  { id: 'agent_consolidation', label: '4. Agent Consolidation' },
  { id: 'prompt_optimization', label: '5. Prompt Optimization' },
];

export const ExperimentsPage: React.FC = () => {
  const {
    overview,
    baseline,
    experiments,
    activeExperimentId,
    activeExperimentStatus,
    startingExperiment,
    selectExperiment,
    startExperiment,
  } = useWorkplaceStore();

  const [experimentType, setExperimentType] = useState<
    'two_phase_study' | 'dimension_ablation_suite' | 'combined_single_workflow'
  >('two_phase_study');
  const [selectedDims, setSelectedDims] = useState<OptimizationDimension[]>([
    'communication_data_handoff',
    'workflow_parallelization',
    'model_selection',
    'agent_consolidation',
    'prompt_optimization',
  ]);
  const [numRuns, setNumRuns] = useState<number>(2);
  const [maxFaithfulnessDrop, setMaxFaithfulnessDrop] = useState<number>(0.1);
  const [maxPrecisionDrop, setMaxPrecisionDrop] = useState<number>(0.1);
  const [maxRecallDrop, setMaxRecallDrop] = useState<number>(0.1);

  const hasApprovedScope = Boolean(overview?.approved_scope && overview.approved_candidates.length > 0);

  const toggleDimension = (dim: OptimizationDimension) => {
    setSelectedDims((prev) =>
      prev.includes(dim) ? prev.filter((d) => d !== dim) : [...prev, dim]
    );
  };

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!overview) return;
    await startExperiment({
      workflow_id: overview.workflow_id,
      experiment_type: experimentType,
      selected_dimensions: selectedDims,
      num_runs: numRuns,
      quality_criteria: {
        max_faithfulness_drop: maxFaithfulnessDrop,
        max_context_precision_drop: maxPrecisionDrop,
        max_context_recall_drop: maxRecallDrop,
      },
      enable_trace_replay_fallback: false,
    });
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 14 }}>
      {/* Left: Start Experiment Form */}
      <form onSubmit={handleStart} className="pixel-panel" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div className="pixel-title" style={{ fontSize: 11 }}>
          Optimization Experiment Runner
        </div>

        <div className="pixel-panel-inset mono" style={{ padding: 10, fontSize: 11 }}>
          <div>
            Workflow: <strong>{overview?.workflow_name}</strong> ({overview?.agents.length ?? 0} agents)
          </div>
          <div style={{ marginTop: 4 }}>
            Historical Baseline: <strong>{baseline?.matched_input_count ?? 0} matched records</strong> (Read-only • 0 Groq API calls)
          </div>
        </div>

        {!hasApprovedScope && (
          <div className="pixel-badge pixel-badge-warn" style={{ padding: 8, whiteSpace: 'normal' }}>
            <AlertTriangle size={15} />
            No approved_optimization_scope JSON exists for {overview?.workflow_id} yet. Discovery candidates are shown on the right, and running live experiments is disabled until candidates are approved in Streamlit/S11.
          </div>
        )}

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
          <strong>Experiment Execution Mode</strong>
          <select
            className="pixel-select"
            value={experimentType}
            onChange={(e) => setExperimentType(e.target.value as any)}
          >
            <option value="two_phase_study">Two-Phase Study (Fixed Foundation + Variable + Pareto)</option>
            <option value="dimension_ablation_suite">Dimension Ablation Suite (Per-Dimension + Combined)</option>
            <option value="combined_single_workflow">Combined Single Optimized Workflow</option>
          </select>
        </label>

        <div>
          <strong style={{ fontSize: 13 }}>Optimization Dimensions Included</strong>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 6 }}>
            {ALL_DIMENSIONS.map((d) => (
              <label key={d.id} className="mono" style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={selectedDims.includes(d.id)}
                  onChange={() => toggleDimension(d.id)}
                />
                {d.label}
              </label>
            ))}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
          <label style={{ fontSize: 12 }}>
            <strong>Evaluated Runs (num_runs)</strong>
            <input
              type="number"
              min={1}
              max={10}
              className="pixel-input"
              style={{ width: '100%', marginTop: 4 }}
              value={numRuns}
              onChange={(e) => setNumRuns(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            <strong>Max Faithfulness Drop</strong>
            <input
              type="number"
              step="0.01"
              min={0}
              max={0.5}
              className="pixel-input"
              style={{ width: '100%', marginTop: 4 }}
              value={maxFaithfulnessDrop}
              onChange={(e) => setMaxFaithfulnessDrop(Number(e.target.value))}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            <strong>Max Context Precision Drop</strong>
            <input
              type="number"
              step="0.01"
              min={0}
              max={0.5}
              className="pixel-input"
              style={{ width: '100%', marginTop: 4 }}
              value={maxPrecisionDrop}
              onChange={(e) => setMaxPrecisionDrop(Number(e.target.value))}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            <strong>Max Context Recall Drop</strong>
            <input
              type="number"
              step="0.01"
              min={0}
              max={0.5}
              className="pixel-input"
              style={{ width: '100%', marginTop: 4 }}
              value={maxRecallDrop}
              onChange={(e) => setMaxRecallDrop(Number(e.target.value))}
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={!hasApprovedScope || startingExperiment || selectedDims.length === 0}
          className="pixel-btn pixel-btn-primary"
          style={{ justifyContent: 'center', marginTop: 4 }}
        >
          <Play size={15} />
          {startingExperiment ? 'Launching Optimized Subprocess...' : 'Start Optimized Experiment (Live Groq)'}
        </button>

        {activeExperimentStatus && (
          <div className="pixel-panel-inset mono" style={{ padding: 10, fontSize: 11 }}>
            <div>
              Selected Run: <strong>{activeExperimentStatus.experiment_id}</strong> ({activeExperimentStatus.status})
            </div>
            {activeExperimentStatus.current_phase && (
              <div style={{ marginTop: 3 }}>Phase: {activeExperimentStatus.current_phase}</div>
            )}
            {activeExperimentStatus.error && (
              <div className="pixel-badge pixel-badge-fail" style={{ marginTop: 6 }}>
                Failure Diagnostics: {activeExperimentStatus.error}
              </div>
            )}
          </div>
        )}
      </form>

      {/* Right: Saved Experiments + Approved Candidates */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="pixel-panel" style={{ padding: 16 }}>
          <div className="pixel-title" style={{ fontSize: 11, marginBottom: 10 }}>
            Experiment History & Live Studies
          </div>
          {experiments.length === 0 ? (
            <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
              No saved experiments found for {overview?.workflow_id} yet.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {experiments.map((exp) => {
                const selected = exp.experiment_id === activeExperimentId;
                return (
                  <button
                    key={exp.experiment_id}
                    type="button"
                    onClick={() => selectExperiment(exp.experiment_id)}
                    className={`pixel-btn ${selected ? 'pixel-btn-amber' : ''}`}
                    style={{
                      width: '100%',
                      justifyContent: 'space-between',
                      textAlign: 'left',
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 700 }}>{exp.experiment_id}</div>
                      <div className="mono" style={{ fontSize: 11 }}>
                        {exp.experiment_type} • {exp.status}
                      </div>
                    </div>
                    {exp.has_results && <CheckCircle2 size={15} />}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="pixel-panel" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
            <Layers size={15} />
            <span className="pixel-title" style={{ fontSize: 11 }}>
              {hasApprovedScope
                ? `Approved Scope (${overview?.approved_candidates.length} candidates)`
                : 'Discovered Candidates (Read-only Preview)'}
            </span>
          </div>
          <div style={{ maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(hasApprovedScope
              ? overview?.approved_candidates || []
              : Object.values(overview?.discovered_candidates_by_dimension || {}).flat()
            ).map((c, idx) => (
              <div key={c.candidate_id || idx} className="pixel-panel-inset mono" style={{ padding: 8, fontSize: 11 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>{c.candidate_id}</strong>
                  <span className="pixel-badge pixel-badge-pass">
                    {c.optimization_type || c.dimension}
                  </span>
                </div>
                <div style={{ marginTop: 3 }}>
                  Agents: {(c.target_agents || []).join(', ') || 'workflow-wide'}
                </div>
                {c.title && <div style={{ color: 'var(--ink-muted)', marginTop: 2 }}>{c.title}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
