import React, { useMemo, useState } from 'react';
import { ShieldCheck, AlertCircle, HelpCircle, Cpu } from 'lucide-react';
import { useWorkplaceStore } from '../store/workplaceStore';

function formatRagasValue(val: unknown): { text: string; tone: 'pass' | 'warn' | 'fail' | 'unavail' } {
  if (val === null || val === undefined || val === 'unavailable') {
    return { text: 'Unavailable', tone: 'unavail' };
  }
  if (val === 'reference_only') {
    return { text: 'Historical Ref', tone: 'unavail' };
  }
  if (typeof val === 'number' && Number.isFinite(val)) {
    const tone = val >= 0.85 ? 'pass' : val >= 0.7 ? 'warn' : 'fail';
    return { text: val.toFixed(4), tone };
  }
  return { text: String(val), tone: 'unavail' };
}

interface CollectedRagasEntry {
  key: string;
  label: string;
  ragas: Record<string, any>;
  evalRecord?: Record<string, any>;
}

export const QualityLabPage: React.FC = () => {
  const { activeExperimentId, activeExperimentResults } = useWorkplaceStore();
  const [selectedKey, setSelectedKey] = useState<string>('');

  const ragasEntries = useMemo<CollectedRagasEntry[]>(() => {
    if (!activeExperimentResults) return [];
    const list: CollectedRagasEntry[] = [];

    const paRagas = activeExperimentResults.phase_a_foundation?.ragas_evaluations || {};
    const paEvals = activeExperimentResults.phase_a_foundation?.dimension_evaluations || {};
    for (const [dim, rag] of Object.entries(paRagas)) {
      if (rag && typeof rag === 'object') {
        list.push({
          key: `phase_a:${dim}`,
          label: `Phase A — ${dim.replace(/_/g, ' ')}`,
          ragas: rag as Record<string, any>,
          evalRecord: paEvals[dim],
        });
      }
    }

    const pbRagas = activeExperimentResults.phase_b_variable_dimensions?.ragas_evaluations || {};
    const pbEvals = activeExperimentResults.phase_b_variable_dimensions?.dimension_evaluations || {};
    for (const [dim, rag] of Object.entries(pbRagas)) {
      if (rag && typeof rag === 'object') {
        list.push({
          key: `phase_b:${dim}`,
          label: `Phase B — ${dim.replace(/_/g, ' ')}`,
          ragas: rag as Record<string, any>,
          evalRecord: pbEvals[dim],
        });
      }
    }

    const perDim = activeExperimentResults.per_dimension_results || {};
    for (const [dim, item] of Object.entries(perDim)) {
      const rag = (item as any)?.ragas_evaluation;
      if (rag && typeof rag === 'object') {
        list.push({
          key: `per_dim:${dim}`,
          label: `Ablation — ${dim.replace(/_/g, ' ')}`,
          ragas: rag,
          evalRecord: item as Record<string, any>,
        });
      }
    }

    const combRag =
      activeExperimentResults.combined_result?.ragas_evaluation ||
      activeExperimentResults.phase_c_combined?.ragas_evaluation;
    if (combRag && typeof combRag === 'object') {
      list.push({
        key: 'combined',
        label: 'Combined Workflow',
        ragas: combRag,
        evalRecord: activeExperimentResults.phase_c_combined || activeExperimentResults.combined_result,
      });
    }

    return list;
  }, [activeExperimentResults]);

  const activeEntry =
    ragasEntries.find((e) => e.key === selectedKey) || ragasEntries[0] || null;

  const compRows = (activeExperimentResults?.comparative_configurations || []).filter(
    (r) => r.scope_kind !== 'historical_baseline'
  );

  const judgeUsage = activeExperimentResults?.ragas_evaluation_model_usage || {};

  if (!activeExperimentResults) {
    return (
      <div className="pixel-panel" style={{ padding: 20 }}>
        <div className="pixel-title" style={{ fontSize: 11, marginBottom: 8 }}>
          Quality Lab (RAGAS Evaluation Engine)
        </div>
        <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
          Select or run an experiment in the Experiments tab to inspect Faithfulness, Context Precision, Context Recall, and sample-level RAGAS verdicts.
        </div>
      </div>
    );
  }

  const samples: Record<string, any>[] =
    activeEntry?.ragas?.samples ||
    activeEntry?.ragas?.per_sample_scores ||
    [];

  const faith = formatRagasValue(activeEntry?.ragas?.faithfulness);
  const precision = formatRagasValue(activeEntry?.ragas?.context_precision);
  const recall = formatRagasValue(activeEntry?.ragas?.context_recall);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Metric Definitions & RAGAS Judge Resource Banner */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 12,
        }}
      >
        <div className="pixel-panel" style={{ padding: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, fontSize: 13 }}>
            <HelpCircle size={15} /> Faithfulness
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            Measures whether every claim in the optimized agent response is grounded strictly in its upstream handoff context (hallucination check).
          </div>
        </div>
        <div className="pixel-panel" style={{ padding: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, fontSize: 13 }}>
            <HelpCircle size={15} /> Context Precision
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            Measures whether relevant context fields are preserved upfront without noisy or bloated filler in the agent handoff payload.
          </div>
        </div>
        <div className="pixel-panel" style={{ padding: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, fontSize: 13 }}>
            <HelpCircle size={15} /> Context Recall
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            Measures whether all essential historical reference information survived optimization without losing required facts.
          </div>
        </div>
        <div className="pixel-panel" style={{ padding: 12, background: '#ece6f4' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, fontSize: 13 }}>
            <Cpu size={15} /> Separate RAGAS Judge Usage
          </div>
          <div className="mono" style={{ fontSize: 11, marginTop: 4 }}>
            Calls: <strong>{judgeUsage.api_calls ?? 'Unavailable'}</strong> • Tokens:{' '}
            <strong>{judgeUsage.total_tokens ?? 'Unavailable'}</strong> • Cost:{' '}
            <strong>
              {typeof judgeUsage.estimated_cost_usd === 'number'
                ? `$${judgeUsage.estimated_cost_usd.toFixed(4)}`
                : 'Unavailable'}
            </strong>
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--ink-muted)', marginTop: 2 }}>
            Tracked separately — never mixed into MAS optimization metrics.
          </div>
        </div>
      </div>

      {/* Comparative Quality Gate Table across All Configurations */}
      <div className="pixel-panel" style={{ padding: 14 }}>
        <div className="pixel-title" style={{ fontSize: 11, marginBottom: 10 }}>
          Quality Gate Verdicts — {activeExperimentId}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }} className="mono">
            <thead>
              <tr style={{ background: 'var(--bg-inset)', textAlign: 'left' }}>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Configuration</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Faithfulness</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Context Precision</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Context Recall</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Quality Gate</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Gate Reasons</th>
              </tr>
            </thead>
            <tbody>
              {compRows.map((row, idx) => {
                const f = formatRagasValue(row.faithfulness);
                const p = formatRagasValue(row.context_precision);
                const r = formatRagasValue(row.context_recall);
                const accepted = Boolean(row.quality_accepted);
                return (
                  <tr key={idx}>
                    <td style={{ padding: 8, border: '1px solid #bfb59b', fontWeight: 700 }}>
                      {row.name}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      <span className={`pixel-badge pixel-badge-${f.tone}`}>{f.text}</span>
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      <span className={`pixel-badge pixel-badge-${p.tone}`}>{p.text}</span>
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      <span className={`pixel-badge pixel-badge-${r.tone}`}>{r.text}</span>
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      <span className={`pixel-badge ${accepted ? 'pixel-badge-pass' : 'pixel-badge-fail'}`}>
                        {String(row.quality_status || (accepted ? 'ACCEPTED' : 'REJECTED'))}
                      </span>
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b', fontSize: 11 }}>
                      {Array.isArray(row.quality_reasons) && row.quality_reasons.length > 0
                        ? row.quality_reasons.join('; ')
                        : accepted
                        ? 'Meets all configured thresholds'
                        : 'Unavailable / check sub-report'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Dimension RAGAS Drilldown + Sample-Level Evidence */}
      {activeEntry && (
        <div className="pixel-panel" style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldCheck size={16} />
              <span className="pixel-title" style={{ fontSize: 11 }}>
                Sample-Level RAGAS Context Inspector
              </span>
            </div>

            <select
              className="pixel-select"
              value={activeEntry.key}
              onChange={(e) => setSelectedKey(e.target.value)}
            >
              {ragasEntries.map((ent) => (
                <option key={ent.key} value={ent.key}>
                  {ent.label}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            <span className={`pixel-badge pixel-badge-${faith.tone}`}>Faithfulness: {faith.text}</span>
            <span className={`pixel-badge pixel-badge-${precision.tone}`}>
              Context Precision: {precision.text}
            </span>
            <span className={`pixel-badge pixel-badge-${recall.tone}`}>Context Recall: {recall.text}</span>
            <span className="pixel-badge pixel-badge-unavail">
              RAGAS Status: {String(activeEntry.ragas?.status || 'unknown')}
            </span>
          </div>

          {Array.isArray(activeEntry.ragas?.missing_context_samples) &&
            activeEntry.ragas.missing_context_samples.length > 0 && (
              <div className="pixel-badge pixel-badge-warn" style={{ padding: 8 }}>
                <AlertCircle size={14} />
                Missing context detected on {activeEntry.ragas.missing_context_samples.length} sample(s); never interpolated to 0.
              </div>
            )}

          {samples.length === 0 ? (
            <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
              No sample-level payload records included in this section.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 360, overflowY: 'auto' }}>
              {samples.slice(0, 12).map((samp, idx) => (
                <div key={idx} className="pixel-panel-inset mono" style={{ padding: 10, fontSize: 11 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
                    <strong>
                      Sample #{idx + 1} {samp.agent_id ? `• ${samp.agent_id}` : ''}
                    </strong>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <span>F: {formatRagasValue(samp.faithfulness).text}</span>
                      <span>P: {formatRagasValue(samp.context_precision).text}</span>
                      <span>R: {formatRagasValue(samp.context_recall).text}</span>
                    </div>
                  </div>
                  {samp.user_input && (
                    <div style={{ marginTop: 4 }}>
                      <strong>Input:</strong> {String(samp.user_input).slice(0, 180)}
                    </div>
                  )}
                  {samp.reference && (
                    <div style={{ marginTop: 3, color: 'var(--ink-espresso)' }}>
                      <strong>Reference:</strong> {String(samp.reference).slice(0, 180)}
                    </div>
                  )}
                  {samp.response && (
                    <div style={{ marginTop: 3, color: 'var(--accent-forest)' }}>
                      <strong>Response:</strong> {String(samp.response).slice(0, 180)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
