import React from 'react';
import { Award, CheckCircle2, XCircle, BarChart3 } from 'lucide-react';
import { useWorkplaceStore } from '../store/workplaceStore';

interface MetricBarChartProps {
  title: string;
  unit: string;
  rows: Record<string, any>[];
  valueKey: 'mean_latency_seconds' | 'mean_tokens' | 'mean_cost_usd';
  barColor: string;
  formatter: (v: number) => string;
}

const MetricBarChart: React.FC<MetricBarChartProps> = ({
  title,
  unit,
  rows,
  valueKey,
  barColor,
  formatter,
}) => {
  const numericValues = rows.map((r) => Number(r[valueKey] ?? 0));
  const maxVal = Math.max(0.0001, ...numericValues);

  return (
    <div className="pixel-panel" style={{ padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>{title}</strong>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
          {unit} (lower is better)
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {rows.map((r, idx) => {
          const val = Number(r[valueKey] ?? 0);
          const pct = Math.max(4, Math.round((val / maxVal) * 100));
          const isBase = r.scope_kind === 'historical_baseline';
          return (
            <div key={idx} className="mono" style={{ fontSize: 11 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontWeight: isBase ? 700 : 500 }}>{r.name}</span>
                <strong>{formatter(val)}</strong>
              </div>
              <div
                style={{
                  width: '100%',
                  height: 12,
                  background: 'var(--bg-inset)',
                  border: '1px solid var(--border-pixel)',
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: '100%',
                    background: isBase ? '#9a8466' : barColor,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const PerformanceReviewPage: React.FC = () => {
  const { activeExperimentId, activeExperimentResults } = useWorkplaceStore();

  if (!activeExperimentResults) {
    return (
      <div className="pixel-panel" style={{ padding: 20 }}>
        <div className="pixel-title" style={{ fontSize: 11, marginBottom: 8 }}>
          Performance Review & Pareto Frontier
        </div>
        <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
          Select a completed experiment or study to compare Latency, Token Consumption, Cost, and Pareto-efficient workflow configurations against the fixed historical baseline.
        </div>
      </div>
    );
  }

  const rows = activeExperimentResults.comparative_configurations || [];
  const pareto = activeExperimentResults.pareto_selection || {};
  const paretoSet: Record<string, any>[] = pareto.pareto_efficient_configurations || [];
  const recommended = pareto.recommended_overall_configuration;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Pareto Decision Banner — Never invents weighted score */}
      <div className="pixel-panel" style={{ padding: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Award size={18} />
            <span className="pixel-title" style={{ fontSize: 11 }}>
              Pareto-Optimal Configuration Review ({activeExperimentId})
            </span>
          </div>
          <span className="pixel-badge pixel-badge-pass">
            Weighted Scoring Used: {String(Boolean(pareto.weighted_scoring_used))}
          </span>
        </div>

        <div
          className="pixel-panel-inset mono"
          style={{ padding: 10, marginTop: 10, fontSize: 12 }}
        >
          <div>
            <strong>Selection Note:</strong>{' '}
            {pareto.selection_note || 'Pareto dominance computed in backend across latency, tokens, and cost.'}
          </div>
          {recommended && (
            <div style={{ marginTop: 6, color: 'var(--accent-forest)' }}>
              <strong>Recommended Single Overall Winner:</strong> {recommended.name}
            </div>
          )}
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 10,
            marginTop: 10,
          }}
        >
          {[
            { label: 'Best Latency', item: pareto.best_latency_configuration },
            { label: 'Best Token Efficiency', item: pareto.best_token_efficiency_configuration },
            { label: 'Best Cost Efficiency', item: pareto.best_cost_efficiency_configuration },
            { label: 'Combined Summary', item: pareto.combined_configuration_summary },
          ].map((card, i) => (
            <div key={i} className="pixel-panel-inset mono" style={{ padding: 10, fontSize: 11 }}>
              <div style={{ color: 'var(--ink-muted)', fontWeight: 700 }}>{card.label}</div>
              <div style={{ fontWeight: 700, fontSize: 13, marginTop: 3 }}>
                {card.item?.name || 'None eligible (quality gate)'}
              </div>
              {card.item && (
                <div style={{ marginTop: 3 }}>
                  {Number(card.item.mean_latency_seconds ?? 0).toFixed(2)}s •{' '}
                  {Math.round(Number(card.item.mean_tokens ?? 0))} tok • $
                  {Number(card.item.mean_cost_usd ?? 0).toFixed(5)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Side-by-side Performance Comparison Charts */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 12,
        }}
      >
        <MetricBarChart
          title="Mean Execution Latency"
          unit="seconds"
          rows={rows}
          valueKey="mean_latency_seconds"
          barColor="#246a49"
          formatter={(v) => `${v.toFixed(2)}s`}
        />
        <MetricBarChart
          title="Mean Total Tokens"
          unit="tokens"
          rows={rows}
          valueKey="mean_tokens"
          barColor="#3b5e8c"
          formatter={(v) => `${Math.round(v).toLocaleString()} tok`}
        />
        <MetricBarChart
          title="Mean Inference Cost"
          unit="USD"
          rows={rows}
          valueKey="mean_cost_usd"
          barColor="#d9822b"
          formatter={(v) => `$${v.toFixed(5)}`}
        />
      </div>

      {/* Full Metrics Comparison Matrix */}
      <div className="pixel-panel" style={{ padding: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
          <BarChart3 size={16} />
          <span className="pixel-title" style={{ fontSize: 11 }}>
            Baseline vs Optimized Comparative Matrix
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }} className="mono">
            <thead>
              <tr style={{ background: 'var(--bg-inset)', textAlign: 'left' }}>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Configuration</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Latency (s)</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Prompt Tok</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Completion Tok</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Total Tokens</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Cost (USD)</th>
                <th style={{ padding: 8, border: '2px solid var(--border-pixel)' }}>Pareto / Quality</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const isPareto = paretoSet.some((p) => p.name === r.name);
                const isBase = r.scope_kind === 'historical_baseline';
                return (
                  <tr key={idx} style={{ background: isBase ? '#f2ead4' : undefined }}>
                    <td style={{ padding: 8, border: '1px solid #bfb59b', fontWeight: 700 }}>
                      {r.name}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {typeof r.mean_latency_seconds === 'number'
                        ? r.mean_latency_seconds.toFixed(3)
                        : 'Unavailable'}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {r.mean_prompt_tokens ?? 'Unavailable'}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {r.mean_completion_tokens ?? 'Unavailable'}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {r.mean_tokens ?? 'Unavailable'}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {typeof r.mean_cost_usd === 'number'
                        ? `$${r.mean_cost_usd.toFixed(6)}`
                        : 'Unavailable'}
                    </td>
                    <td style={{ padding: 8, border: '1px solid #bfb59b' }}>
                      {isBase ? (
                        <span className="pixel-badge pixel-badge-unavail">HISTORICAL BASELINE</span>
                      ) : isPareto ? (
                        <span className="pixel-badge pixel-badge-pass">
                          <CheckCircle2 size={12} /> PARETO EFFICIENT
                        </span>
                      ) : r.quality_accepted ? (
                        <span className="pixel-badge pixel-badge-pass">QUALITY ACCEPTED</span>
                      ) : (
                        <span className="pixel-badge pixel-badge-fail">
                          <XCircle size={12} /> {String(r.quality_status || 'REJECTED')}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
