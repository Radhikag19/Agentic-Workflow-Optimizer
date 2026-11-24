import React, { useEffect, useMemo, useRef, useState } from 'react';
import { buildOfficeLayout, type OfficeLayoutModel } from './officeLayout';
import { OfficeSceneController } from './officeScene';
import type {
  AgentRuntimeTelemetry,
  ExperimentResultsResponse,
  WorkflowOverviewResponse,
} from '../types/workplace';
import type { DimensionViewMode } from '../store/workplaceStore';

interface OfficeCanvasPaneProps {
  title: string;
  subtitle: string;
  layout: OfficeLayoutModel;
  selectedAgentId: string | null;
  isBaseline: boolean;
  onSelectAgent: (id: string) => void;
}

const OfficeCanvasPane: React.FC<OfficeCanvasPaneProps> = ({
  title,
  subtitle,
  layout,
  selectedAgentId,
  isBaseline,
  onSelectAgent,
}) => {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<OfficeSceneController | null>(null);
  const [webglError, setWebglError] = useState<string | null>(null);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ctrl = new OfficeSceneController();
    controllerRef.current = ctrl;
    ctrl.mount(el, layout.width, layout.height).catch((err) => {
      setWebglError(err?.message || 'Pixi canvas initialization unavailable');
    });
    return () => {
      ctrl.destroy();
      controllerRef.current = null;
    };
  }, [layout.width, layout.height]);

  useEffect(() => {
    controllerRef.current?.renderLayout(layout, selectedAgentId, isBaseline, onSelectAgent);
  }, [layout, selectedAgentId, isBaseline, onSelectAgent]);

  return (
    <div className="pixel-panel" style={{ overflow: 'hidden' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 12px',
          borderBottom: '2px solid var(--border-pixel)',
          background: isBaseline ? 'var(--bg-inset)' : '#dce8dc',
        }}
      >
        <div>
          <div className="pixel-title" style={{ fontSize: '10px' }}>
            {title}
          </div>
          <div className="mono" style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: 2 }}>
            {subtitle}
          </div>
        </div>
        <span className={`pixel-badge ${isBaseline ? 'pixel-badge-warn' : 'pixel-badge-pass'}`}>
          {isBaseline ? 'FIXED HISTORICAL (0 GROQ CALLS)' : 'OPTIMIZED GROQ RUNTIME'}
        </span>
      </div>

      {webglError ? (
        <div style={{ padding: 16 }}>
          <div className="pixel-badge pixel-badge-warn">SVG/HTML Fallback Office View</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
              gap: 10,
              marginTop: 12,
            }}
          >
            {layout.desks.map((d) => (
              <button
                key={d.agent.agent_id}
                type="button"
                onClick={() => onSelectAgent(d.agent.agent_id)}
                className="pixel-panel-inset"
                style={{
                  padding: 10,
                  textAlign: 'left',
                  cursor: 'pointer',
                  borderColor:
                    d.agent.agent_id === selectedAgentId ? 'var(--accent-amber)' : 'var(--border-pixel)',
                }}
              >
                <div style={{ fontWeight: 700, fontSize: 13 }}>{d.agent.agent_name}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
                  {d.modelAssigned}
                </div>
                <div className="mono" style={{ fontSize: 11, marginTop: 4 }}>
                  State: {d.telemetry.state}
                </div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div ref={hostRef} className="office-canvas-container" style={{ width: '100%' }} />
      )}
    </div>
  );
};

interface OfficeCanvasProps {
  overview: WorkflowOverviewResponse | null;
  results: ExperimentResultsResponse | null;
  telemetry: Record<string, AgentRuntimeTelemetry>;
  activeDimension: DimensionViewMode;
  comparisonMode: boolean;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
}

export const OfficeCanvas: React.FC<OfficeCanvasProps> = ({
  overview,
  results,
  telemetry,
  activeDimension,
  comparisonMode,
  selectedAgentId,
  onSelectAgent,
}) => {
  const baselineLayout = useMemo(
    () =>
      buildOfficeLayout({
        overview,
        results,
        telemetry,
        activeDimension: 'baseline',
        side: 'baseline',
      }),
    [overview, results, telemetry]
  );

  const optimizedLayout = useMemo(
    () =>
      buildOfficeLayout({
        overview,
        results,
        telemetry,
        activeDimension,
        side: activeDimension === 'baseline' ? 'baseline' : 'optimized',
      }),
    [overview, results, telemetry, activeDimension]
  );

  if (comparisonMode) {
    return (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(430px, 1fr))',
          gap: 14,
        }}
      >
        <OfficeCanvasPane
          title="LEFT: ORIGINAL TARGET MAS"
          subtitle={`Fixed historical snapshot (${overview?.historical_baseline?.matched_input_count ?? 0} matched traces) • Model: ${overview?.baseline_model ?? '-'}`}
          layout={baselineLayout}
          selectedAgentId={selectedAgentId}
          isBaseline={true}
          onSelectAgent={onSelectAgent}
        />
        <OfficeCanvasPane
          title="RIGHT: OPTIMIZED TARGET MAS"
          subtitle={`Dimension lens: ${activeDimension.replace(/_/g, ' ')} • Live / verified artifacts`}
          layout={optimizedLayout}
          selectedAgentId={selectedAgentId}
          isBaseline={false}
          onSelectAgent={onSelectAgent}
        />
      </div>
    );
  }

  const singleIsBaseline = activeDimension === 'baseline';
  return (
    <OfficeCanvasPane
      title={singleIsBaseline ? 'HISTORICAL BASELINE OFFICE' : 'OPTIMIZED TARGET MAS FLOOR'}
      subtitle={
        singleIsBaseline
          ? `Read-only historical traces (${overview?.historical_baseline?.matched_input_count ?? 0} matched) — baseline is never re-executed`
          : `Active visualization: ${activeDimension.replace(/_/g, ' ')} — click any desk or agent to inspect`
      }
      layout={singleIsBaseline ? baselineLayout : optimizedLayout}
      selectedAgentId={selectedAgentId}
      isBaseline={singleIsBaseline}
      onSelectAgent={onSelectAgent}
    />
  );
};
