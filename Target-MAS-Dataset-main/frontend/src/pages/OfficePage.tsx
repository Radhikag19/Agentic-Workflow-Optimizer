import React from 'react';
import { DimensionToolbar } from '../components/DimensionToolbar';
import { DimensionLensBanner } from '../components/DimensionLensBanner';
import { TraceSimulationBar } from '../components/TraceSimulationBar';
import { AgentInspector } from '../components/AgentInspector';
import { ActivityFeed } from '../components/ActivityFeed';
import { OfficeCanvas } from '../office/OfficeCanvas';
import { buildOfficeLayout } from '../office/officeLayout';
import { buildSimulationWaves } from '../store/traceSimulator';
import { useWorkplaceStore } from '../store/workplaceStore';

export const OfficePage: React.FC = () => {
  const {
    overview,
    baseline,
    activeExperimentId,
    activeExperimentStatus,
    activeExperimentResults,
    events,
    agentTelemetry,
    activeDimension,
    comparisonMode,
    selectedAgentId,
    wsConnected,
    simPlaying,
    simTraceIndex,
    simStepIndex,
    setActiveDimension,
    setComparisonMode,
    setSelectedAgentId,
    startTraceSimulation,
    pauseTraceSimulation,
    stepTraceSimulation,
    resetTraceSimulation,
    selectSimTraceIndex,
  } = useWorkplaceStore();

  const agents = overview?.agents ?? [];
  const selectedAgent = agents.find((a) => a.agent_id === selectedAgentId) ?? null;
  const selectedTelemetry = selectedAgentId ? agentTelemetry[selectedAgentId] ?? null : null;

  const derivedLayout = buildOfficeLayout({
    overview,
    results: activeExperimentResults,
    telemetry: agentTelemetry,
    activeDimension,
    side: activeDimension === 'baseline' ? 'baseline' : 'optimized',
  });

  const baselineMeanLatency =
    baseline?.metrics?.workflow_mean_latency_seconds ??
    overview?.historical_baseline?.metrics?.mean_latency_seconds;
  const baselineMeanTokens =
    baseline?.metrics?.workflow_mean_tokens ??
    overview?.historical_baseline?.metrics?.mean_tokens;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Top KPI & Integrity Strip */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 10,
        }}
      >
        <div className="pixel-panel" style={{ padding: '10px 12px' }}>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
            TARGET WORKFLOW
          </div>
          <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2 }}>
            {overview?.workflow_name || 'Loading...'}
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--accent-forest)' }}>
            {agents.length} dynamic agents • {overview?.baseline_model || '-'}
          </div>
        </div>

        <div className="pixel-panel" style={{ padding: '10px 12px' }}>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
            FIXED HISTORICAL BASELINE
          </div>
          <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2 }}>
            {baseline?.matched_input_count ?? overview?.historical_baseline?.matched_input_count ?? 0} /{' '}
            {baseline?.total_historical_traces_found ??
              overview?.historical_baseline?.total_historical_traces_found ??
              0}{' '}
            traces
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-espresso)' }}>
            {baselineMeanLatency != null ? `${Number(baselineMeanLatency).toFixed(2)}s` : 'N/A'} •{' '}
            {baselineMeanTokens != null ? `${Math.round(Number(baselineMeanTokens))} tok` : 'N/A'} • 0 Groq calls
          </div>
        </div>

        <div className="pixel-panel" style={{ padding: '10px 12px' }}>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
            ACTIVE EXPERIMENT / STUDY
          </div>
          <div style={{ fontWeight: 700, fontSize: 14, marginTop: 2, wordBreak: 'break-all' }}>
            {activeExperimentId || 'Baseline Inspection Only'}
          </div>
          <div className="mono" style={{ fontSize: 11, marginTop: 2 }}>
            Status:{' '}
            <strong>{activeExperimentStatus?.status?.toUpperCase() || 'NO RUN SELECTED'}</strong>
          </div>
        </div>

        <div className="pixel-panel" style={{ padding: '10px 12px' }}>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
            DIMENSION EVIDENCE ACTIVE
          </div>
          <div style={{ fontWeight: 700, fontSize: 14, marginTop: 2 }}>
            {activeDimension.replace(/_/g, ' ').toUpperCase()}
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-espresso)' }}>
            {derivedLayout.parallelClusters.length} parallel groups •{' '}
            {derivedLayout.handoffEdges.filter((e) => e.removedFields.length > 0).length} pruned edges
          </div>
        </div>
      </div>

      {/* Interactive Optimization Dimension Lens & Before/After Toggle */}
      <DimensionToolbar
        activeDimension={activeDimension}
        comparisonMode={comparisonMode}
        onSelectDimension={setActiveDimension}
        onToggleComparison={setComparisonMode}
      />

      {/* Dimension Evidence Context Strip & Historical Trace Simulation */}
      <DimensionLensBanner
        activeDimension={activeDimension}
        overview={overview}
        results={activeExperimentResults}
      />

      <TraceSimulationBar
        replayList={
          (baseline?.traces && baseline.traces.length > 0 ? baseline.traces : baseline?.replay_traces) || []
        }
        waveCount={overview ? buildSimulationWaves(overview, activeDimension).length : 0}
        simPlaying={simPlaying}
        simTraceIndex={simTraceIndex}
        simStepIndex={simStepIndex}
        onSelectTrace={selectSimTraceIndex}
        onPlay={startTraceSimulation}
        onPause={pauseTraceSimulation}
        onStep={stepTraceSimulation}
        onReset={resetTraceSimulation}
      />

      {/* Office Canvas + Inspector Sidebar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: comparisonMode ? '1fr' : 'minmax(0, 2.2fr) minmax(300px, 1fr)',
          gap: 14,
          alignItems: 'start',
        }}
      >
        <OfficeCanvas
          overview={overview}
          results={activeExperimentResults}
          telemetry={agentTelemetry}
          activeDimension={activeDimension}
          comparisonMode={comparisonMode}
          selectedAgentId={selectedAgentId}
          onSelectAgent={setSelectedAgentId}
        />

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: comparisonMode ? 'repeat(auto-fit, minmax(300px, 1fr))' : '1fr',
            gap: 14,
          }}
        >
          <AgentInspector agent={selectedAgent} telemetry={selectedTelemetry} />
          <ActivityFeed
            events={events}
            wsConnected={wsConnected}
            activeExperimentId={activeExperimentId}
          />
        </div>
      </div>
    </div>
  );
};
