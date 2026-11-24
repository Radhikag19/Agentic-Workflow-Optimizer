import React from 'react';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';

interface TraceSimulationBarProps {
  replayList: Record<string, any>[];
  waveCount: number;
  simPlaying: boolean;
  simTraceIndex: number;
  simStepIndex: number;
  onSelectTrace: (idx: number) => void;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
}

export const TraceSimulationBar: React.FC<TraceSimulationBarProps> = ({
  replayList,
  waveCount,
  simPlaying,
  simTraceIndex,
  simStepIndex,
  onSelectTrace,
  onPlay,
  onPause,
  onStep,
  onReset,
}) => {
  const disabled = replayList.length === 0 || waveCount === 0;

  return (
    <div
      className="pixel-panel"
      style={{
        padding: '8px 12px',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 10,
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
        <span className="pixel-title" style={{ fontSize: 9 }}>
          Trace Simulation:
        </span>
        <select
          aria-label="Historical Trace Record"
          className="pixel-select"
          style={{ fontSize: 11, padding: '4px 8px' }}
          value={simTraceIndex}
          onChange={(e) => onSelectTrace(Number(e.target.value))}
          disabled={replayList.length === 0}
        >
          {replayList.length === 0 ? (
            <option value={0}>No historical traces available</option>
          ) : (
            replayList.map((tr, i) => (
              <option key={i} value={i}>
                Trace #{i + 1}: {String(tr.run_id || `trace_${i + 1}`).slice(0, 26)} (
                {tr.duration_seconds != null ? `${Number(tr.duration_seconds).toFixed(1)}s` : '?'})
              </option>
            ))
          )}
        </select>

        <button
          type="button"
          className={`pixel-btn ${simPlaying ? 'pixel-btn-amber' : 'pixel-btn-primary'}`}
          style={{ padding: '5px 10px', fontSize: 12 }}
          disabled={disabled}
          onClick={simPlaying ? onPause : onPlay}
        >
          {simPlaying ? <Pause size={13} /> : <Play size={13} />}
          {simPlaying ? 'Pause' : 'Simulate Trace'}
        </button>

        <button
          type="button"
          className="pixel-btn"
          style={{ padding: '5px 9px', fontSize: 12 }}
          disabled={disabled}
          onClick={onStep}
        >
          <SkipForward size={13} /> Step
        </button>

        <button
          type="button"
          className="pixel-btn"
          style={{ padding: '5px 9px', fontSize: 12 }}
          disabled={replayList.length === 0}
          onClick={onReset}
        >
          <RotateCcw size={13} /> Reset
        </button>
      </div>

      <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
        Step: <strong>{Math.max(0, simStepIndex)} / {waveCount}</strong> • 0 Groq API Calls
      </div>
    </div>
  );
};
