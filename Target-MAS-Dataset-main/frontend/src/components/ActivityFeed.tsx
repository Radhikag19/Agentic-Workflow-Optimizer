import React from 'react';
import { Radio, Activity } from 'lucide-react';
import type { OfficeEvent } from '../types/workplace';

interface ActivityFeedProps {
  events: OfficeEvent[];
  wsConnected: boolean;
  activeExperimentId: string | null;
}

export const ActivityFeed: React.FC<ActivityFeedProps> = ({
  events,
  wsConnected,
  activeExperimentId,
}) => {
  const ordered = [...events].slice(-25).reverse();

  return (
    <div className="pixel-panel" style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Activity size={15} />
          <span className="pixel-title" style={{ fontSize: '10px' }}>
            Live Office Ticker
          </span>
        </div>
        <span className={`pixel-badge ${wsConnected ? 'pixel-badge-pass' : 'pixel-badge-unavail'}`}>
          <Radio size={11} />
          {wsConnected ? 'WS LIVE' : activeExperimentId ? 'HISTORY LOG' : 'IDLE'}
        </span>
      </div>

      <div
        className="pixel-panel-inset mono"
        style={{
          maxHeight: 280,
          overflowY: 'auto',
          padding: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          fontSize: 11,
        }}
      >
        {ordered.length === 0 ? (
          <div style={{ color: 'var(--ink-muted)' }}>
            No live runtime events streamed yet. Start an experiment in the Experiments tab or inspect saved baseline/study reports.
          </div>
        ) : (
          ordered.map((ev) => (
            <div
              key={ev.event_id}
              style={{
                borderBottom: '1px dashed #bfb59b',
                paddingBottom: 5,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <strong style={{ color: 'var(--accent-forest)' }}>{ev.event_type}</strong>
                <span style={{ color: 'var(--ink-muted)', fontSize: 10 }}>
                  {ev.timestamp.slice(11, 19)}
                </span>
              </div>
              <div style={{ color: 'var(--ink-espresso)', marginTop: 2 }}>
                {ev.agent_id ? `[${ev.agent_id}] ` : ''}
                {String(
                  ev.payload?.summary ||
                    ev.payload?.error ||
                    ev.payload?.model ||
                    ev.payload?.experiment_type ||
                    JSON.stringify(ev.payload).slice(0, 90)
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
