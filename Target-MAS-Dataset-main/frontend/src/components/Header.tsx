import React from 'react';
import { Volume2, VolumeX, Building2, FlaskConical, ShieldCheck, BarChart3, FileText } from 'lucide-react';
import type { NavTab, WorkflowSummary } from '../types/workplace';

interface HeaderProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  workflows: WorkflowSummary[];
  selectedWorkflowId: string;
  onSelectWorkflow: (wfId: string) => void;
  soundMuted: boolean;
  onToggleSound: () => void;
}

const TABS: { id: NavTab; label: string; icon: React.ReactNode }[] = [
  { id: 'office', label: 'Office', icon: <Building2 size={15} /> },
  { id: 'experiments', label: 'Experiments', icon: <FlaskConical size={15} /> },
  { id: 'quality_lab', label: 'Quality Lab', icon: <ShieldCheck size={15} /> },
  { id: 'performance_review', label: 'Performance Review', icon: <BarChart3 size={15} /> },
  { id: 'reports', label: 'Reports', icon: <FileText size={15} /> },
];

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  workflows,
  selectedWorkflowId,
  onSelectWorkflow,
  soundMuted,
  onToggleSound,
}) => {
  return (
    <header
      className="pixel-panel"
      style={{
        padding: '12px 18px',
        background: '#1f3a2e',
        color: '#faf6ea',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 14,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div
          style={{
            width: 38,
            height: 38,
            background: '#d9822b',
            border: '3px solid #faf6ea',
            display: 'grid',
            placeItems: 'center',
            fontFamily: 'var(--font-pixel)',
            fontSize: 13,
            color: '#1f3a2e',
            fontWeight: 700,
          }}
        >
          AW
        </div>
        <div>
          <h1 className="pixel-title" style={{ fontSize: 13, color: '#faf6ea' }}>
            AGENTIC WORKPLACE
          </h1>
          <div className="mono" style={{ fontSize: 11, color: '#d4e2d4', marginTop: 3 }}>
            Where Multi-Agent Systems go to work.
          </div>
        </div>
      </div>

      <nav style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {TABS.map((t) => {
          const active = activeTab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => onSelectTab(t.id)}
              className="pixel-btn"
              style={{
                background: active ? '#e8b339' : '#2a4e3e',
                color: active ? '#1f3a2e' : '#faf6ea',
                borderColor: '#faf6ea',
                boxShadow: active ? '2px 2px 0 #0d1c15' : 'none',
                padding: '7px 11px',
              }}
            >
              {t.icon}
              {t.label}
            </button>
          );
        })}
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <label className="mono" style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>MAS:</span>
          <select
            aria-label="Target MAS Workflow"
            className="pixel-select"
            value={selectedWorkflowId}
            onChange={(e) => onSelectWorkflow(e.target.value)}
            style={{ fontSize: 12, padding: '5px 8px' }}
          >
            {workflows.map((wf) => (
              <option key={wf.workflow_id} value={wf.workflow_id}>
                {wf.workflow_name} ({wf.agent_count} agents)
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={onToggleSound}
          title={soundMuted ? 'Enable retro sound effects' : 'Mute sound effects'}
          className="pixel-btn"
          style={{
            padding: '6px 9px',
            background: '#2a4e3e',
            color: '#faf6ea',
            borderColor: '#faf6ea',
          }}
        >
          {soundMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
          <span className="mono" style={{ fontSize: 11 }}>
            {soundMuted ? 'Muted' : 'SFX On'}
          </span>
        </button>
      </div>
    </header>
  );
};
