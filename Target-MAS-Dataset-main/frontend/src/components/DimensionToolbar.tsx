import React from 'react';
import { Columns, Layers, Cpu, Scissors, FileText, GitMerge, History } from 'lucide-react';
import type { DimensionViewMode } from '../store/workplaceStore';

interface DimensionToolbarProps {
  activeDimension: DimensionViewMode;
  comparisonMode: boolean;
  onSelectDimension: (dim: DimensionViewMode) => void;
  onToggleComparison: (enabled: boolean) => void;
}

const DIMENSION_BUTTONS: { id: DimensionViewMode; label: string; icon: React.ReactNode }[] = [
  { id: 'all_combined', label: 'All Optimized', icon: <Layers size={14} /> },
  { id: 'baseline', label: 'Historical Baseline', icon: <History size={14} /> },
  { id: 'communication_data_handoff', label: '1. Data Handoff', icon: <FileText size={14} /> },
  { id: 'workflow_parallelization', label: '2. Parallelization', icon: <Columns size={14} /> },
  { id: 'model_selection', label: '3. Model Selection', icon: <Cpu size={14} /> },
  { id: 'agent_consolidation', label: '4. Consolidation', icon: <GitMerge size={14} /> },
  { id: 'prompt_optimization', label: '5. Prompt Opt.', icon: <Scissors size={14} /> },
];

export const DimensionToolbar: React.FC<DimensionToolbarProps> = ({
  activeDimension,
  comparisonMode,
  onSelectDimension,
  onToggleComparison,
}) => {
  return (
    <div
      className="pixel-panel"
      style={{
        padding: '10px 12px',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 10,
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        <span className="pixel-title" style={{ fontSize: '9px', marginRight: 4 }}>
          Office Lens:
        </span>
        {DIMENSION_BUTTONS.map((btn) => {
          const active = activeDimension === btn.id;
          return (
            <button
              key={btn.id}
              type="button"
              onClick={() => onSelectDimension(btn.id)}
              className={`pixel-btn ${active ? 'pixel-btn-primary' : ''}`}
              style={{ padding: '6px 9px', fontSize: 12 }}
            >
              {btn.icon}
              {btn.label}
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => onToggleComparison(!comparisonMode)}
        className={`pixel-btn ${comparisonMode ? 'pixel-btn-amber' : ''}`}
        style={{ padding: '6px 11px', fontSize: 12 }}
      >
        <Columns size={14} />
        {comparisonMode ? 'Split Comparison: ON' : 'Compare Before / After'}
      </button>
    </div>
  );
};
