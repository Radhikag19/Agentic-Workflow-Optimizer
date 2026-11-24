import React, { useState } from 'react';
import { FileText, FileCode, Copy, Check } from 'lucide-react';
import { useWorkplaceStore } from '../store/workplaceStore';

export const ReportsPage: React.FC = () => {
  const { activeExperimentId, activeReportBundle } = useWorkplaceStore();
  const [tab, setTab] = useState<'markdown' | 'json'>('markdown');
  const [copied, setCopied] = useState(false);

  if (!activeReportBundle) {
    return (
      <div className="pixel-panel" style={{ padding: 20 }}>
        <div className="pixel-title" style={{ fontSize: 11, marginBottom: 8 }}>
          Experiment & Study Reports
        </div>
        <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
          Select a completed experiment or study in the Experiments tab to view its full generated Markdown and JSON report bundle.
        </div>
      </div>
    );
  }

  const textToCopy =
    tab === 'markdown'
      ? activeReportBundle.markdown_report || ''
      : JSON.stringify(activeReportBundle.json_report, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore clipboard restrictions
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div
        className="pixel-panel"
        style={{
          padding: 14,
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <div>
          <div className="pixel-title" style={{ fontSize: 11 }}>
            Report Artifact Viewer: {activeExperimentId}
          </div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            JSON: {activeReportBundle.json_report_path || 'N/A'} • MD:{' '}
            {activeReportBundle.markdown_report_path || 'N/A'}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            type="button"
            onClick={() => setTab('markdown')}
            className={`pixel-btn ${tab === 'markdown' ? 'pixel-btn-primary' : ''}`}
          >
            <FileText size={14} /> Markdown Report
          </button>
          <button
            type="button"
            onClick={() => setTab('json')}
            className={`pixel-btn ${tab === 'json' ? 'pixel-btn-primary' : ''}`}
          >
            <FileCode size={14} /> Raw JSON
          </button>
          <button type="button" onClick={handleCopy} className="pixel-btn">
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {activeReportBundle.sub_reports && activeReportBundle.sub_reports.length > 0 && (
        <div className="pixel-panel" style={{ padding: 12 }}>
          <div className="pixel-title" style={{ fontSize: 10, marginBottom: 8 }}>
            Generated Per-Configuration Sub-Reports ({activeReportBundle.sub_reports.length})
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {activeReportBundle.sub_reports.map((sr, i) => (
              <span key={i} className="pixel-badge pixel-badge-pass">
                {sr.relative_path || sr.json_report_path}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="pixel-panel" style={{ padding: 14 }}>
        <pre
          className="pixel-panel-inset mono"
          style={{
            margin: 0,
            padding: 14,
            fontSize: 12,
            lineHeight: 1.55,
            maxHeight: 620,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
          }}
        >
          {tab === 'markdown'
            ? activeReportBundle.markdown_report || 'Markdown report not available.'
            : JSON.stringify(activeReportBundle.json_report, null, 2)}
        </pre>
      </div>
    </div>
  );
};
