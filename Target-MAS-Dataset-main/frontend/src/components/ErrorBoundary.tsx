import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(err: unknown): ErrorBoundaryState {
    return {
      hasError: true,
      message: err instanceof Error ? err.message : String(err),
    };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="pixel-panel"
          style={{
            padding: 18,
            background: 'var(--badge-fail-bg)',
            color: 'var(--badge-fail-fg)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
            <AlertTriangle size={18} />
            Agentic Workplace Panel Recovery
          </div>
          <div className="mono" style={{ fontSize: 12, marginTop: 8 }}>
            {this.state.message || 'An unexpected UI render error occurred.'}
          </div>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, message: '' })}
            className="pixel-btn"
            style={{ marginTop: 12 }}
          >
            <RotateCcw size={14} /> Retry View
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
