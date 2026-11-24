import React, { useEffect } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Header } from './components/Header';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { OfficePage } from './pages/OfficePage';
import { PerformanceReviewPage } from './pages/PerformanceReviewPage';
import { QualityLabPage } from './pages/QualityLabPage';
import { ReportsPage } from './pages/ReportsPage';
import { useWorkplaceStore } from './store/workplaceStore';

export const App: React.FC = () => {
  const {
    workflows,
    selectedWorkflowId,
    activeTab,
    soundMuted,
    loading,
    error,
    initialize,
    selectWorkflow,
    setActiveTab,
    toggleSoundMuted,
    clearError,
    disconnectSocket,
  } = useWorkplaceStore();

  useEffect(() => {
    void initialize();
    return () => {
      disconnectSocket();
    };
  }, [initialize, disconnectSocket]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', gap: 14, padding: 14 }}>
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        workflows={workflows}
        selectedWorkflowId={selectedWorkflowId}
        onSelectWorkflow={(wfId) => void selectWorkflow(wfId)}
        soundMuted={soundMuted}
        onToggleSound={toggleSoundMuted}
      />

      {error && (
        <div
          className="pixel-panel"
          style={{
            padding: '10px 14px',
            background: 'var(--badge-fail-bg)',
            color: 'var(--badge-fail-fg)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span className="mono" style={{ fontSize: 12 }}>
            {error}
          </span>
          <button type="button" onClick={clearError} className="pixel-btn" style={{ padding: '4px 8px' }}>
            Dismiss
          </button>
        </div>
      )}

      {loading && (
        <div className="pixel-panel mono" style={{ padding: 12, fontSize: 12 }}>
          Loading Target MAS office workspace from FastAPI backend...
        </div>
      )}

      <main style={{ flex: 1 }}>
        <ErrorBoundary>
          {activeTab === 'office' && <OfficePage />}
          {activeTab === 'experiments' && <ExperimentsPage />}
          {activeTab === 'quality_lab' && <QualityLabPage />}
          {activeTab === 'performance_review' && <PerformanceReviewPage />}
          {activeTab === 'reports' && <ReportsPage />}
        </ErrorBoundary>
      </main>
    </div>
  );
};

export default App;
