import React from 'react';
import type { ExperimentResultsResponse, WorkflowOverviewResponse } from '../types/workplace';
import type { DimensionViewMode } from '../store/workplaceStore';

interface DimensionLensBannerProps {
  activeDimension: DimensionViewMode;
  overview: WorkflowOverviewResponse | null;
  results: ExperimentResultsResponse | null;
}

export const DimensionLensBanner: React.FC<DimensionLensBannerProps> = ({
  activeDimension,
  overview,
  results,
}) => {
  if (activeDimension === 'all_combined' || activeDimension === 'baseline') {
    return null;
  }

  const approved = (overview?.approved_candidates || []).filter(
    (c) => (c.optimization_type || c.dimension) === activeDimension
  );
  const discovered = overview?.discovered_candidates_by_dimension?.[activeDimension] || [];
  const cands = approved.length > 0 ? approved : discovered;
  const isDiscoveredPreview = approved.length === 0 && discovered.length > 0;
  const evalRow = (results?.comparative_configurations || []).find(
    (r) => r.dimension === activeDimension
  );

  const summaryText = cands
    .map((c) => {
      const sup = c.supporting_evidence || {};
      if (activeDimension === 'communication_data_handoff' && Array.isArray(sup.fields_excluded)) {
        return `${c.candidate_id} (pruned: ${sup.fields_excluded.join(', ')})`;
      }
      if (activeDimension === 'model_selection' && Array.isArray(sup.proposed_candidate_models)) {
        const lastModel = sup.proposed_candidate_models[sup.proposed_candidate_models.length - 1];
        return `${c.candidate_id} (${sup.current_model || '120b'} -> ${lastModel})`;
      }
      return `${c.candidate_id} [${(c.target_agents || []).join(', ')}]`;
    })
    .join(' • ');

  return (
    <div
      className="pixel-panel-inset mono"
      style={{
        padding: '9px 12px',
        fontSize: 11,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <div>
        <strong>Lens Details ({activeDimension.replace(/_/g, ' ')}): </strong>
        {cands.length > 0 ? (
          <span>
            {isDiscoveredPreview ? '[DISCOVERED PREVIEW — NOT YET APPROVED] ' : ''}
            {summaryText}
          </span>
        ) : (
          <span>
            {activeDimension === 'agent_consolidation'
              ? 'Not applied / all desks preserved (consolidation unverified or excluded from scope).'
              : 'No approved or discovered candidate in current scope for this dimension.'}
          </span>
        )}
      </div>

      {evalRow && (
        <span className={`pixel-badge ${evalRow.quality_accepted ? 'pixel-badge-pass' : 'pixel-badge-fail'}`}>
          QUALITY GATE: {String(evalRow.quality_status || (evalRow.quality_accepted ? 'ACCEPTED' : 'REJECTED')).toUpperCase()}
          {Array.isArray(evalRow.quality_reasons) && evalRow.quality_reasons.length > 0
            ? ` (${evalRow.quality_reasons[0]})`
            : ''}
        </span>
      )}
    </div>
  );
};
