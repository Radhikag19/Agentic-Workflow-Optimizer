import React from 'react';
import { User, Cpu, Clock, Coins, ArrowRight } from 'lucide-react';
import type { AgentProfile, AgentRuntimeTelemetry } from '../types/workplace';

interface AgentInspectorProps {
  agent: AgentProfile | null;
  telemetry: AgentRuntimeTelemetry | null;
}

export const AgentInspector: React.FC<AgentInspectorProps> = ({ agent, telemetry }) => {
  if (!agent) {
    return (
      <div className="pixel-panel" style={{ padding: 14 }}>
        <div className="pixel-title" style={{ fontSize: '10px', marginBottom: 8 }}>
          Agent Desk Inspector
        </div>
        <div className="mono" style={{ fontSize: 12, color: 'var(--ink-muted)' }}>
          Click any agent sprite or desk on the office floor to inspect role, responsibilities,
          assigned model, dependencies, prompt preview, and baseline vs live telemetry.
        </div>
      </div>
    );
  }

  const responsibilitiesList: string[] = Array.isArray(agent.responsibilities)
    ? agent.responsibilities
    : typeof agent.responsibilities === 'string' && String(agent.responsibilities).trim()
    ? [String(agent.responsibilities)]
    : [];
  const helpersList = Array.isArray(agent.helper_functions) ? agent.helper_functions : [];
  const isHybrid = agent.execution_style === 'hybrid_python_plus_llm' || helpersList.length > 0;

  return (
    <div className="pixel-panel" style={{ padding: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <User size={16} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{agent.agent_name}</div>
            <div className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
              {agent.agent_id} • {agent.role}
            </div>
          </div>
        </div>
        <span
          className={`pixel-badge ${
            telemetry?.state === 'failed' || telemetry?.state === 'blocked'
              ? 'pixel-badge-fail'
              : telemetry?.state === 'working' || telemetry?.state === 'parallel_working'
              ? 'pixel-badge-warn'
              : 'pixel-badge-pass'
          }`}
        >
          {telemetry?.state || 'idle'}
        </span>
      </div>

      <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        <span className={`pixel-badge ${isHybrid ? 'pixel-badge-warn' : 'pixel-badge-pass'}`}>
          {isHybrid ? 'Hybrid: Python Helper + LLM' : 'Pure LLM Reasoning'}
        </span>
        {helpersList.map((hf) => (
          <span key={hf} className="pixel-badge pixel-badge-unavail">
            fn: {hf}()
          </span>
        ))}
      </div>

      <div
        className="pixel-panel-inset mono"
        style={{
          padding: 10,
          marginTop: 10,
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
          gap: 8,
          fontSize: 11,
        }}
      >
        <div>
          <div style={{ color: 'var(--ink-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Cpu size={12} /> Model
          </div>
          <div style={{ fontWeight: 700, marginTop: 2, wordBreak: 'break-word' }}>
            {telemetry?.model_used || agent.model}
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--ink-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Clock size={12} /> Mean Latency
          </div>
          <div style={{ fontWeight: 700, marginTop: 2 }}>
            {telemetry?.latency_seconds != null
              ? `${telemetry.latency_seconds.toFixed(3)}s`
              : agent.historical_mean_latency_seconds != null
              ? `${agent.historical_mean_latency_seconds.toFixed(3)}s (hist)`
              : 'Unavailable'}
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--ink-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Coins size={12} /> Mean Tokens
          </div>
          <div style={{ fontWeight: 700, marginTop: 2 }}>
            {telemetry?.tokens_used != null
              ? Math.round(telemetry.tokens_used)
              : agent.historical_mean_tokens != null
              ? `${Math.round(agent.historical_mean_tokens)} (hist)`
              : 'Unavailable'}
          </div>
        </div>
      </div>

      {telemetry?.error_message && (
        <div
          className="pixel-badge pixel-badge-fail"
          style={{ marginTop: 10, width: '100%', justifyContent: 'flex-start' }}
        >
          Error: {telemetry.error_message}
        </div>
      )}

      <div style={{ marginTop: 10, fontSize: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Data Handoff & Dependencies</div>
        <div className="mono" style={{ fontSize: 11, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <span>Deps: {agent.dependencies.length > 0 ? agent.dependencies.join(', ') : 'None (root)'}</span>
          <ArrowRight size={12} />
          <span>In: {agent.input_fields.join(', ') || 'workflow_input'}</span>
          <ArrowRight size={12} />
          <span>Out: {agent.output_fields.join(', ') || 'result'}</span>
        </div>
      </div>

      {responsibilitiesList.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Responsibilities</div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {responsibilitiesList.slice(0, 4).map((r, i) => (
              <li key={i} style={{ marginBottom: 2 }}>
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {telemetry?.output_preview && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Recorded / Live Output Preview</div>
          <pre
            className="pixel-panel-inset mono"
            style={{
              margin: 0,
              padding: 8,
              fontSize: 10,
              maxHeight: 105,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
            }}
          >
            {telemetry.output_preview}
          </pre>
        </div>
      )}

      {agent.original_prompt_preview && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Prompt / Node Preview</div>
          <pre
            className="pixel-panel-inset mono"
            style={{
              margin: 0,
              padding: 8,
              fontSize: 10,
              maxHeight: 110,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
            }}
          >
            {agent.original_prompt_preview}
          </pre>
        </div>
      )}
    </div>
  );
};
