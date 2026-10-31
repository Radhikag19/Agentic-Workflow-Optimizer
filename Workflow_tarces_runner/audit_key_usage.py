"""
Audit script to calculate exact live token consumption per Groq API Key
across all generated Ingestion_Traces and recent benchmark runs.
"""

import json
from pathlib import Path
from collections import defaultdict

key_usage = defaultdict(lambda: {
    'total_tokens': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'calls': 0,
    'workflows': set()
})

trace_dirs = [
    Path('Ingestion_Traces/contract_review'),
    Path('Ingestion_Traces/invoice_processing'),
    Path('Ingestion_Traces/customer_support'),
    Path('Ingestion_Traces/procurement_supplier_selection'),
    Path('Ingestion_Traces/sales_lead_qualification'),
    Path('Ingestion_Traces/recruitment_screening'),
    Path('Ingestion_Traces/inventory_replenishment'),
    Path('Ingestion_Traces/incident_triage'),
    Path('Ingestion_Traces/expense_reimbursement'),
    Path('Ingestion_Traces/market_research'),
]

total_files = 0
for td in trace_dirs:
    if not td.exists():
        continue
    for tf in sorted(list(td.glob('live_*.json'))):
        total_files += 1
        try:
            with open(tf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            wf = data.get('workflow_name', td.name)
            for ex in data.get('agent_executions', []):
                key_id = ex.get('assigned_key_identifier', 'UNKNOWN')
                key_name = 'UNKNOWN'
                for k in range(20, 0, -1):
                    if f'GROQ_API_KEY_{k}' in key_id:
                        key_name = f'GROQ_API_KEY_{k}'
                        break
                
                toks = ex.get('tokens', 0)
                p_toks = ex.get('prompt_tokens', 0)
                c_toks = ex.get('completion_tokens', 0)
                
                key_usage[key_name]['total_tokens'] += toks
                key_usage[key_name]['prompt_tokens'] += p_toks
                key_usage[key_name]['completion_tokens'] += c_toks
                key_usage[key_name]['calls'] += 1
                key_usage[key_name]['workflows'].add(wf)
        except Exception as e:
            print(f'Error reading {tf}: {e}')

print(f'Total Ingestion Trace Files Analyzed: {total_files}')
print('=' * 85)
header = f"{'Key Name':<17} | {'Tokens Used':<12} | {'Daily Limit':<11} | {'Tokens Left':<11} | {'% Used':<8} | {'Calls':<6}"
print(header)
print('=' * 85)

DAILY_LIMIT = 200000
total_all_tokens = 0

for k in range(1, 21):
    k_name = f'GROQ_API_KEY_{k}'
    info = key_usage[k_name]
    used = info['total_tokens']
    total_all_tokens += used
    rem = max(0, DAILY_LIMIT - used)
    pct = (used / DAILY_LIMIT) * 100
    calls = info['calls']
    status_str = "EXHAUSTED" if rem <= 15000 else ("LOW" if rem <= 50000 else "HEALTHY")
    print(f"{k_name:<17} | {used:<12,d} | {DAILY_LIMIT:<11,d} | {rem:<11,d} | {pct:>6.1f}% | {calls:<6}")

print('=' * 85)
print(f"Total Tokens Consumed Across All Keys Today: {total_all_tokens:,d} / 4,000,000 ({total_all_tokens/40000:.1f}%)")
print('=' * 85)

# Also break down by workflow per key
print("\n--- Detailed Breakdown by Workflow per API Key ---")
for k in range(1, 21):
    k_name = f'GROQ_API_KEY_{k}'
    info = key_usage[k_name]
    wfs = ', '.join(sorted(list(info['workflows']))) if info['workflows'] else 'None'
    print(f"{k_name}: {info['total_tokens']:,d} tokens across {info['calls']} calls (Workflows: {wfs})")
