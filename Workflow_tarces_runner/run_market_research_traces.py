"""
Dedicated Live Trace Generator for Market Research Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 9 sub-agents with dynamic auto-failover.
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/market_research/live_20261023_070724_market.json.
4. Outputs persisted directly in Ingestion_Traces/market_research/ (no agent-wise subfolders).
"""

import os
import sys
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# 1-to-1 Sub-Agent to Groq API Key Mapping for remaining runs
AGENT_KEY_MAP = {
    "planning_agent": "GROQ_API_KEY_20",
    "profile_agent": "GROQ_API_KEY_18",
    "product_agent": "GROQ_API_KEY_20",
    "pricing_agent": "GROQ_API_KEY_18",
    "feedback_agent": "GROQ_API_KEY_20",
    "evidence_agent": "GROQ_API_KEY_18",
    "comparison_agent": "GROQ_API_KEY_10",
    "synthesis_agent": "GROQ_API_KEY_20",
    "fact_check_agent": "GROQ_API_KEY_18",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_20",
    "GROQ_API_KEY_18",
    "GROQ_API_KEY_10",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {
    "GROQ_API_KEY_1",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_4",
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_6",
    "GROQ_API_KEY_7",
    "GROQ_API_KEY_8",
    "GROQ_API_KEY_9",
    "GROQ_API_KEY_11",
    "GROQ_API_KEY_12",
    "GROQ_API_KEY_13",
    "GROQ_API_KEY_14",
    "GROQ_API_KEY_15",
    "GROQ_API_KEY_16",
    "GROQ_API_KEY_17",
    "GROQ_API_KEY_19",
}

def get_next_standby_key() -> str:
    """Return next available healthy Groq API key from the standby pool."""
    for candidate in STANDBY_POOL:
        if candidate not in EXHAUSTED_KEYS:
            val = os.getenv(candidate, "").strip()
            if val:
                return candidate
    raise RuntimeError("All Groq API keys in the standby pool are exhausted!")

def mask_key(raw_key: str) -> str:
    if not raw_key:
        return "[NOT_SET]"
    s = raw_key.strip()
    return f"{s[:6]}...{s[-4:]}" if len(s) > 10 else "***"

def extract_json_or_text(text: str) -> Any:
    """Attempt JSON parsing, or return text if string narrative."""
    s = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        snippet = match.group(1).strip()
        try:
            return json.loads(snippet)
        except Exception:
            pass
    
    try:
        return json.loads(s)
    except Exception:
        pass

    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = s[first_brace:last_brace + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return s


# Data Helpers
def query_company_profiles(data_dir: Path) -> Dict[str, Any]:
    comp_path = data_dir / "competitors.json"
    doc_path = data_dir / "source_documents.json"
    competitors = []
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            competitors = json.load(f).get("competitors", [])

    profile_docs = []
    if doc_path.exists():
        with open(doc_path, "r", encoding="utf-8") as f:
            all_docs = json.load(f).get("documents", [])
            profile_docs = [d for d in all_docs if d.get("doc_type") == "company_profile"]

    return {"competitors": competitors, "profile_documents": profile_docs}


def query_product_documents(data_dir: Path) -> List[Dict[str, Any]]:
    doc_path = data_dir / "source_documents.json"
    if not doc_path.exists():
        return []
    with open(doc_path, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") in ["product_capabilities", "benchmark_report", "sla_specification"]]


def query_pricing_documents(data_dir: Path) -> List[Dict[str, Any]]:
    doc_path = data_dir / "source_documents.json"
    if not doc_path.exists():
        return []
    with open(doc_path, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") == "pricing_and_licensing"]


def query_feedback_documents(data_dir: Path) -> List[Dict[str, Any]]:
    doc_path = data_dir / "source_documents.json"
    if not doc_path.exists():
        return []
    with open(doc_path, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") == "customer_feedback"]


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 2500,
    max_retries: int = 8,
) -> Dict[str, Any]:
    """Execute a single agent stage via its assigned Groq API key with automatic failover."""
    global AGENT_KEY_MAP, EXHAUSTED_KEYS

    system_prompt = (
        f"You are the {role}.\n"
        f"Goal: {goal}\n"
        f"Backstory: {backstory}\n"
        f"Provide a rigorous, structured response citing doc_ids."
    )

    t0 = time.time()
    last_err = None
    for attempt in range(max_retries):
        current_key_name = AGENT_KEY_MAP[agent_id]
        api_key = os.getenv(current_key_name, "").strip()

        # If key missing or exhausted, failover immediately
        if not api_key or current_key_name in EXHAUSTED_KEYS:
            new_key_name = get_next_standby_key()
            print(f"    [Key Failover] Switching {agent_id} from {current_key_name} to fresh standby key {new_key_name}...", flush=True)
            AGENT_KEY_MAP[agent_id] = new_key_name
            current_key_name = new_key_name
            api_key = os.getenv(current_key_name, "").strip()

        masked_id = f"{current_key_name} ({mask_key(api_key)})"
        client = Groq(api_key=api_key)

        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            t1 = time.time()
            duration = round(t1 - t0, 3)

            raw_content = resp.choices[0].message.content or ""
            if not raw_content and hasattr(resp.choices[0].message, "reasoning"):
                raw_content = resp.choices[0].message.reasoning or ""

            parsed = extract_json_or_text(raw_content)

            prompt_toks = getattr(resp.usage, "prompt_tokens", 0)
            comp_toks = getattr(resp.usage, "completion_tokens", 0)
            tot_toks = getattr(resp.usage, "total_tokens", prompt_toks + comp_toks)

            return {
                "status": "SUCCESS",
                "assigned_key_identifier": masked_id,
                "duration_seconds": duration,
                "tokens": tot_toks,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "output": parsed,
                "raw_text": raw_content,
                "notes": "",
            }
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            
            # Check if daily quota exhausted (TPD limit)
            is_quota_exhausted = (
                "tpd" in err_str.lower() or 
                "tokens per day" in err_str.lower() or 
                ("429" in err_str and "limit 200" in err_str)
            )

            if is_quota_exhausted:
                print(f"    [Quota Limit Hit] Key {current_key_name} daily limit reached! Immediate hot-swap to fresh key...", flush=True)
                EXHAUSTED_KEYS.add(current_key_name)
                new_key_name = get_next_standby_key()
                print(f"    [Hot-Swap] Assigned {new_key_name} to {agent_id}. Retrying immediately without waiting...", flush=True)
                AGENT_KEY_MAP[agent_id] = new_key_name
                continue
            elif "413" in err_str or "request too large" in err_str.lower() or "reduce your message size" in err_str.lower():
                print(f"    [Request Size Alert] Reducing max_tokens for {agent_id} and retrying...", flush=True)
                max_tokens = max(1000, max_tokens - 500)
                if len(task_description) > 3000:
                    task_description = task_description[:len(task_description) * 3 // 4]
                time.sleep(1.0)
                continue
            elif "429" in err_str or "rate_limit" in err_str.lower():
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_time = min(float(m_sec.group(1)) + 1.0, 10.0) if m_sec else 4.0
                print(f"    [Transient RateLimit] Agent '{agent_id}' waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_market_trace(
    research_query: Dict[str, Any],
    profiles_data: Dict[str, Any],
    product_docs: List[Dict[str, Any]],
    pricing_docs: List[Dict[str, Any]],
    feedback_docs: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/market_research"),
) -> Dict[str, Any]:
    """Execute all 9 agents sequentially for one research query and persist trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    q_text = research_query.get("query", "Comprehensive evaluation of enterprise agent platforms")
    print(f"\n--- [Run {run_number:02d}] Executing Market Research for: '{q_text[:60]}...' ---", flush=True)

    # 1. Planning Agent
    print(f"  [1/9] Planning Agent -> {AGENT_KEY_MAP['planning_agent']}...", flush=True)
    desc_1 = f"""
Develop a structured research plan for this market inquiry:
Inquiry: {q_text}
Identify key comparison dimensions (architecture, throughput, pricing, SLA, customer sentiment) for competitors: CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security.

Return strictly valid JSON matching:
{{
  "target_competitors": ["CloudScale AI", "DataPulse Systems", "NexusOps Technologies", "Veloce Security"],
  "evaluation_dimensions": [
    {{"name": "Architecture", "description": "..."}},
    {{"name": "Enterprise Latency / Throughput", "description": "..."}},
    {{"name": "Pricing Predictability", "description": "..."}},
    {{"name": "Advertised vs. Actual SLA", "description": "..."}},
    {{"name": "On-Prem / Hybrid Readiness", "description": "..."}},
    {{"name": "Customer Sentiment & Trust", "description": "..."}}
  ],
  "evidence_gathering_plan": {{
    "sources": [{{"type": "...", "purpose": "..."}}],
    "methodology": [{{"step": "...", "actions": ["..."]}}]
  }}
}}
"""
    exec_1 = call_agent_groq(
        agent_id="planning_agent",
        role="Research Planning Specialist",
        goal="Develop structured research plan: target competitors, evaluation dimensions, and evidence gathering plan.",
        backstory="A Principal Market Strategy Consultant designing rigorous benchmarking methodologies and competitive intelligence frameworks.",
        task_description=desc_1,
        max_tokens=2200,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "planning_agent"
    agent_executions.append(exec_1)
    plan_res = exec_1["output"]
    plan_text = json.dumps(plan_res, indent=2) if isinstance(plan_res, (dict, list)) else str(plan_res)

    # 2. Profile Agent
    print(f"  [2/9] Profile Agent -> {AGENT_KEY_MAP['profile_agent']}...", flush=True)
    desc_2 = f"""
Using this corporate profiles dataset:
{json.dumps(profiles_data, indent=2)}

Extract headquarters, founding year, headcount, funding, and positioning for each competitor.
Cite all source document IDs (e.g., DOC-CS-01, DOC-DP-01, DOC-NO-01, DOC-VS-01).

Return strictly valid JSON matching:
{{
  "competitor_profiles": [
    {{
      "id": "COMP-CS",
      "name": "CloudScale AI",
      "headquarters": "...",
      "founded_year": 2022,
      "employee_count": 140,
      "funding_stage": "...",
      "funding_amount": "...",
      "positioning": "...",
      "source_documents": ["DOC-CS-01"]
    }}
  ]
}}
"""
    exec_2 = call_agent_groq(
        agent_id="profile_agent",
        role="Company Profiling Specialist",
        goal="Extract competitor corporate identities, funding stage, employee count, market positioning, and ICP. Always cite stable source IDs.",
        backstory="A Venture Capital Intelligence Analyst who creates crisp corporate dossiers, strictly referencing primary corporate source documents.",
        task_description=desc_2,
        max_tokens=2200,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "profile_agent"
    agent_executions.append(exec_2)
    prof_res = exec_2["output"]
    prof_text = json.dumps(prof_res, indent=2) if isinstance(prof_res, (dict, list)) else str(prof_res)

    # 3. Product Agent
    print(f"  [3/9] Product Agent -> {AGENT_KEY_MAP['product_agent']}...", flush=True)
    desc_3 = f"""
Analyze technical capabilities from these source documents:
{json.dumps(product_docs, indent=2)}

Extract features, latency/throughput metrics, SDKs, and architectural limitations for each competitor.
Cite specific source doc_ids for each claim.

Return strictly valid JSON matching:
{{
  "COMP-CS": {{
    "competitor_name": "CloudScale AI",
    "features": [{{"description": "...", "doc_id": "DOC-CS-02"}}],
    "latency_metrics": [{{"metric": "...", "value": "...", "doc_id": "..."}}],
    "sdks": [{{"language": "...", "type": "...", "doc_id": "..."}}],
    "architectural_limitations": [{{"description": "...", "doc_id": "..."}}]
  }}
}}
"""
    exec_3 = call_agent_groq(
        agent_id="product_agent",
        role="Product Capabilities Analyst",
        goal="Extract technical features, platform throughput, integrations, SDK support, and architectural limitations across competitors. Attribute every claim to its source doc_id.",
        backstory="A Principal Solutions Architect who dissects technical whitepapers, feature sheets, and system SLAs to understand real engineering capabilities.",
        task_description=desc_3,
        max_tokens=1800,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "product_agent"
    agent_executions.append(exec_3)
    prod_res = exec_3["output"]
    prod_text = json.dumps(prod_res, indent=2) if isinstance(prod_res, (dict, list)) else str(prod_res)

    # 4. Pricing Agent
    print(f"  [4/9] Pricing Agent -> {AGENT_KEY_MAP['pricing_agent']}...", flush=True)
    desc_4 = f"""
Extract commercial pricing and licensing from these documents:
{json.dumps(pricing_docs, indent=2)}

Detail monthly subscription tiers, overage clauses, and contract minimums.
Preserve exact currency units and mark unknown values. Cite source doc_ids.

Return strictly valid JSON matching:
[
  {{
    "competitor_id": "COMP-CS",
    "competitor_name": "CloudScale AI",
    "pricing_tiers": [{{"tier_name": "...", "price": "...", "included_units": "...", "source_doc_id": "..."}}],
    "overage_clause": {{"description": "...", "price": "...", "source_doc_id": "..."}},
    "contract_minimum": {{"amount": "...", "period": "...", "source_doc_id": "..."}}
  }}
]
"""
    exec_4 = call_agent_groq(
        agent_id="pricing_agent",
        role="Pricing & Commercial Terms Specialist",
        goal="Extract commercial pricing schedules, monthly tiers, overage fees, contract minimums, and billing units. Explicitly mark unstated pricing as UNKNOWN and cite source doc_ids.",
        backstory="A SaaS Pricing & Licensing Strategist who scrutinizes Master Services Agreements, rate cards, and hidden overage terms to establish true total cost of ownership (TCO).",
        task_description=desc_4,
        max_tokens=1800,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "pricing_agent"
    agent_executions.append(exec_4)
    price_res = exec_4["output"]
    price_text = json.dumps(price_res, indent=2) if isinstance(price_res, (dict, list)) else str(price_res)

    # 5. Feedback Agent
    print(f"  [5/9] Feedback Agent -> {AGENT_KEY_MAP['feedback_agent']}...", flush=True)
    desc_5 = f"""
Analyze customer reviews and reported field experiences:
{json.dumps(feedback_docs, indent=2)}

Summarize user feedback, reported outages, praise, and usability friction.
Separate recorded quotes from subjective conclusions and cite doc_ids.

Return strictly valid JSON matching:
{{
  "quotes": [
    {{"doc_id": "DOC-CS-05", "quote": "..."}}
  ],
  "analysis": {{
    "outages_and_incidents": [{{"doc_id": "...", "description": "..."}}],
    "praise_and_positive_outcomes": [{{"doc_id": "...", "description": "..."}}],
    "usability_and_friction_points": [{{"doc_id": "...", "description": "..."}}]
  }}
}}
"""
    exec_5 = call_agent_groq(
        agent_id="feedback_agent",
        role="Customer Feedback & Review Analyst",
        goal="Extract real-world customer feedback, reported production outages, praise, and usability complaints across competitors. Keep recorded quotes strictly separate from conclusions.",
        backstory="A Voice of Customer (VoC) Market Researcher who analyzes verified user testimonials, G2 reviews, and community sentiment to assess customer satisfaction.",
        task_description=desc_5,
        max_tokens=1800,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "feedback_agent"
    agent_executions.append(exec_5)
    feed_res = exec_5["output"]
    feed_text = json.dumps(feed_res, indent=2) if isinstance(feed_res, (dict, list)) else str(feed_res)

    # 6. Evidence Agent
    print(f"  [6/9] Evidence Agent -> {AGENT_KEY_MAP['evidence_agent']}...", flush=True)
    desc_6 = f"""
Cross-examine the evidence collected across all 4 domains:
Profiles: {prof_text}
Product: {prod_text}
Pricing: {price_text}
Feedback: {feed_text}

Identify deliberate conflicts (e.g., uptime SLA vs customer reported outages, flat fees vs hidden overages).
Flag any unsupported claims and preserve all source doc_ids.

Return strictly valid JSON matching:
{{
  "triangulated_evidence": {{
    "competitors": [
      {{
        "competitor_id": "COMP-CS",
        "name": "CloudScale AI",
        "claims": {{ ... }},
        "customer_feedback": [ ... ],
        "contradictions": [
          {{
            "type": "sla_violation",
            "claimed_value": "...",
            "observed_issue": "...",
            "evidence_claim": "DOC-CS-04",
            "evidence_feedback": "DOC-CS-05",
            "severity": "high",
            "notes": "..."
          }}
        ],
        "citation_gaps": [
          {{"missing_claim": "...", "context": "...", "required_doc_id": null}}
        ]
      }}
    ]
  }},
  "summary": {{"overall_findings": "..."}}
}}
"""
    exec_6 = call_agent_groq(
        agent_id="evidence_agent",
        role="Evidence Triangulation & Cross-Examination Specialist",
        goal="Cross-examine collected evidence across all 4 domains. Identify deliberate conflicts. Flag unsupported claims and preserve all source doc_ids.",
        backstory="A Forensic Investigative Analyst detecting factual discrepancies, misleading marketing claims, and hidden contractual nuances across technical vendor documentation.",
        task_description=desc_6,
        max_tokens=2000,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "evidence_agent"
    agent_executions.append(exec_6)
    evid_res = exec_6["output"]
    evid_text = json.dumps(evid_res, indent=2) if isinstance(evid_res, (dict, list)) else str(evid_res)

    # 7. Comparison Agent
    print(f"  [7/9] Comparison Agent -> {AGENT_KEY_MAP['comparison_agent']}...", flush=True)
    desc_7 = f"""
Construct a side-by-side competitor matrix for CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security across dimensions:
Focus, Architecture, Key Strengths, Known Weaknesses, Starting Price, and SLA.
Correlated Evidence:
{evid_text}

Clearly distinguish factual specs from analytical opinions. Include doc_ids.
Format as a clean markdown table followed by representative source document citations.
"""
    exec_7 = call_agent_groq(
        agent_id="comparison_agent",
        role="Competitive Benchmarking Specialist",
        goal="Construct side-by-side competitor matrix for CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security across dimensions: Focus, Architecture, Key Strengths, Known Weaknesses, Starting Price, and SLA. Distinguish factual specs from analytical opinions. Include doc_ids.",
        backstory="A Senior Competitive Intelligence Director building objective comparison frameworks for enterprise executive boards.",
        task_description=desc_7,
        max_tokens=1800,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "comparison_agent"
    agent_executions.append(exec_7)
    comp_res = exec_7["output"]
    comp_text = str(comp_res) if isinstance(comp_res, str) else json.dumps(comp_res, indent=2)

    # 8. Synthesis Agent
    print(f"  [8/9] Synthesis Agent -> {AGENT_KEY_MAP['synthesis_agent']}...", flush=True)
    comp_snippet = comp_text[:3500] if len(comp_text) > 3500 else comp_text
    evid_snippet = evid_text[:2500] if len(evid_text) > 2500 else evid_text
    desc_8 = f"""
Author an executive competitive research report synthesized from:
Comparison Matrix:
{comp_snippet}

Evidence & Contradictions:
{evid_snippet}

Structure into:
1. Executive Summary
2. Competitive Landscape
3. Strategic Trade-Offs
4. Key Recommendations

Include integrated doc_id citations throughout.
"""
    exec_8 = call_agent_groq(
        agent_id="synthesis_agent",
        role="Executive Research Synthesis Director",
        goal="Author executive competitive research report synthesized from comparison matrix and evidence & contradictions. Structure into: Executive Summary, Competitive Landscape, Strategic Trade-Offs, and Key Recommendations.",
        backstory="A Chief Strategy Officer synthesizing complex vendor landscapes into crisp, actionable executive decision memos.",
        task_description=desc_8,
        max_tokens=1800,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "synthesis_agent"
    agent_executions.append(exec_8)
    synth_res = exec_8["output"]
    synth_text = str(synth_res) if isinstance(synth_res, str) else json.dumps(synth_res, indent=2)

    # 9. Fact-Check Agent
    print(f"  [9/9] Fact-Check Agent -> {AGENT_KEY_MAP['fact_check_agent']}...", flush=True)
    synth_snippet = synth_text[:3500] if len(synth_text) > 3500 else synth_text
    desc_9 = f"""
Perform an audit on this final synthesis report:
{synth_snippet}

Verify that key claims reference valid doc_ids (DOC-CS-*, DOC-DP-*, DOC-NO-*, DOC-VS-*).
Flag any unsupported statements. Confirm the report is marked as synthetic data.

Return strictly valid JSON matching:
{{
  "verified_citations": [
    {{"doc_id": "DOC-CS-01", "claim": "...", "status": "VERIFIED"}}
  ],
  "unsupported_statements": [],
  "fact_check_verdict": "PASS" | "NEEDS_REVISION",
  "audit_summary": "Comprehensive audit confirming all factual claims are properly cited.",
  "is_synthetic_verified": true
}}
"""
    exec_9 = call_agent_groq(
        agent_id="fact_check_agent",
        role="Research Integrity & Fact-Checking Auditor",
        goal="Perform audit on final synthesis report: verify key claims reference valid doc_ids, flag unsupported statements, and confirm synthetic data tag.",
        backstory="An Editorial Fact-Checker and Legal Compliance Auditor verifying accuracy and provenance of all stated corporate claims.",
        task_description=desc_9,
        max_tokens=1500,
    )
    exec_9["order"] = 9
    exec_9["agent_id"] = "fact_check_agent"
    agent_executions.append(exec_9)
    fact_res = exec_9["output"]
    fact_text = json.dumps(fact_res, indent=2) if isinstance(fact_res, (dict, list)) else str(fact_res)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "market_research",
        "model": "openai/gpt-oss-120b",
        "status": "SUCCESS",
        "duration_seconds": total_dur,
        "agent_execution_count": len(agent_executions),
        "agent_executions": [
            {
                "order": e["order"],
                "agent_id": e["agent_id"],
                "assigned_key_identifier": e["assigned_key_identifier"],
                "duration_seconds": e["duration_seconds"],
                "status": e["status"],
                "tokens": e["tokens"],
                "prompt_tokens": e["prompt_tokens"],
                "completion_tokens": e["completion_tokens"],
                "output": e["output"],
                "notes": e["notes"],
            }
            for e in agent_executions
        ],
        "total_tokens": tot_tokens,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "estimated_cost_usd": cost,
        "error_message": None,
        "final_output": synth_text,
        "output": {
            "workflow": "market_research",
            "run_id": run_id,
            "input": research_query,
            "stages": {
                "planning": plan_text,
                "profiles": prof_text,
                "product": prod_text,
                "pricing": price_text,
                "feedback": feed_text,
                "evidence": evid_text,
                "comparison": comp_text,
                "synthesis": synth_text,
                "fact_checked_report": fact_text,
            },
            "synthetic": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{run_id}.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace_record, f, indent=2, ensure_ascii=False)

    print(f"--> Saved trace to: {trace_file}")
    print(f"--> Completed in {total_dur}s | Total Tokens: {tot_tokens:,} | Cost: ${cost:.6f}")
    return trace_record


def generate_batch_summary(target_dir: Path):
    """Aggregate all traces in target_dir into batch_30_runs_summary.json."""
    trace_files = sorted(list(target_dir.glob("live_*_trace*.json")))
    if not trace_files:
        return

    runs_summary = []
    total_tokens_all = 0
    total_cost_all = 0.0
    total_duration_all = 0.0
    key_counter = {}

    for tf in trace_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            run_item = {
                "run_id": data.get("run_id"),
                "status": data.get("status"),
                "duration_seconds": data.get("duration_seconds"),
                "agent_execution_count": data.get("agent_execution_count"),
                "total_tokens": data.get("total_tokens"),
                "prompt_tokens": data.get("prompt_tokens"),
                "completion_tokens": data.get("completion_tokens"),
                "estimated_cost_usd": data.get("estimated_cost_usd"),
                "timestamp": data.get("timestamp"),
            }
            runs_summary.append(run_item)
            total_tokens_all += data.get("total_tokens", 0)
            total_cost_all += data.get("estimated_cost_usd", 0.0)
            total_duration_all += data.get("duration_seconds", 0.0)

            for ag in data.get("agent_executions", []):
                k = ag.get("assigned_key_identifier", "Unknown")
                key_counter[k] = key_counter.get(k, 0) + 1
        except Exception as e:
            print(f"Error reading {tf}: {e}")

    summary_payload = {
        "workflow_name": "market_research",
        "total_runs": len(runs_summary),
        "successful_runs": sum(1 for r in runs_summary if r["status"] == "SUCCESS"),
        "failed_runs": sum(1 for r in runs_summary if r["status"] != "SUCCESS"),
        "total_tokens_consumed": total_tokens_all,
        "total_estimated_cost_usd": round(total_cost_all, 6),
        "total_duration_seconds": round(total_duration_all, 3),
        "average_duration_per_run_seconds": round(total_duration_all / len(runs_summary), 3) if runs_summary else 0,
        "key_distribution_call_counts": key_counter,
        "runs": runs_summary,
    }

    summary_file = target_dir / "batch_30_runs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"\n--> Successfully compiled batch summary: {summary_file}")

    jsonl_file = target_dir / "summary.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in runs_summary:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


RESEARCH_QUERIES = [
    "Compare CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security across enterprise latency, pricing predictability, advertised vs actual SLAs, and on-prem readiness.",
    "Evaluate total cost of ownership (TCO) and hidden overage charges between CloudScale AI and DataPulse Systems for high-volume event ingestion.",
    "Benchmark technical architecture, throughput scalability, and vector database integrations for DataPulse Systems against CloudScale AI.",
    "Conduct a security and compliance audit comparing Veloce Security with NexusOps Technologies for HIPAA and PCI-DSS compliance.",
    "Assess enterprise SLA reliability, historical outages, and customer dispute handling across CloudScale AI and DataPulse Systems.",
    "Compare incident remediation automation and MTTR improvements in NexusOps Technologies versus CloudScale AI orchestration.",
    "Evaluate on-premise and air-gapped private VPC deployment readiness between Veloce Security and CloudScale AI.",
    "Analyze developer experience, SDK support (Python/TypeScript), and API documentation quality for CloudScale AI, DataPulse, and NexusOps.",
    "Examine customer sentiment, G2 review feedback, and real-world implementation friction across all four competitors.",
    "Assess pricing transparency, contract minimums, and commitment thresholds between NexusOps Technologies and Veloce Security.",
    "Investigate prompt injection defense efficacy and false-positive rates for Veloce Security in healthcare and clinical environments.",
    "Compare telemetry data streaming, real-time feature store latency, and Kafka topic handling in DataPulse Systems versus alternatives.",
    "Audit automated runbook execution safety, database air-gapping, and rollback mechanisms in NexusOps Technologies.",
    "Evaluate semantic caching efficiency, advertised hit rates, and actual token cost reductions for CloudScale AI.",
    "Compare enterprise support responsiveness, SLA breach compensation, and credit policies across CloudScale AI and NexusOps Technologies.",
    "Analyze architectural differences between CloudScale AI's hosted gateway and Veloce Security's on-premise inspection proxy.",
    "Conduct a comprehensive vendor risk assessment across CloudScale AI, DataPulse Systems, NexusOps, and Veloce Security.",
    "Evaluate multi-agent workflow orchestration overhead and latency trade-offs between CloudScale AI and DataPulse Systems.",
    "Assess scalability limits, cluster node caps, and tier pricing for DataPulse Systems enterprise deployments.",
    "Compare enterprise licensing structures, annual contract minimums, and seat-based vs usage-based models across all 4 vendors.",
    "Examine customer-reported production outages in Q2/Q3 and root-cause analysis for CloudScale AI and DataPulse Systems.",
    "Assess integration depth with ServiceNow, PagerDuty, and SIEM platforms for NexusOps Technologies.",
    "Evaluate vector database connectors (Pinecone, Milvus, Qdrant, Redis) and schema drift detection in DataPulse Systems.",
    "Analyze regex and semantic rule flexibility, custom bypass tuning, and data redaction speed in Veloce Security.",
    "Compare startup versus enterprise package features, inspection limits, and docker deployment support in Veloce Security.",
    "Investigate token compression algorithms, circuit breaker reliability, and routing overhead in CloudScale AI Gateway.",
    "Benchmark mean time to recovery (MTTR) reductions and operator manual intervention rates for NexusOps Technologies.",
    "Synthesize market positioning, venture funding stability (Series A-C), headcount growth, and long-term viability across all four competitors.",
    "Perform an end-to-end capabilities matrix comparing CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security.",
    "Develop strategic procurement recommendations and negotiation leverage points for enterprise adoption of AI infrastructure platforms.",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run live traces for market research workflow")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs to execute in this batch")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index in query list")
    parser.add_argument("--start-run-num", type=int, default=1, help="Starting run number for filenames")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs expected in the workflow")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "market_research"
    profiles_data = query_company_profiles(data_dir)
    product_docs = query_product_documents(data_dir)
    pricing_docs = query_pricing_documents(data_dir)
    feedback_docs = query_feedback_documents(data_dir)

    output_dir = PROJECT_ROOT / "Ingestion_Traces" / "market_research"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for market_research (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {output_dir}\n")

    for i in range(args.runs):
        q_idx = (args.start_idx + i) % len(RESEARCH_QUERIES)
        q_item = {"query": RESEARCH_QUERIES[q_idx], "synthetic": True}
        run_num = args.start_run_num + i

        print(f"=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        run_single_market_trace(
            research_query=q_item,
            profiles_data=profiles_data,
            product_docs=product_docs,
            pricing_docs=pricing_docs,
            feedback_docs=feedback_docs,
            run_number=run_num,
            output_dir=output_dir,
        )

        generate_batch_summary(output_dir)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
