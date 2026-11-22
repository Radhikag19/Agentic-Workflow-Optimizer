"""
Stage 2 (Dimension G): Deterministic Payload Size & Redundancy Analysis.

Detects oversized text/document fields, large JSON objects, repeated fields
across multiple handoffs, byte-identical duplicate values, and static metadata.
Reuses `count_tokens` from `consolidation/token_budget.py` without treating
static declarations as runtime payload measurements.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from optimization_agent.consolidation.token_budget import count_tokens

LARGE_PAYLOAD_TOKEN_THRESHOLD = 400
LARGE_PAYLOAD_BYTE_THRESHOLD = 1600
LARGE_FIELD_TOKEN_THRESHOLD = 150

_DOCUMENT_FIELD_HINTS = {
    "full_document",
    "raw_document",
    "raw_text",
    "document_text",
    "full_text",
    "entire_document",
    "raw_contract",
    "contract_text",
    "transcript",
    "pdf_text",
    "raw_content",
    "raw_ticket_record",
}

_METADATA_FIELD_HINTS = {
    "source_metadata",
    "debug_info",
    "raw_headers",
    "ingestion_metadata",
    "audit_metadata",
    "system_metadata",
    "metadata",
    "trace_metadata",
    "raw_trace",
    "intake_warnings",
    "processing_metadata",
}


def analyze_payloads(
    handoffs: List[Dict[str, Any]],
    model_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze size, tokenization, and redundancy properties across all verified handoffs.
    Attaches `payload_analysis` to each handoff record without claiming removal safety yet.
    """
    field_frequency: Dict[str, int] = {}
    value_signature_freq: Dict[str, int] = {}

    for h in handoffs:
        for f in (h.get("transmitted_fields") or []):
            norm_f = f.strip().lower()
            field_frequency[norm_f] = field_frequency.get(norm_f, 0) + 1
        payload = h.get("actual_payload")
        if isinstance(payload, dict):
            for k, v in payload.items():
                if v in (None, "", [], {}):
                    continue
                sig = f"{k.strip().lower()}::{json.dumps(v, sort_keys=True, ensure_ascii=False)}"
                value_signature_freq[sig] = value_signature_freq.get(sig, 0) + 1

    enriched: List[Dict[str, Any]] = []
    for h in handoffs:
        rec = dict(h)
        payload = rec.get("actual_payload")
        transmitted = list(rec.get("transmitted_fields") or [])
        repeated_across = [f for f in transmitted if field_frequency.get(f.strip().lower(), 0) >= 2]

        large_text_fields: List[str] = []
        large_json_objects: List[str] = []
        duplicate_value_pairs: List[Dict[str, str]] = []
        repeated_metadata_fields: List[str] = []
        repeated_static_fields: List[str] = []
        per_field_metrics: Dict[str, Dict[str, Any]] = {}

        if isinstance(payload, dict) and payload:
            seen_val_map: Dict[str, str] = {}
            for k, v in payload.items():
                f_json = json.dumps({k: v}, sort_keys=True, ensure_ascii=False)
                f_bytes = len(f_json.encode("utf-8"))
                f_tok_info = count_tokens(f_json, model_id)
                f_tokens = int(f_tok_info["tokens"])
                k_low = k.strip().lower()

                per_field_metrics[k] = {
                    "bytes": f_bytes,
                    "tokens": f_tokens,
                    "type": type(v).__name__,
                    "handoff_frequency": field_frequency.get(k_low, 1),
                }

                # Detect oversized text / document payloads
                if isinstance(v, str) and (
                    f_tokens >= LARGE_FIELD_TOKEN_THRESHOLD
                    or (k_low in _DOCUMENT_FIELD_HINTS and f_tokens >= 25)
                ):
                    large_text_fields.append(k)
                elif isinstance(v, (dict, list)) and f_tokens >= LARGE_FIELD_TOKEN_THRESHOLD:
                    large_json_objects.append(k)

                # Detect metadata & static repeated values across multiple handoffs
                sig = f"{k_low}::{json.dumps(v, sort_keys=True, ensure_ascii=False)}"
                if k_low in _METADATA_FIELD_HINTS or (
                    ("metadata" in k_low or "debug" in k_low) and field_frequency.get(k_low, 1) >= 2
                ):
                    repeated_metadata_fields.append(k)
                if value_signature_freq.get(sig, 0) >= 2 and isinstance(v, (dict, list, str)) and f_bytes >= 32:
                    repeated_static_fields.append(k)

                # Detect exact duplicate values across distinct keys in same payload
                if isinstance(v, str):
                    norm_val = json.dumps(v.strip(), ensure_ascii=False) if len(v.strip()) >= 8 else ""
                elif isinstance(v, (dict, list)) and v:
                    norm_val = json.dumps(v, sort_keys=True, ensure_ascii=False)
                else:
                    norm_val = ""

                if norm_val:
                    if norm_val in seen_val_map:
                        duplicate_value_pairs.append({
                            "canonical_field": seen_val_map[norm_val],
                            "duplicate_field": k,
                            "basis": "identical_serialized_value",
                        })
                    else:
                        seen_val_map[norm_val] = k

            orig_tokens = rec.get("payload_tokens") or 0
            orig_bytes = rec.get("payload_size_bytes") or 0
            is_large_payload = bool(
                orig_tokens >= LARGE_PAYLOAD_TOKEN_THRESHOLD
                or orig_bytes >= LARGE_PAYLOAD_BYTE_THRESHOLD
                or large_text_fields
                or large_json_objects
            )
            size_status = "runtime_payload_measured"
        else:
            # Statically inferred only — never fabricate runtime payload tokens or byte counts
            for k in transmitted:
                k_low = k.strip().lower()
                if k_low in _DOCUMENT_FIELD_HINTS:
                    large_text_fields.append(k)
                if k_low in _METADATA_FIELD_HINTS or "metadata" in k_low or "debug" in k_low:
                    repeated_metadata_fields.append(k)
                per_field_metrics[k] = {
                    "bytes": None,
                    "tokens": None,
                    "type": "unknown_static",
                    "handoff_frequency": field_frequency.get(k_low, 1),
                }
            is_large_payload = bool(large_text_fields)
            size_status = "static_only_no_runtime_payload"

        rec["payload_analysis"] = {
            "original_payload_tokens": rec.get("payload_tokens"),
            "original_payload_bytes": rec.get("payload_size_bytes"),
            "transmitted_field_count": len(transmitted),
            "is_large_payload": is_large_payload,
            "large_text_or_document_fields": large_text_fields,
            "large_json_objects": large_json_objects,
            "repeated_across_handoffs": repeated_across,
            "repeated_metadata_fields": sorted(set(repeated_metadata_fields)),
            "repeated_static_fields": sorted(set(repeated_static_fields)),
            "duplicate_value_pairs": duplicate_value_pairs,
            "per_field_metrics": per_field_metrics,
            "size_evidence_status": size_status,
        }
        enriched.append(rec)

    return enriched
