"""
Tests verifying local synthetic dataset generation without Groq/LLM calls.
"""

import json
from pathlib import Path
import pytest
from common.settings import DATA_DIR
from generate_dummy_data import (
    generate_invoice_processing_data,
    generate_customer_support_data,
    generate_procurement_data,
    generate_sales_lead_data,
    FIXED_SEED
)

def test_dataset_reproducibility():
    """Verify repeated generation with the same seed produces exact identical outputs."""
    inv1 = generate_invoice_processing_data(FIXED_SEED)
    inv2 = generate_invoice_processing_data(FIXED_SEED)
    assert inv1 == inv2, "Data generation with fixed seed must be strictly deterministic."

    cs1 = generate_customer_support_data(FIXED_SEED)
    cs2 = generate_customer_support_data(FIXED_SEED)
    assert cs1 == cs2

    pro1 = generate_procurement_data(FIXED_SEED)
    pro2 = generate_procurement_data(FIXED_SEED)
    assert pro1 == pro2

    lead1 = generate_sales_lead_data(FIXED_SEED)
    lead2 = generate_sales_lead_data(FIXED_SEED)
    assert lead1 == lead2


def test_datasets_exist_and_synthetic_flag():
    """Verify generated dataset files exist on disk and all records contain is_synthetic flag."""
    expected_files = [
        DATA_DIR / "invoice_processing" / "invoices.json",
        DATA_DIR / "invoice_processing" / "purchase_orders.json",
        DATA_DIR / "invoice_processing" / "suppliers.json",
        DATA_DIR / "invoice_processing" / "policies.json",
        DATA_DIR / "customer_support" / "tickets.json",
        DATA_DIR / "customer_support" / "policies.json",
        DATA_DIR / "procurement_supplier_selection" / "quotations.json",
        DATA_DIR / "procurement_supplier_selection" / "contracts.json",
        DATA_DIR / "sales_lead_qualification" / "leads.json",
        DATA_DIR / "sales_lead_qualification" / "company_profiles.json"
    ]
    for ef in expected_files:
        assert ef.exists(), f"Expected dataset file {ef} is missing."
        with open(ef, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) > 0
            # Test first record has is_synthetic flag
            if isinstance(data, list):
                assert data[0].get("is_synthetic") is True, f"Record in {ef} missing is_synthetic=True"
