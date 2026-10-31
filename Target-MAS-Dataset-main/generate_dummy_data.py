"""
Synthetic Dataset Generator for Agentic Workflow Optimizer.

STRICT REQUIREMENT COMPLIANCE:
- 100% Local Python generation.
- ZERO LLM calls, ZERO external API calls, ZERO Groq dependencies.
- Completely offline, reproducible via fixed random seed.
- Generates realistic synthetic datasets with consistent cross-entity relationships
  and deliberate benchmark edge cases.
"""

import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

FIXED_SEED = 42

BASE_DATA_DIR = Path(__file__).parent / "data"

def ensure_dirs():
    for sub in [
        "invoice_processing",
        "customer_support",
        "procurement_supplier_selection",
        "sales_lead_qualification",
    ]:
        (BASE_DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Invoice Processing and Accounts Payable Datasets
# ==============================================================================
def generate_invoice_processing_data(seed=FIXED_SEED):
    rng = random.Random(seed)
    
    # 1. Suppliers
    suppliers = [
        {"supplier_id": "SUP-101", "name": "Apex Cloud Systems", "tax_id": "US-8829101", "payment_terms": "Net 30", "status": "active", "rating": "A", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-102", "name": "Global Office Supplies Co.", "tax_id": "US-4492019", "payment_terms": "Net 15", "status": "active", "rating": "B", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-103", "name": "Prime Industrial Machinery", "tax_id": "US-9912834", "payment_terms": "Net 45", "status": "active", "rating": "A", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-104", "name": "CyberShield Security Ltd.", "tax_id": "US-3382910", "payment_terms": "Net 30", "status": "active", "rating": "A", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-105", "name": "NextGen Logistics Corp.", "tax_id": "US-1102934", "payment_terms": "Net 60", "status": "under_review", "rating": "C", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-106", "name": "Vanguard Hardware Inc.", "tax_id": "US-5548291", "payment_terms": "Net 30", "status": "active", "rating": "B", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-107", "name": "Echo Data Solutions", "tax_id": "US-7719283", "payment_terms": "Net 15", "status": "active", "rating": "A", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-108", "name": "Quantum Facilities Management", "tax_id": "US-2291048", "payment_terms": "Net 30", "status": "active", "rating": "B", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-109", "name": "Omni Packaging Works", "tax_id": "US-6610294", "payment_terms": "Net 30", "status": "suspended", "rating": "D", "currency": "USD", "is_synthetic": True},
        {"supplier_id": "SUP-110", "name": "Silverline Logistics", "tax_id": "US-9901827", "payment_terms": "Net 45", "status": "active", "rating": "B", "currency": "USD", "is_synthetic": True},
    ]

    # 2. Accounts Payable Policies
    policies = [
        {
            "policy_id": "POL-AP-001",
            "name": "Three-Way Matching Policy",
            "rule": "Every invoice above $1,000 must strictly match a valid Purchase Order in supplier ID, line items, unit price within +/- 2%, and quantity.",
            "threshold": 1000.0,
            "severity": "high",
            "is_synthetic": True
        },
        {
            "policy_id": "POL-AP-002",
            "name": "Mandatory Tax ID & Supplier Validation",
            "rule": "Invoices from unapproved or suspended suppliers, or invoices missing valid tax IDs, cannot be approved.",
            "threshold": 0.0,
            "severity": "critical",
            "is_synthetic": True
        },
        {
            "policy_id": "POL-AP-003",
            "name": "High Value Approval Threshold",
            "rule": "Invoices totaling over $25,000 require executive financial review and cannot be auto-recommended without explicit signoff.",
            "threshold": 25000.0,
            "severity": "medium",
            "is_synthetic": True
        },
        {
            "policy_id": "POL-AP-004",
            "name": "Early Due Date & Late Submission Window",
            "rule": "Invoice due date must be at least 10 business days after invoice issue date unless discount terms apply.",
            "threshold": 10.0,
            "severity": "low",
            "is_synthetic": True
        },
        {
            "policy_id": "POL-AP-005",
            "name": "Strict Arithmetic Accuracy",
            "rule": "Calculated line items total, taxes, and discounts must exactly equal the stated invoice total within $0.05 rounding.",
            "threshold": 0.05,
            "severity": "critical",
            "is_synthetic": True
        }
    ]

    # 3. Purchase Orders
    purchase_orders = []
    base_date = datetime(2026, 8, 1)
    catalog = [
        ("Cloud Infrastructure Tier A", 1200.0),
        ("High-Performance Server Blade", 3500.0),
        ("Ergonomic Office Chair", 250.0),
        ("Bulk Whiteboard Markers Pack", 45.0),
        ("Industrial Lathe Component", 4200.0),
        ("Annual Security License Seat", 180.0),
        ("Pallet Storage Monthly Rental", 650.0),
        ("SSD NVMe 2TB Enterprise", 220.0),
        ("Corrugated Shipping Cartons (500)", 175.0),
        ("Automated Conveyor Belt Sensor", 890.0)
    ]

    for po_idx in range(1, 121):
        sup = suppliers[(po_idx - 1) % len(suppliers)]
        item_sample = rng.sample(catalog, rng.randint(1, 3))
        lines = []
        subtotal = 0.0
        for l_idx, (desc, price) in enumerate(item_sample, 1):
            qty = rng.randint(2, 20)
            line_tot = round(qty * price, 2)
            subtotal += line_tot
            lines.append({
                "line_number": l_idx,
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "line_total": line_tot
            })
        tax = round(subtotal * 0.08, 2)
        total = round(subtotal + tax, 2)
        po_date = (base_date + timedelta(days=po_idx * 2)).strftime("%Y-%m-%d")

        purchase_orders.append({
            "po_id": f"PO-2026-{po_idx:04d}",
            "supplier_id": sup["supplier_id"],
            "supplier_name": sup["name"],
            "date": po_date,
            "currency": "USD",
            "lines": lines,
            "subtotal": subtotal,
            "tax": tax,
            "total_amount": total,
            "status": "approved",
            "is_synthetic": True
        })

    # 4. Historical Invoices (to test duplicate detection)
    historical_invoices = []
    for h_idx in range(1, 101):
        po = purchase_orders[h_idx % len(purchase_orders)]
        inv_date = (datetime.strptime(po["date"], "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
        historical_invoices.append({
            "invoice_id": f"INV-HIST-{h_idx:04d}",
            "invoice_number": f"INV-HIST-{h_idx:04d}",
            "supplier_id": po["supplier_id"],
            "po_reference": po["po_id"],
            "invoice_date": inv_date,
            "total_amount": po["total_amount"],
            "status": "paid",
            "is_synthetic": True
        })

    # 5. Invoices (Current batch under processing - includes normal cases + edge cases)
    invoices = []
    for i_idx in range(1, 201):
        po = purchase_orders[(i_idx - 1) % len(purchase_orders)]
        inv_id = f"INV-2026-{i_idx:04d}"
        inv_date_dt = datetime.strptime(po["date"], "%Y-%m-%d") + timedelta(days=rng.randint(3, 14))
        inv_date = inv_date_dt.strftime("%Y-%m-%d")
        due_date = (inv_date_dt + timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Deep copy lines from PO
        lines = [dict(l) for l in po["lines"]]
        subtotal = po["subtotal"]
        tax = po["tax"]
        discount = 0.0
        total_amount = po["total_amount"]
        po_ref = po["po_id"]
        supplier_id = po["supplier_id"]
        supplier_name = po["supplier_name"]
        test_case_tag = "normal"

        # Deliberately inject test cases for optimizer benchmarking
        if i_idx == 10:
            # Arithmetic error test case
            total_amount = round(total_amount + 150.00, 2)
            test_case_tag = "arithmetic_mismatch"
        elif i_idx == 20:
            # Exact duplicate test case
            inv_id = historical_invoices[0]["invoice_number"]
            supplier_id = historical_invoices[0]["supplier_id"]
            total_amount = historical_invoices[0]["total_amount"]
            po_ref = historical_invoices[0]["po_reference"]
            test_case_tag = "exact_duplicate"
        elif i_idx == 30:
            # Missing PO reference
            po_ref = ""
            test_case_tag = "missing_po"
        elif i_idx == 40:
            # Suspended supplier
            supplier_id = "SUP-109"
            supplier_name = "Omni Packaging Works"
            test_case_tag = "policy_violation_suspended_supplier"
        elif i_idx == 50:
            # High value threshold exceeding $25k
            lines[0]["quantity"] = 30
            lines[0]["line_total"] = round(lines[0]["quantity"] * lines[0]["unit_price"], 2)
            subtotal = sum(l["line_total"] for l in lines)
            tax = round(subtotal * 0.08, 2)
            total_amount = round(subtotal + tax, 2)
            test_case_tag = "high_value_policy_threshold"
        elif i_idx == 60:
            # Quantity mismatch against PO
            lines[0]["quantity"] = lines[0]["quantity"] + 15
            lines[0]["line_total"] = round(lines[0]["quantity"] * lines[0]["unit_price"], 2)
            subtotal = sum(l["line_total"] for l in lines)
            tax = round(subtotal * 0.08, 2)
            total_amount = round(subtotal + tax, 2)
            test_case_tag = "po_quantity_mismatch"
        elif i_idx == 70:
            # Unit price mismatch against PO
            lines[0]["unit_price"] = round(lines[0]["unit_price"] * 1.25, 2)
            lines[0]["line_total"] = round(lines[0]["quantity"] * lines[0]["unit_price"], 2)
            subtotal = sum(l["line_total"] for l in lines)
            tax = round(subtotal * 0.08, 2)
            total_amount = round(subtotal + tax, 2)
            test_case_tag = "po_price_mismatch"

        invoices.append({
            "invoice_id": inv_id,
            "invoice_number": inv_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "invoice_date": inv_date,
            "due_date": due_date,
            "currency": "USD",
            "po_reference": po_ref,
            "line_items": lines,
            "subtotal": subtotal,
            "tax": tax,
            "discount": discount,
            "total_amount": total_amount,
            "test_case_tag": test_case_tag,
            "is_synthetic": True
        })

    return {
        "suppliers": suppliers,
        "policies": policies,
        "purchase_orders": purchase_orders,
        "historical_invoices": historical_invoices,
        "invoices": invoices
    }


# ==============================================================================
# 2. Customer Support Ticket Resolution Datasets
# ==============================================================================
def generate_customer_support_data(seed=FIXED_SEED):
    rng = random.Random(seed)

    customers = [
        {"customer_id": f"CUST-{idx:04d}", "name": f"Enterprise Client {idx}", "tier": rng.choice(["Enterprise", "Pro", "Standard"]), "joined_date": "2024-03-15", "is_synthetic": True}
        for idx in range(1, 101)
    ]

    policies = [
        {
            "policy_id": "SUP-POL-101",
            "category": "Refund Policy",
            "conditions": "Full refunds permitted within 30 days of billing for unused services. Beyond 30 days, pro-rated service credits require Senior Manager approval.",
            "sla_hours": 24,
            "is_synthetic": True
        },
        {
            "policy_id": "SUP-POL-102",
            "category": "Account Access & Password Reset",
            "conditions": "User must verify email address via 2FA token before account unlock. Agents must never provide temporary passwords over plain chat.",
            "sla_hours": 2,
            "is_synthetic": True
        },
        {
            "policy_id": "SUP-POL-103",
            "category": "Hardware & Shipping Damage",
            "conditions": "Damaged goods must be reported within 14 calendar days of delivery with photos. Expedited replacement dispatched immediately upon claim verification.",
            "sla_hours": 12,
            "is_synthetic": True
        },
        {
            "policy_id": "SUP-POL-104",
            "category": "Service Outage Compensation",
            "conditions": "Enterprise SLA uptime guarantee is 99.9%. Service downtime exceeding 1 hour qualifies for 10% monthly service credit.",
            "sla_hours": 4,
            "is_synthetic": True
        },
        {
            "policy_id": "SUP-POL-105",
            "category": "Billing Dispute & Duplicate Charges",
            "conditions": "Duplicate billing charges must be refunded to the original payment method within 3 to 5 business days without requiring ticket escalation.",
            "sla_hours": 8,
            "is_synthetic": True
        }
    ]

    troubleshooting_guides = [
        {
            "guide_id": "TSG-201",
            "issue_category": "account_access",
            "steps": [
                "Verify customer identity with registered email and phone.",
                "Check account lock status in auth registry.",
                "Send self-service password reset link with 15-minute expiry.",
                "Instruct user to clear browser cookies and try incognito mode."
            ],
            "is_synthetic": True
        },
        {
            "guide_id": "TSG-202",
            "issue_category": "billing",
            "steps": [
                "Locate transaction ID in the synthetic billing ledger.",
                "Confirm charge timestamp and payment gateway authorization code.",
                "If duplicate transaction ID detected within 10 minutes, issue reversal transaction.",
                "Send confirmation receipt to customer."
            ],
            "is_synthetic": True
        },
        {
            "guide_id": "TSG-203",
            "issue_category": "technical_failure",
            "steps": [
                "Gather error code and stack trace from customer log submission.",
                "Check cluster health dashboard for node failures.",
                "Advise client to regenerate API token if 401 Unauthorized.",
                "Escalate to engineering if API endpoint returns 502/503."
            ],
            "is_synthetic": True
        },
        {
            "guide_id": "TSG-204",
            "issue_category": "delivery",
            "steps": [
                "Confirm carrier tracking number and shipping status.",
                "Verify delivery address against order confirmation.",
                "If marked delivered but not received, initiate carrier investigation trace."
            ],
            "is_synthetic": True
        }
    ]

    historical_resolutions = [
        {
            "case_id": f"HIST-RES-{i:04d}",
            "ticket_category": rng.choice(["account_access", "billing", "delivery", "technical_failure"]),
            "summary": f"Customer reported issue resolving successfully via standard SOP guide TSG-{rng.choice([201, 202, 203, 204])}.",
            "resolution_action": "Standard SOP Applied",
            "customer_satisfaction": rng.choice([4, 5]),
            "is_synthetic": True
        }
        for i in range(1, 81)
    ]

    # Tickets with varying complexity and edge cases
    tickets = []
    ticket_types = [
        ("account_access", "Cannot log into dashboard - error code AUTH-403", "low", "simple"),
        ("billing", "Billed twice for annual subscription on Oct 14th", "medium", "simple"),
        ("technical_failure", "API webhook returning 500 internal server error intermittently", "high", "complex"),
        ("refund_request", "Requesting full refund 65 days after contract start", "medium", "policy_restricted"),
        ("delivery", "Replacement parts order #ORD-8819 not received after 2 weeks", "high", "moderate"),
        ("account_access", "Locked out after updating phone number", "low", "simple"),
        ("billing", "Invoice shows wrong company tax ID", "low", "simple"),
        ("technical_failure", "Database export query hangs at 99%", "high", "complex")
    ]

    for t_idx in range(1, 201):
        ttype, tdesc, urg, comp = ticket_types[(t_idx - 1) % len(ticket_types)]
        cust = customers[(t_idx - 1) % len(customers)]
        
        # Test case tags
        tag = "normal"
        if t_idx == 10:
            tag = "straightforward_refund_eligible"
        elif t_idx == 20:
            tag = "out_of_policy_refund_blocked"
        elif t_idx == 30:
            tag = "missing_customer_evidence"
        elif t_idx == 40:
            tag = "urgent_outage_escalation"

        tickets.append({
            "ticket_id": f"TICK-2026-{t_idx:04d}",
            "customer_id": cust["customer_id"],
            "customer_name": cust["name"],
            "customer_tier": cust["tier"],
            "subject": f"[{ttype.upper()}] {tdesc}",
            "description": f"Customer {cust['name']} reports: {tdesc}. Need resolution assistance.",
            "category": ttype,
            "urgency": urg,
            "apparent_complexity": comp,
            "created_at": (datetime(2026, 9, 1) + timedelta(hours=t_idx * 6)).isoformat(),
            "test_case_tag": tag,
            "is_synthetic": True
        })

    return {
        "customers": customers,
        "policies": policies,
        "troubleshooting_guides": troubleshooting_guides,
        "historical_resolutions": historical_resolutions,
        "tickets": tickets
    }


# ==============================================================================
# 3. Procurement and Supplier Selection Datasets
# ==============================================================================
def generate_procurement_data(seed=FIXED_SEED):
    rng = random.Random(seed)

    suppliers = [
        {
            "supplier_id": "PRO-SUP-01",
            "name": "Titanium Engineering Solutions",
            "capability": "High-precision aerospace machining & structural alloy parts",
            "capacity_per_month": 50000,
            "iso_certified": True,
            "financial_stability_score": 92,
            "is_synthetic": True
        },
        {
            "supplier_id": "PRO-SUP-02",
            "name": "Kestrel Industrial Fasteners",
            "capability": "Standardized industrial fasteners, bolts, and bulk structural hardware",
            "capacity_per_month": 250000,
            "iso_certified": True,
            "financial_stability_score": 85,
            "is_synthetic": True
        },
        {
            "supplier_id": "PRO-SUP-03",
            "name": "Atlas Smart Robotics Corp",
            "capability": "Automated warehouse robotics and optical inspection sensors",
            "capacity_per_month": 1200,
            "iso_certified": True,
            "financial_stability_score": 78,
            "is_synthetic": True
        },
        {
            "supplier_id": "PRO-SUP-04",
            "name": "Pacific Precision Castings",
            "capability": "Cast iron and aluminum engine casings",
            "capacity_per_month": 15000,
            "iso_certified": False,
            "financial_stability_score": 64,
            "is_synthetic": True
        },
        {
            "supplier_id": "PRO-SUP-05",
            "name": "Nordic Carbon Composites",
            "capability": "Lightweight carbon fiber panels and automotive fairings",
            "capacity_per_month": 8000,
            "iso_certified": True,
            "financial_stability_score": 90,
            "is_synthetic": True
        }
    ]

    risk_records = [
        {"supplier_id": "PRO-SUP-01", "risk_level": "low", "delivery_failure_rate": 0.01, "quality_defect_rate": 0.005, "audit_status": "passed", "is_synthetic": True},
        {"supplier_id": "PRO-SUP-02", "risk_level": "low", "delivery_failure_rate": 0.02, "quality_defect_rate": 0.010, "audit_status": "passed", "is_synthetic": True},
        {"supplier_id": "PRO-SUP-03", "risk_level": "medium", "delivery_failure_rate": 0.05, "quality_defect_rate": 0.025, "audit_status": "conditional", "is_synthetic": True},
        {"supplier_id": "PRO-SUP-04", "risk_level": "high", "delivery_failure_rate": 0.12, "quality_defect_rate": 0.048, "audit_status": "warning_flagged", "is_synthetic": True},
        {"supplier_id": "PRO-SUP-05", "risk_level": "low", "delivery_failure_rate": 0.015, "quality_defect_rate": 0.008, "audit_status": "passed", "is_synthetic": True}
    ]

    contracts = [
        {
            "contract_id": f"CONT-2026-{s['supplier_id']}",
            "supplier_id": s["supplier_id"],
            "payment_terms": "Net 45 days after verified receipt",
            "warranty_months": 24,
            "penalty_for_delay": "1.5% per week of delay up to 15% maximum",
            "cancellation_terms": "30 days written notice with no penalty for quality breach",
            "is_synthetic": True
        }
        for s in suppliers
    ]

    evaluation_criteria = {
        "weights": {
            "cost_score": 0.35,
            "delivery_score": 0.25,
            "risk_score": 0.20,
            "contract_terms_score": 0.10,
            "quality_capability_score": 0.10
        },
        "minimum_passing_score": 70.0,
        "is_synthetic": True
    }

    # Procurement Requests and multiple Supplier Quotations per request
    procurement_requests = []
    quotations = []
    delivery_schedules = []

    req_templates = [
        ("Precision CNC Machined Bearing Brackets", 500, "2026-11-15"),
        ("Structural Grade Titanium Hex Bolts M12", 10000, "2026-10-30"),
        ("Automated Vision Sensor Sub-assemblies", 80, "2026-12-01"),
        ("Carbon Fiber Structural Fairing Shells", 250, "2026-11-20")
    ]

    for r_idx, (item_desc, req_qty, req_deadline) in enumerate(req_templates, 1):
        req_id = f"RFQ-2026-{r_idx:03d}"
        procurement_requests.append({
            "request_id": req_id,
            "title": f"Procurement RFQ for {item_desc}",
            "item_description": item_desc,
            "required_quantity": req_qty,
            "target_delivery_date": req_deadline,
            "budget_cap": 85000.0,
            "currency": "USD",
            "is_synthetic": True
        })

        # 3 competing quotations per procurement request
        for q_offset, sup in enumerate(suppliers[:3]):
            q_id = f"QUO-{req_id}-{sup['supplier_id']}"
            base_unit_price = 45.0 + (q_offset * 6.5)
            # Variations
            unit_price = round(base_unit_price * (0.95 + (q_offset * 0.05)), 2)
            item_total = round(unit_price * req_qty, 2)
            shipping_cost = round(500.0 + (q_offset * 250.0), 2)
            tax_amount = round(item_total * 0.07, 2)
            total_evaluated_cost = round(item_total + shipping_cost + tax_amount, 2)
            
            lead_time_days = 20 + (q_offset * 10)
            promised_date = (datetime(2026, 9, 1) + timedelta(days=lead_time_days)).strftime("%Y-%m-%d")

            quotations.append({
                "quotation_id": q_id,
                "request_id": req_id,
                "supplier_id": sup["supplier_id"],
                "supplier_name": sup["name"],
                "unit_price": unit_price,
                "quantity_quoted": req_qty,
                "item_total": item_total,
                "shipping_cost": shipping_cost,
                "tax_amount": tax_amount,
                "total_cost": total_evaluated_cost,
                "currency": "USD",
                "validity_days": 60,
                "is_synthetic": True
            })

            delivery_schedules.append({
                "quotation_id": q_id,
                "supplier_id": sup["supplier_id"],
                "lead_time_days": lead_time_days,
                "promised_delivery_date": promised_date,
                "required_delivery_date": req_deadline,
                "meets_deadline": promised_date <= req_deadline,
                "is_synthetic": True
            })

    return {
        "supplier_profiles": suppliers,
        "risk_records": risk_records,
        "contracts": contracts,
        "evaluation_criteria": evaluation_criteria,
        "procurement_requests": procurement_requests,
        "quotations": quotations,
        "delivery_schedules": delivery_schedules
    }


# ==============================================================================
# 4. Sales Lead Qualification Datasets
# ==============================================================================
def generate_sales_lead_data(seed=FIXED_SEED):
    rng = random.Random(seed)

    company_profiles = [
        {"company_id": "COMP-101", "name": "Fintech Horizon Global", "industry": "Financial Services", "employee_count": 4500, "annual_revenue_usd": 650000000, "region": "North America", "is_synthetic": True},
        {"company_id": "COMP-102", "name": "Apex Biotech Labs", "industry": "Healthcare & Life Sciences", "employee_count": 850, "annual_revenue_usd": 120000000, "region": "North America", "is_synthetic": True},
        {"company_id": "COMP-103", "name": "Corner Bakery Express", "industry": "Retail & Food Services", "employee_count": 6, "annual_revenue_usd": 350000, "region": "North America", "is_synthetic": True},
        {"company_id": "COMP-104", "name": "CloudScale DevOps Inc", "industry": "Software & Technology", "employee_count": 320, "annual_revenue_usd": 48000000, "region": "Europe", "is_synthetic": True},
        {"company_id": "COMP-105", "name": "Acme Local Bodega", "industry": "Retail", "employee_count": 2, "annual_revenue_usd": 120000, "region": "North America", "is_synthetic": True},
        {"company_id": "COMP-106", "name": "Omni Logistics International", "industry": "Supply Chain & Logistics", "employee_count": 12000, "annual_revenue_usd": 2100000000, "region": "Global", "is_synthetic": True}
    ]

    products = [
        {"product_id": "PROD-ENT-SUITE", "name": "Enterprise Autonomous Agent Suite", "tier": "Enterprise", "min_budget": 50000.0, "target_employees_min": 200, "is_synthetic": True},
        {"product_id": "PROD-GROWTH-BOT", "name": "Growth Automation Cloud", "tier": "Mid-Market", "min_budget": 15000.0, "target_employees_min": 50, "is_synthetic": True}
    ]

    ideal_customer_profiles = [
        {
            "icp_id": "ICP-TIER-1-ENTERPRISE",
            "target_industries": ["Financial Services", "Healthcare & Life Sciences", "Software & Technology", "Supply Chain & Logistics"],
            "min_employees": 200,
            "max_employees": 100000,
            "min_annual_revenue_usd": 20000000,
            "target_roles": ["CTO", "VP of Engineering", "Head of AI", "Chief Digital Officer", "Director of IT"],
            "disqualified_industries": ["Retail & Food Services", "Personal Services", "Independent Bodegas"],
            "is_synthetic": True
        }
    ]

    qualification_rules = {
        "scoring_weights": {
            "industry_fit": 25,
            "company_size_fit": 25,
            "role_authority": 20,
            "intent_signals": 20,
            "budget_timeline": 10
        },
        "thresholds": {
            "sales_qualified_lead": 75,
            "marketing_nurture": 45,
            "disqualified": 44
        },
        "is_synthetic": True
    }

    leads = [
        {
            "lead_id": "LEAD-2026-0001",
            "contact_name": "Sarah Jenkins",
            "contact_email": "sjenkins@fintechhorizon.com",
            "job_title": "VP of Engineering",
            "company_id": "COMP-101",
            "stated_budget": 85000.0,
            "purchase_timeline": "Within 3 months",
            "test_case_tag": "high_fit_high_intent",
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0002",
            "contact_name": "Bob Miller",
            "contact_email": "bob@cornerbakeryexpress.com",
            "job_title": "Store Owner",
            "company_id": "COMP-103",
            "stated_budget": 200.0,
            "purchase_timeline": "Unknown / Someday",
            "test_case_tag": "clearly_unsuitable_early_disqualification",
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0003",
            "contact_name": "Marcus Vance",
            "contact_email": "mvance@cloudscaledev.com",
            "job_title": "Head of AI Architecture",
            "company_id": "COMP-104",
            "stated_budget": 60000.0,
            "purchase_timeline": "Within 6 months",
            "test_case_tag": "moderate_fit_high_intent",
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0004",
            "contact_name": "Elena Rostova",
            "contact_email": "erostova@apexbiotech.com",
            "job_title": "Director of IT",
            "company_id": "COMP-102",
            "stated_budget": 120000.0,
            "purchase_timeline": "Immediate (Q4)",
            "test_case_tag": "enterprise_fast_track",
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0005",
            "contact_name": "Tom Bodega",
            "contact_email": "tom@acmebodega.com",
            "job_title": "Cashier",
            "company_id": "COMP-105",
            "stated_budget": 50.0,
            "purchase_timeline": "None",
            "test_case_tag": "unsuitable_wrong_role_wrong_size",
            "is_synthetic": True
        }
    ]

    engagement_histories = [
        {
            "lead_id": "LEAD-2026-0001",
            "events": [
                {"type": "demo_request", "date": "2026-09-10", "notes": "Requested live demo for 50-person engineering org"},
                {"type": "whitepaper_download", "date": "2026-09-08", "notes": "Downloaded 'Autonomous AI Agent Architecture'"},
                {"type": "pricing_page_visit", "date": "2026-09-12", "notes": "Viewed Enterprise Tier pricing 4 times"}
            ],
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0002",
            "events": [
                {"type": "general_newsletter_signup", "date": "2026-09-01", "notes": "Entered email on footer newsletter"}
            ],
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0003",
            "events": [
                {"type": "demo_request", "date": "2026-09-14", "notes": "Asked about API rate limits and CrewAI compatibility"}
            ],
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0004",
            "events": [
                {"type": "executive_briefing_request", "date": "2026-09-15", "notes": "Requested direct call with product VP"},
                {"type": "security_whitepaper_download", "date": "2026-09-11", "notes": "Downloaded SOC2 & HIPAA compliance pack"}
            ],
            "is_synthetic": True
        },
        {
            "lead_id": "LEAD-2026-0005",
            "events": [],
            "is_synthetic": True
        }
    ]

    return {
        "company_profiles": company_profiles,
        "products": products,
        "ideal_customer_profiles": ideal_customer_profiles,
        "qualification_rules": qualification_rules,
        "leads": leads,
        "engagement_histories": engagement_histories
    }


def generate_all():
    print(f"Generating synthetic datasets using fixed seed={FIXED_SEED}...")
    ensure_dirs()

    # 1. Invoice Processing
    inv_data = generate_invoice_processing_data(FIXED_SEED)
    inv_dir = BASE_DATA_DIR / "invoice_processing"
    with open(inv_dir / "suppliers.json", "w", encoding="utf-8") as f:
        json.dump(inv_data["suppliers"], f, indent=2)
    with open(inv_dir / "policies.json", "w", encoding="utf-8") as f:
        json.dump(inv_data["policies"], f, indent=2)
    with open(inv_dir / "purchase_orders.json", "w", encoding="utf-8") as f:
        json.dump(inv_data["purchase_orders"], f, indent=2)
    with open(inv_dir / "historical_invoices.json", "w", encoding="utf-8") as f:
        json.dump(inv_data["historical_invoices"], f, indent=2)
    with open(inv_dir / "invoices.json", "w", encoding="utf-8") as f:
        json.dump(inv_data["invoices"], f, indent=2)
    print(f" [OK] Generated Invoice Processing datasets ({len(inv_data['invoices'])} invoices, {len(inv_data['purchase_orders'])} POs, {len(inv_data['suppliers'])} suppliers)")

    # 2. Customer Support
    cs_data = generate_customer_support_data(FIXED_SEED)
    cs_dir = BASE_DATA_DIR / "customer_support"
    with open(cs_dir / "customers.json", "w", encoding="utf-8") as f:
        json.dump(cs_data["customers"], f, indent=2)
    with open(cs_dir / "policies.json", "w", encoding="utf-8") as f:
        json.dump(cs_data["policies"], f, indent=2)
    with open(cs_dir / "troubleshooting_guides.json", "w", encoding="utf-8") as f:
        json.dump(cs_data["troubleshooting_guides"], f, indent=2)
    with open(cs_dir / "historical_resolutions.json", "w", encoding="utf-8") as f:
        json.dump(cs_data["historical_resolutions"], f, indent=2)
    with open(cs_dir / "tickets.json", "w", encoding="utf-8") as f:
        json.dump(cs_data["tickets"], f, indent=2)
    print(f" [OK] Generated Customer Support datasets ({len(cs_data['tickets'])} tickets, {len(cs_data['customers'])} customers, {len(cs_data['policies'])} policies)")

    # 3. Procurement
    pro_data = generate_procurement_data(FIXED_SEED)
    pro_dir = BASE_DATA_DIR / "procurement_supplier_selection"
    with open(pro_dir / "supplier_profiles.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["supplier_profiles"], f, indent=2)
    with open(pro_dir / "risk_records.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["risk_records"], f, indent=2)
    with open(pro_dir / "contracts.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["contracts"], f, indent=2)
    with open(pro_dir / "evaluation_criteria.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["evaluation_criteria"], f, indent=2)
    with open(pro_dir / "procurement_requests.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["procurement_requests"], f, indent=2)
    with open(pro_dir / "quotations.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["quotations"], f, indent=2)
    with open(pro_dir / "delivery_schedules.json", "w", encoding="utf-8") as f:
        json.dump(pro_data["delivery_schedules"], f, indent=2)
    print(f" [OK] Generated Procurement datasets ({len(pro_data['procurement_requests'])} requests, {len(pro_data['quotations'])} quotations, {len(pro_data['supplier_profiles'])} suppliers)")

    # 4. Sales Lead
    lead_data = generate_sales_lead_data(FIXED_SEED)
    lead_dir = BASE_DATA_DIR / "sales_lead_qualification"
    with open(lead_dir / "company_profiles.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["company_profiles"], f, indent=2)
    with open(lead_dir / "products.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["products"], f, indent=2)
    with open(lead_dir / "ideal_customer_profiles.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["ideal_customer_profiles"], f, indent=2)
    with open(lead_dir / "qualification_rules.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["qualification_rules"], f, indent=2)
    with open(lead_dir / "leads.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["leads"], f, indent=2)
    with open(lead_dir / "engagement_histories.json", "w", encoding="utf-8") as f:
        json.dump(lead_data["engagement_histories"], f, indent=2)
    print(f" [OK] Generated Sales Lead datasets ({len(lead_data['leads'])} leads, {len(lead_data['company_profiles'])} companies)")

    print("All synthetic datasets generated successfully without any external LLM dependencies.")

if __name__ == "__main__":
    generate_all()
