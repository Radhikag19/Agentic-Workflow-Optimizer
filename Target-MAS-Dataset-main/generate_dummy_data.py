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




"""
Synthetic Dummy Data Generator for Agentic Workflow Optimizer (AWO).

Generates fixed-seed synthetic datasets for all three workflows:
1. Recruitment Screening: 250 fictional resumes, 15 job descriptions, candidate records, evidence IDs.
2. Inventory Replenishment: 120 SKUs, historical daily sales, suppliers, policies, purchase orders.
3. Contract Review: 120 contracts across multiple contract types, 8 compliance policies, structured clause library.

STRICT CONSTRAINT: Deterministic local generation with random seed 42. No external APIs or LLM keys used.
"""

import json
import os
import random
from datetime import date, timedelta

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# =====================================================================
# Workflow 1: Recruitment Screening Data Generation
# =====================================================================

JOB_TITLES = [
    ("JOB-01", "Senior Cloud Backend Engineer", "Engineering", "Remote"),
    ("JOB-02", "Senior Data Engineer", "Data", "Hybrid - New York"),
    ("JOB-03", "Frontend React Architect", "Engineering", "Remote"),
    ("JOB-04", "DevOps & Platform Engineer", "Operations", "Hybrid - Austin"),
    ("JOB-05", "Machine Learning Specialist", "AI Research", "San Francisco"),
    ("JOB-06", "Product Security Analyst", "Security", "Remote"),
    ("JOB-07", "Full Stack Python/TypeScript Developer", "Engineering", "Chicago"),
    ("JOB-08", "Engineering Manager - Infrastructure", "Management", "Seattle"),
    ("JOB-09", "QA Automation Lead", "Quality", "Remote"),
    ("JOB-10", "Site Reliability Engineer (SRE)", "Infrastructure", "Remote"),
    ("JOB-11", "Enterprise Solutions Architect", "Sales Engineering", "New York"),
    ("JOB-12", "Mobile iOS Engineer", "Mobile", "Remote"),
    ("JOB-13", "Distributed Systems Researcher", "Research", "Boston"),
    ("JOB-14", "Technical Project Manager", "PMO", "Atlanta"),
    ("JOB-15", "Database Administrator (PostgreSQL)", "Operations", "Remote"),
]

MANDATORY_CRITERIA_POOL = {
    "JOB-01": [
        ("CRIT-M1", "technical_skill", "Minimum 5 years of Python or Go backend systems experience"),
        ("CRIT-M2", "technical_skill", "Hands-on expertise with AWS or GCP cloud architecture and containerization (Docker, Kubernetes)"),
        ("CRIT-M3", "education", "Bachelor's degree in Computer Science, Software Engineering, or equivalent practical experience"),
    ],
    "JOB-02": [
        ("CRIT-M1", "technical_skill", "Minimum 4 years building scalable data pipelines using Spark, Airflow, or Kafka"),
        ("CRIT-M2", "technical_skill", "Strong proficiency in SQL, data modeling, and distributed query engines"),
    ],
}

PREFERRED_CRITERIA_POOL = {
    "JOB-01": [
        ("CRIT-P1", "technical_skill", "Experience with gRPC and high-throughput event streaming"),
        ("CRIT-P2", "certification", "AWS Certified Solutions Architect or equivalent professional certificate"),
    ],
    "JOB-02": [
        ("CRIT-P1", "technical_skill", "Experience with Snowflake or Databricks lakehouse architectures"),
    ],
}

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Avery", "Dakota", "Jamie", "Logan", "Reese", "Quinn"]
LAST_NAMES = ["Smith", "Chen", "Patel", "Johnson", "Kim", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Jackson", "White"]
TECH_SKILLS_POOL = ["Python", "Go", "Docker", "Kubernetes", "AWS", "GCP", "PostgreSQL", "Redis", "Kafka", "React", "TypeScript", "Spark", "Airflow", "Terraform", "FastAPI", "GraphQL"]


def generate_recruitment_data():
    target_dir = os.path.join(DATA_DIR, "recruitment_screening")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Generate 15 Job Descriptions
    jobs = []
    for job_id, title, dept, loc in JOB_TITLES:
        mand = MANDATORY_CRITERIA_POOL.get(job_id, [
            ("CRIT-M1", "technical_skill", f"Minimum 4 years professional software engineering experience in {dept}"),
            ("CRIT-M2", "technical_skill", "Experience with modern CI/CD, testing, and production deployment"),
        ])
        pref = PREFERRED_CRITERIA_POOL.get(job_id, [
            ("CRIT-P1", "technical_skill", "Contribution to open-source software or engineering leadership"),
            ("CRIT-P2", "certification", "Industry recognized technical certification"),
        ])

        job_doc = {
            "job_id": job_id,
            "title": title,
            "department": dept,
            "location": loc,
            "summary": f"We are seeking an experienced {title} to build resilient, mission-critical systems.",
            "mandatory_criteria": [
                {"criterion_id": f"{job_id}-{cid}", "category": cat, "description": desc, "requirement_type": "mandatory"}
                for cid, cat, desc in mand
            ],
            "preferred_criteria": [
                {"criterion_id": f"{job_id}-{cid}", "category": cat, "description": desc, "requirement_type": "preferred"}
                for cid, cat, desc in pref
            ],
        }
        jobs.append(job_doc)

    with open(os.path.join(target_dir, "jobs.json"), "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    # 2. Generate 250 Resumes with Deliberate Edge Cases
    resumes = []
    for i in range(1, 251):
        cand_id = f"CAND-{i:03d}"
        app_id = f"APP-{i:03d}"
        target_job = jobs[(i - 1) % len(jobs)]
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"

        # Assign deliberate edge cases
        edge_case = "normal"
        if i % 25 == 1:
            edge_case = "missing_skills"
        elif i % 25 == 2:
            edge_case = "inconsistent_dates"
        elif i % 25 == 3:
            edge_case = "keyword_stuffing"
        elif i % 25 == 4:
            edge_case = "incomplete_resume"
        elif i % 25 == 5:
            edge_case = "irrelevant_experience"
        elif i % 25 == 6:
            edge_case = "ambiguous_experience"

        skills = random.sample(TECH_SKILLS_POOL, k=random.randint(4, 8))
        if edge_case == "missing_skills":
            skills = ["HTML", "CSS", "WordPress"]  # lacking required backend
        elif edge_case == "keyword_stuffing":
            skills = TECH_SKILLS_POOL + ["AI", "Blockchain", "Quantum", "Hadoop", "Rust", "C++", "Cybersecurity", "ZeroTrust"]
        elif edge_case == "irrelevant_experience":
            skills = ["Culinary Arts", "Menu Design", "Kitchen Inventory", "Food Safety"]

        # Date consistency
        start_year = random.randint(2015, 2019)
        end_year = 2024
        if edge_case == "inconsistent_dates":
            # Start year after end year
            start_year = 2025
            end_year = 2021

        sections = {}
        evidence_ids = []

        if edge_case != "incomplete_resume":
            sections["contact_information"] = {
                "name": name,
                "email": f"{first.lower()}.{last.lower()}{i}@example.org",
                "phone": f"+1-555-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
                "location": target_job["location"],
            }

        ev_edu = f"EVD-{i:03d}-EDU"
        evidence_ids.append(ev_edu)
        sections["education"] = {
            "evidence_id": ev_edu,
            "degree": "B.S. in Computer Science" if edge_case != "irrelevant_experience" else "B.A. in Hospitality Management",
            "institution": "State University",
            "graduation_year": start_year - 1,
        }

        ev_exp = f"EVD-{i:03d}-EXP1"
        evidence_ids.append(ev_exp)
        if edge_case == "ambiguous_experience":
            desc = "Assisted with various technological systems and participated in meetings regarding software."
        elif edge_case == "irrelevant_experience":
            desc = "Managed high-volume kitchen prep and scheduled culinary shifts."
        else:
            desc = f"Designed and deployed scalable services using {', '.join(skills[:3])}. Improved query latency by 35%."

        sections["experience"] = [
            {
                "evidence_id": ev_exp,
                "title": f"Software Engineer" if edge_case != "irrelevant_experience" else "Executive Chef",
                "company": "Apex Dynamics Corp",
                "start_date": f"{start_year}-01",
                "end_date": f"{end_year}-12",
                "description": desc,
            }
        ]

        ev_sk = f"EVD-{i:03d}-SKL"
        evidence_ids.append(ev_sk)
        sections["skills"] = {
            "evidence_id": ev_sk,
            "technical_skills": skills,
        }

        resume_record = {
            "application_id": app_id,
            "candidate_id": cand_id,
            "candidate_name": name,
            "target_job_id": target_job["job_id"],
            "target_job_title": target_job["title"],
            "edge_case_type": edge_case,
            "sections": sections,
            "source_evidence_ids": evidence_ids,
        }
        resumes.append(resume_record)

    with open(os.path.join(target_dir, "resumes.json"), "w", encoding="utf-8") as f:
        json.dump(resumes, f, indent=2)

    print(f"[DataGen] Created {len(jobs)} jobs and {len(resumes)} resumes in {target_dir}")


# =====================================================================
# Workflow 2: Inventory Replenishment Data Generation
# =====================================================================

def generate_inventory_data():
    target_dir = os.path.join(DATA_DIR, "inventory_replenishment")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Generate 20 Suppliers
    suppliers = []
    for s_idx in range(1, 21):
        sup_id = f"SUP-{s_idx:02d}"
        moq = random.choice([50, 100, 200, 500])
        pack_size = random.choice([5, 10, 25, 50])
        lead_time = random.choice([3, 7, 14, 21, 30])
        reliability = round(random.uniform(0.75, 0.99), 2)
        suppliers.append({
            "supplier_id": sup_id,
            "supplier_name": f"Vendor Supply Partner {s_idx}",
            "lead_time_days": lead_time,
            "minimum_order_quantity": moq,
            "pack_size": pack_size,
            "reliability_score": reliability,
            "contact_email": f"orders@vendor{s_idx}.example.com",
        })

    with open(os.path.join(target_dir, "suppliers.json"), "w", encoding="utf-8") as f:
        json.dump(suppliers, f, indent=2)

    # 2. Generate 120 SKUs and Inventory Snapshots
    categories = ["Electronics", "Office Supplies", "Industrial Parts", "Packaging", "Raw Materials"]
    skus = []
    sales_history = {}
    purchase_orders = []

    today = date(2026, 10, 15)

    for i in range(1, 121):
        sku_id = f"SKU-{i:04d}"
        sup = suppliers[(i - 1) % len(suppliers)]
        cat = random.choice(categories)

        # Edge cases
        edge_case = "normal"
        if i % 15 == 1:
            edge_case = "stockout"
        elif i % 15 == 2:
            edge_case = "excess_inventory"
        elif i % 15 == 3:
            edge_case = "demand_spike"
        elif i % 15 == 4:
            edge_case = "delayed_supplier"
        elif i % 15 == 5:
            edge_case = "intermittent_demand"
        elif i % 15 == 6:
            edge_case = "insufficient_history"

        # Generate base demand
        base_daily_demand = random.randint(5, 40)
        reorder_threshold = int(base_daily_demand * sup["lead_time_days"] * 1.5)
        safety_stock = int(base_daily_demand * 5)

        if edge_case == "stockout":
            current_stock = 0
            reserved_stock = 0
            incoming_stock = 0
        elif edge_case == "excess_inventory":
            current_stock = reorder_threshold * 4
            reserved_stock = 20
            incoming_stock = 0
        else:
            current_stock = random.randint(int(reorder_threshold * 0.4), int(reorder_threshold * 1.8))
            reserved_stock = min(current_stock, random.randint(0, int(current_stock * 0.3)))
            incoming_stock = random.choice([0, sup["pack_size"] * 2, sup["pack_size"] * 5])

        # Generate 90 days of daily sales
        num_days = 10 if edge_case == "insufficient_history" else 90
        daily_sales = []
        for d in range(num_days):
            day_date = (today - timedelta(days=num_days - d)).isoformat()
            if edge_case == "intermittent_demand":
                qty = base_daily_demand * 3 if random.random() < 0.2 else 0
            elif edge_case == "demand_spike" and d >= num_days - 5:
                qty = base_daily_demand * 4
            else:
                qty = max(0, int(random.gauss(base_daily_demand, base_daily_demand * 0.25)))
            daily_sales.append({"date": day_date, "units_sold": qty})

        sales_history[sku_id] = {
            "sku_id": sku_id,
            "recorded_days": num_days,
            "daily_sales": daily_sales,
        }

        # Open Purchase Order if incoming_stock > 0
        if incoming_stock > 0:
            po_id = f"PO-2026-{i:03d}"
            purchase_orders.append({
                "po_id": po_id,
                "sku_id": sku_id,
                "supplier_id": sup["supplier_id"],
                "quantity": incoming_stock,
                "order_date": (today - timedelta(days=4)).isoformat(),
                "expected_delivery_date": (today + timedelta(days=sup["lead_time_days"] - 4)).isoformat(),
                "status": "in_transit" if edge_case != "delayed_supplier" else "delayed",
            })

        sku_doc = {
            "sku_id": sku_id,
            "sku_name": f"{cat} Component Model {i:03d}",
            "category": cat,
            "warehouse_id": "WH-CENTRAL-01" if i % 2 == 0 else "WH-EAST-02",
            "supplier_id": sup["supplier_id"],
            "current_stock": current_stock,
            "reserved_stock": reserved_stock,
            "incoming_stock": incoming_stock,
            "unit_cost": round(random.uniform(10.0, 150.0), 2),
            "reorder_threshold": reorder_threshold,
            "safety_stock": safety_stock,
            "edge_case_type": edge_case,
            "source_record_ids": [f"INV-{sku_id}", f"SUP-{sup['supplier_id']}"],
        }
        skus.append(sku_doc)

    with open(os.path.join(target_dir, "inventory_records.json"), "w", encoding="utf-8") as f:
        json.dump(skus, f, indent=2)

    with open(os.path.join(target_dir, "sales_history.json"), "w", encoding="utf-8") as f:
        json.dump(sales_history, f, indent=2)

    with open(os.path.join(target_dir, "purchase_orders.json"), "w", encoding="utf-8") as f:
        json.dump(purchase_orders, f, indent=2)

    # General Replenishment Policy
    policies = {
        "planning_rules": {
            "default_safety_stock_days": 5,
            "lead_time_demand_formula": "daily_avg_demand * lead_time_days",
            "base_replenishment_formula": "max(0, lead_time_demand + safety_stock - (current_stock - reserved_stock + incoming_stock))",
            "rounding_rule": "round_up_to_nearest_pack_size",
            "moq_rule": "max(rounded_quantity, supplier_moq) if base_replenishment > 0 else 0",
        }
    }
    with open(os.path.join(target_dir, "policies.json"), "w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2)

    print(f"[DataGen] Created {len(skus)} SKUs, {len(suppliers)} suppliers, {len(purchase_orders)} POs in {target_dir}")


# =====================================================================
# Workflow 3: Contract Review Data Generation
# =====================================================================

CONTRACT_TYPES = [
    "SaaS Subscription Agreement",
    "Master Services Agreement (MSA)",
    "Non-Disclosure Agreement (NDA)",
    "Vendor Procurement Agreement",
    "Software License Agreement",
]

POLICIES = [
    {
        "policy_id": "POL-CORP-SEC-01",
        "policy_name": "Corporate Data Protection & Security Standard",
        "version": "2.4",
        "requirements": [
            {
                "requirement_id": "REQ-SEC-01",
                "clause_type": "data_protection",
                "is_mandatory": True,
                "text": "Vendor must notify customer of any security breach within 48 hours of discovery.",
                "threshold": "48 hours notification",
            },
            {
                "requirement_id": "REQ-SEC-02",
                "clause_type": "data_protection",
                "is_mandatory": True,
                "text": "Data must be encrypted in transit using TLS 1.3 and at rest using AES-256.",
                "threshold": "TLS 1.3 / AES-256",
            },
        ],
    },
    {
        "policy_id": "POL-CORP-FIN-02",
        "policy_name": "Commercial Payment Terms & Billing Policy",
        "version": "3.1",
        "requirements": [
            {
                "requirement_id": "REQ-FIN-01",
                "clause_type": "payment_terms",
                "is_mandatory": True,
                "text": "Payment terms shall not exceed Net 45 days from receipt of undisputed invoice.",
                "threshold": "Net 45 maximum",
            },
            {
                "requirement_id": "REQ-FIN-02",
                "clause_type": "payment_terms",
                "is_mandatory": False,
                "text": "Late payment penalties should not exceed 1.5% per month.",
                "threshold": "1.5% monthly cap",
            },
        ],
    },
    {
        "policy_id": "POL-CORP-LGL-03",
        "policy_name": "Legal Liability & Indemnification Standards",
        "version": "1.8",
        "requirements": [
            {
                "requirement_id": "REQ-LGL-01",
                "clause_type": "liability_cap",
                "is_mandatory": True,
                "text": "Aggregate liability cap must not exceed 12 months of total fees paid or payable.",
                "threshold": "12 months fees cap",
            },
            {
                "requirement_id": "REQ-LGL-02",
                "clause_type": "termination",
                "is_mandatory": True,
                "text": "Customer must have the right to terminate for convenience with no more than 30 days prior written notice.",
                "threshold": "30 days convenience termination",
            },
        ],
    },
]


def generate_contract_data():
    target_dir = os.path.join(DATA_DIR, "contract_review")
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, "policies.json"), "w", encoding="utf-8") as f:
        json.dump(POLICIES, f, indent=2)

    contracts = []
    for i in range(1, 121):
        contract_id = f"CTR-2026-{i:04d}"
        c_type = random.choice(CONTRACT_TYPES)
        vendor_name = f"Enterprise Partner {i:03d} LLC"

        # Edge cases
        edge_case = "compliant"
        if i % 15 == 1:
            edge_case = "payment_term_violation"  # Net 90 instead of Net 45
        elif i % 15 == 2:
            edge_case = "missing_data_breach_clause"
        elif i % 15 == 3:
            edge_case = "cross_clause_contradiction"  # Term vs Renewal
        elif i % 15 == 4:
            edge_case = "ambiguous_liability"
        elif i % 15 == 5:
            edge_case = "inconsistent_dates"
        elif i % 15 == 6:
            edge_case = "unlimited_liability"

        # Clause Generation
        clauses = []

        # 1. Payment Clause
        cls_pay_id = f"CLS-{i:04d}-PAY"
        if edge_case == "payment_term_violation":
            pay_text = "Invoices shall be payable within ninety (90) days of receipt by Customer."
        else:
            pay_text = "Customer shall pay all undisputed invoice amounts within thirty (30) days of invoice date."
        clauses.append({
            "clause_id": cls_pay_id,
            "clause_type": "payment_terms",
            "clause_title": "Section 3: Invoicing and Payment Terms",
            "clause_text": pay_text,
            "section_reference": "Section 3.1",
        })

        # 2. Data Protection Clause
        if edge_case != "missing_data_breach_clause":
            cls_data_id = f"CLS-{i:04d}-DATA"
            if edge_case == "ambiguous_liability":
                data_text = "Vendor will endeavor to report security incidents in a reasonable and commercially practical timeframe."
            else:
                data_text = "Vendor shall notify Customer in writing within twenty-four (24) hours of becoming aware of any confirmed Data Breach."
            clauses.append({
                "clause_id": cls_data_id,
                "clause_type": "data_protection",
                "clause_title": "Section 8: Security & Breach Notification",
                "clause_text": data_text,
                "section_reference": "Section 8.2",
            })

        # 3. Liability Clause
        cls_liab_id = f"CLS-{i:04d}-LIAB"
        if edge_case == "unlimited_liability":
            liab_text = "Neither party shall limit its liability under this Agreement for any claims whatsoever."
        elif edge_case == "ambiguous_liability":
            liab_text = "Liability shall be determined in accordance with applicable general trade customs."
        else:
            liab_text = "Each party's maximum aggregate liability shall be strictly limited to the total fees paid under this Agreement in the preceding twelve (12) months."
        clauses.append({
            "clause_id": cls_liab_id,
            "clause_type": "liability_cap",
            "clause_title": "Section 10: Limitation of Liability",
            "clause_text": liab_text,
            "section_reference": "Section 10.1",
        })

        # 4. Termination & Renewal Clauses
        cls_term_id = f"CLS-{i:04d}-TERM"
        cls_ren_id = f"CLS-{i:04d}-REN"
        term_text = "Either party may terminate this Agreement for convenience upon providing thirty (30) days prior written notice."
        if edge_case == "cross_clause_contradiction":
            ren_text = "This Agreement shall automatically renew for successive two-year periods and CANNOT be terminated for convenience prior to full term expiration."
        else:
            ren_text = "This Agreement renews on an annual basis unless either party gives notice of non-renewal at least 30 days prior."

        clauses.append({
            "clause_id": cls_term_id,
            "clause_type": "termination",
            "clause_title": "Section 12: Term and Termination",
            "clause_text": term_text,
            "section_reference": "Section 12.1",
        })
        clauses.append({
            "clause_id": cls_ren_id,
            "clause_type": "renewal",
            "clause_title": "Section 12: Renewal Provisions",
            "clause_text": ren_text,
            "section_reference": "Section 12.4",
        })

        # Inconsistent dates edge case
        effective_date = "2026-01-01"
        expiration_date = "2027-01-01"
        if edge_case == "inconsistent_dates":
            effective_date = "2026-10-01"
            expiration_date = "2025-05-01"  # expiration before effective date

        contract_doc = {
            "contract_id": contract_id,
            "contract_title": f"{c_type} with {vendor_name}",
            "contract_type": c_type,
            "contract_version": "1.0",
            "party_a": "AWO Enterprise Corp",
            "party_b": vendor_name,
            "effective_date": effective_date,
            "expiration_date": expiration_date,
            "edge_case_type": edge_case,
            "clauses": clauses,
            "policy_id": "POL-CORP-SEC-01",
            "policy_version": "2.4",
        }
        contracts.append(contract_doc)

    with open(os.path.join(target_dir, "contracts.json"), "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2)

    print(f"[DataGen] Created {len(contracts)} contracts and {len(POLICIES)} policies in {target_dir}")


def main():
    print("=" * 60)
    print("Generating Synthetic Datasets for AWO Multi-Agent Workflows")
    print(f"Seed: {RANDOM_SEED} (Deterministic, Zero external API calls)")
    print("=" * 60)
    generate_recruitment_data()
    generate_inventory_data()
    generate_contract_data()
    print("=" * 60)
    print("Synthetic dataset generation complete!")


if __name__ == "__main__":
    main()




"""Synthetic Dataset Generator for Agentic Workflow Optimizer (AWO)

Generates deeply detailed, realistic, domain-authentic synthetic datasets for:
1. IT Incident Triage (data/incident_triage/)
2. Employee Expense Reimbursement (data/expense_reimbursement/)
3. Market and Competitor Research (data/market_research/)

Uses a fixed random seed (42) for deterministic, reproducible generation.
All records are marked with `"synthetic": true`.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Fix random seed for strict reproducibility
random.seed(42)

BASE_DATA_DIR = Path(__file__).resolve().parent / "data"


def generate_incident_triage_data():
    out_dir = BASE_DATA_DIR / "incident_triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    services = [
        {"service_id": "auth-gateway", "tier": "Tier-1", "sla_pct": 99.99, "owner": "sec-ops", "dependencies": ["redis-session", "user-db"]},
        {"service_id": "payment-processor", "tier": "Tier-1", "sla_pct": 99.99, "owner": "fintech-core", "dependencies": ["postgres-primary", "stripe-gateway"]},
        {"service_id": "order-service", "tier": "Tier-1", "sla_pct": 99.95, "owner": "checkout-team", "dependencies": ["postgres-primary", "kafka-cluster"]},
        {"service_id": "inventory-api", "tier": "Tier-2", "sla_pct": 99.9, "owner": "logistics-eng", "dependencies": ["postgres-replica", "redis-cache"]},
        {"service_id": "notification-hub", "tier": "Tier-3", "sla_pct": 99.5, "owner": "growth-eng", "dependencies": ["kafka-cluster", "sendgrid-relay"]},
        {"service_id": "database-primary", "tier": "Tier-1", "sla_pct": 99.99, "owner": "dba-team", "dependencies": ["ebs-storage"]},
        {"service_id": "redis-cluster", "tier": "Tier-2", "sla_pct": 99.9, "owner": "infra-ops", "dependencies": ["vpc-peering"]},
        {"service_id": "ingress-lb", "tier": "Tier-1", "sla_pct": 99.99, "owner": "net-infra", "dependencies": ["aws-route53"]},
        {"service_id": "search-indexer", "tier": "Tier-3", "sla_pct": 99.0, "owner": "search-team", "dependencies": ["elasticsearch-prod"]},
        {"service_id": "analytics-pipeline", "tier": "Tier-3", "sla_pct": 98.5, "owner": "data-platform", "dependencies": ["s3-datalake"]},
    ]

    with open(out_dir / "services_catalog.json", "w", encoding="utf-8") as f:
        json.dump({"services": services, "synthetic": True, "count": len(services)}, f, indent=2)

    categories = ["network_failure", "application_error", "authentication_issue", "database_problem", "storage_exhaustion"]
    symptom_pool = {
        "network_failure": ["DNS resolution timeout on route53", "504 Gateway Timeout on ingress-lb", "Packet drop rate spike > 12% across az-us-east-1a", "BGP flap detected on transit gateway"],
        "application_error": ["Unhandled NullPointerException in CheckoutController", "OOMKilled exit code 137 on worker pods", "HTTP 500 spike reaching 42% of traffic", "Thread deadlock detected in OrderFulfillmentWorker"],
        "authentication_issue": ["JWT token signature verification failure", "Expired OAuth certificate on identity-provider", "OAuth2 callback returns 401 Unauthorized", "LDAP sync daemon timeout after 30000ms"],
        "database_problem": ["PostgreSQL active connection pool reached 100% (250/250)", "Deadlock detected on table order_transactions", "Write latency spiked to 4200ms on primary-db", "Replication lag exceeded 450 seconds on replica-02"],
        "storage_exhaustion": ["Disk usage /var/log exceeded 96% on host node-prod-4", "EBS volume IOPS throttled at burst balance 0%", "S3 multipart upload rejected with QuotaExceeded", "Redis memory usage > 98% with maxmemory-policy noeviction"],
    }

    # Generate 35 Runbooks
    runbooks = []
    rb_counter = 101
    for cat, symptoms in symptom_pool.items():
        for i, sym in enumerate(symptoms):
            rb = {
                "runbook_id": f"RB-{rb_counter}",
                "title": f"Standard Mitigation for {cat.replace('_', ' ').title()} - Scenario {i+1}",
                "category": cat,
                "target_service": random.choice(services)["service_id"],
                "matching_symptoms": [sym, f"Secondary metric: Elevated latency during {sym.lower()}"],
                "prerequisites": ["Root access or AWS SSO role ProdOperations", "Kubectl context set to prod-us-east-1"],
                "diagnostic_commands": [
                    f"kubectl logs -l app={cat.split('_')[0]} -n production --tail=200",
                    "curl -s -o /dev/null -w '%{http_code}' https://status.internal/healthz",
                    "promtool query instant 'rate(http_requests_total{status=~\"5..\"}[5m])'"
                ],
                "remediation_steps": [
                    f"Step 1: Drain traffic from degraded pods or instances for {sym.split()[0]}",
                    "Step 2: Apply temporary rate-limit or failover to standby replica/pool",
                    "Step 3: Execute graceful restart via rolling deployment rollback",
                    "Step 4: Verify error rates drop below 0.1% for 5 consecutive minutes"
                ],
                "is_reversible": True,
                "safety_notes": "Do not terminate primary node without promoting read-replica first. Verify write freeze before reboot.",
                "verification_procedure": "Check Grafana dashboard 'Core Microservices SLA' to ensure 2xx ratio >= 99.95%",
                "synthetic": True,
            }
            runbooks.append(rb)
            rb_counter += 1

    # Add extra general runbooks to reach 30
    extra_titles = [
        ("RB-121", "Emergency Traffic Shedding via Cloudflare WAF", "network_failure"),
        ("RB-122", "Postgres Connection Pool Flush and Client Throttling", "database_problem"),
        ("RB-123", "Redis Cache Invalidation and Memory Compaction", "storage_exhaustion"),
        ("RB-124", "Identity Provider Failover to Backup Okta Tenant", "authentication_issue"),
        ("RB-125", "Worker Pod Horizontal Scale-Out Under Deadlock", "application_error"),
        ("RB-126", "EBS Volume Burst Balance Replenishment & Volume Expansion", "storage_exhaustion"),
        ("RB-127", "Kafka Consumer Rebalance Hang Recovery", "application_error"),
        ("RB-128", "Transit Gateway Route Table Reset", "network_failure"),
        ("RB-129", "Stale OAuth2 Token Revocation and Redis Session Flush", "authentication_issue"),
        ("RB-130", "PostgreSQL Read-Replica Promotion under Primary Failure", "database_problem"),
    ]
    for r_id, r_title, r_cat in extra_titles:
        runbooks.append({
            "runbook_id": r_id,
            "title": r_title,
            "category": r_cat,
            "target_service": "auth-gateway" if "Identity" in r_title else "payment-processor",
            "matching_symptoms": ["System-wide degradation under unexpected load", "Cascade failure across upstream services"],
            "prerequisites": ["Emergency incident commander authorization"],
            "diagnostic_commands": ["kubectl top nodes", "pg_stat_activity inspection"],
            "remediation_steps": ["Execute graceful degradation script: ./scripts/emergency-failover.sh"],
            "is_reversible": True,
            "safety_notes": "Must notify customer support before execution.",
            "verification_procedure": "Monitor response latency p99 under 200ms.",
            "synthetic": True
        })

    with open(out_dir / "runbooks.json", "w", encoding="utf-8") as f:
        json.dump({"runbooks": runbooks, "synthetic": True, "count": len(runbooks)}, f, indent=2)

    # Generate 200 Historical Incidents
    incidents = []
    base_time = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
    for idx in range(1, 201):
        cat = random.choice(categories)
        symptom = random.choice(symptom_pool[cat])
        srv = random.choice(services)
        
        # Determine realistic severity based on service tier and users affected
        users = random.choice([25, 150, 800, 3500, 15000, 45000])
        if srv["tier"] == "Tier-1" and users > 5000:
            sev = "SEV-1"
            outage_m = random.randint(30, 180)
        elif srv["tier"] in ["Tier-1", "Tier-2"] and users > 500:
            sev = "SEV-2"
            outage_m = random.randint(20, 90)
        elif users > 100:
            sev = "SEV-3"
            outage_m = random.randint(10, 45)
        else:
            sev = "SEV-4"
            outage_m = random.randint(5, 30)

        matching_rb = random.choice([rb for rb in runbooks if rb["category"] == cat])
        created_at = base_time + timedelta(hours=idx * 4, minutes=random.randint(1, 55))

        incidents.append({
            "incident_id": f"INC-{1000 + idx}",
            "title": f"{srv['service_id']}: {symptom}",
            "service": srv["service_id"],
            "service_tier": srv["tier"],
            "category": cat,
            "severity": sev,
            "impact_users": users,
            "outage_minutes": outage_m,
            "symptoms": [symptom, f"Secondary telemetry spike on {srv['service_id']}"],
            "error_signature": f"ERR_{cat.upper()}_{random.randint(100, 999)}",
            "resolved_runbook_id": matching_rb["runbook_id"],
            "root_cause": f"Underlying condition triggered by {symptom.lower()} causing service {srv['service_id']} degradation.",
            "resolution_summary": f"Executed runbook {matching_rb['runbook_id']}. System recovered in {outage_m} minutes.",
            "created_at": created_at.isoformat(),
            "synthetic": True,
        })

    with open(out_dir / "historical_incidents.json", "w", encoding="utf-8") as f:
        json.dump({"incidents": incidents, "synthetic": True, "count": len(incidents)}, f, indent=2)

    # Generate 50 Log Snippets
    logs = []
    for idx in range(1, 51):
        cat = random.choice(categories)
        srv = random.choice(services)["service_id"]
        level = random.choice(["ERROR", "CRITICAL", "WARN"])
        logs.append({
            "log_id": f"LOG-TR-{5000 + idx}",
            "timestamp": (base_time + timedelta(minutes=idx * 27)).isoformat(),
            "service": srv,
            "level": level,
            "category": cat,
            "message": f"[{srv}] {level}: {random.choice(symptom_pool[cat])} - stacktrace at com.corp.internal.{srv.replace('-', '')}.ServiceHandler.execute(line:{random.randint(45, 420)})",
            "correlation_id": f"c-uuid-{random.randint(100000, 999999)}",
            "synthetic": True,
        })

    with open(out_dir / "log_snippets.json", "w", encoding="utf-8") as f:
        json.dump({"logs": logs, "synthetic": True, "count": len(logs)}, f, indent=2)

    print(f"[DataGen] Incident Triage: generated {len(incidents)} incidents, {len(runbooks)} runbooks, {len(logs)} logs.")


def generate_expense_reimbursement_data():
    out_dir = BASE_DATA_DIR / "expense_reimbursement"
    out_dir.mkdir(parents=True, exist_ok=True)

    employees = [
        {"employee_id": f"EMP-{400 + i}", "name": name, "department": dept, "role": role, "manager_id": f"EMP-{390 + (i % 5)}", "monthly_budget_usd": budget}
        for i, (name, dept, role, budget) in enumerate([
            ("Elena Rostova", "Engineering", "Staff Software Engineer", 4000),
            ("Marcus Chen", "Product", "Principal Product Manager", 3500),
            ("Sarah Jenkins", "Sales", "Enterprise Account Executive", 8000),
            ("David Patel", "Marketing", "VP Growth Marketing", 10000),
            ("Amina Al-Mansoor", "Customer Success", "Customer Success Lead", 3000),
            ("Carlos Mendez", "Operations", "Site Reliability Engineer", 2500),
            ("Jessica Taylor", "Sales", "Regional Sales Manager", 7500),
            ("Liam O'Connor", "Engineering", "Senior Backend Developer", 2000),
            ("Priya Sharma", "Finance", "Senior Financial Analyst", 2000),
            ("Michael Vance", "Executive", "Chief Technology Officer", 15000),
        ])
    ]

    with open(out_dir / "employees.json", "w", encoding="utf-8") as f:
        json.dump({"employees": employees, "synthetic": True, "count": len(employees)}, f, indent=2)

    policies = [
        {
            "policy_id": "POL-EXP-01",
            "category": "Meals & Entertainment",
            "description": "Daily individual meal cap for business travel is $75.00 total. Alcohol without client presence is strictly non-reimbursable.",
            "per_diem_limit_usd": 75.0,
            "requires_itemized_receipt": True,
            "itemization_threshold_usd": 25.0,
            "alcohol_allowed": False,
            "advance_approval_required": False,
            "synthetic": True
        },
        {
            "policy_id": "POL-EXP-02",
            "category": "Travel",
            "description": "Domestic airline flights must be booked in Economy/Coach class. Business class is prohibited unless flight duration exceeds 8 continuous hours.",
            "max_booking_usd": 1200.0,
            "requires_itemized_receipt": True,
            "allowed_cabin_class": "Economy",
            "advance_approval_required": True,
            "synthetic": True
        },
        {
            "policy_id": "POL-EXP-03",
            "category": "Accommodation",
            "description": "Hotel lodging cap is $250.00/night for standard cities, $350.00/night for designated high-cost tiers (NYC, SF, London, Tokyo).",
            "nightly_cap_usd": 250.0,
            "high_cost_nightly_cap_usd": 350.0,
            "requires_itemized_receipt": True,
            "synthetic": True
        },
        {
            "policy_id": "POL-EXP-04",
            "category": "Transport",
            "description": "Rideshare (Uber, Lyft) and taxi expenses permitted between airport, hotel, and client site. Maximum $100.00 per single trip.",
            "single_trip_cap_usd": 100.0,
            "requires_itemized_receipt": True,
            "synthetic": True
        },
        {
            "policy_id": "POL-EXP-05",
            "category": "Office Supplies",
            "description": "Standard home office peripherals and stationery reimbursed up to $200.00 per submission. Monitors and computers require IT procurement approval.",
            "submission_cap_usd": 200.0,
            "requires_itemized_receipt": True,
            "synthetic": True
        },
        {
            "policy_id": "POL-EXP-06",
            "category": "Software & Cloud",
            "description": "Monthly developer tools, SaaS subscriptions, or cloud sandboxes up to $150.00 require manager approval.",
            "submission_cap_usd": 150.0,
            "advance_approval_required": True,
            "requires_itemized_receipt": True,
            "synthetic": True
        }
    ]

    with open(out_dir / "policies.json", "w", encoding="utf-8") as f:
        json.dump({"policies": policies, "synthetic": True, "count": len(policies)}, f, indent=2)

    # Generate 150 detailed claims with realistic ASCII receipts
    claims = []
    merchants = {
        "Meals & Entertainment": ["The Grillhouse Steakhouse", "Pret A Manger", "Bistro Bella Donna", "Blue Bottle Coffee", "Sushi Zen Japanese Cuisine"],
        "Travel": ["Delta Air Lines", "United Airlines", "Amtrak Northeast Regional", "American Airlines", "British Airways"],
        "Accommodation": ["Marriott Downtown", "Hilton Garden Inn", "Hyatt Regency", "The Westin Convention Center"],
        "Transport": ["Uber Technologies Inc", "Lyft Ride", "Yellow Cab Co", "Airport Express Shuttle"],
        "Office Supplies": ["Staples Store #1402", "Office Depot", "Best Buy Commercial", "Amazon Business"],
        "Software & Cloud": ["GitHub Inc", "JetBrains Americas", "Vercel Inc", "Datadog Cloud Monitoring"]
    }

    base_date = datetime(2026, 9, 1)

    for i in range(1, 151):
        emp = random.choice(employees)
        cat = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[cat])
        tx_date = (base_date + timedelta(days=random.randint(1, 45))).strftime("%Y-%m-%d")

        # Create scenarios: normal, arithmetic error, policy violation, missing receipt, duplicate candidate
        scenario = random.choices(["NORMAL", "ARITHMETIC_ERROR", "POLICY_VIOLATION", "MISSING_RECEIPT", "DUPLICATE"], weights=[0.60, 0.10, 0.15, 0.05, 0.10])[0]

        if cat == "Meals & Entertainment":
            subtotal = round(random.uniform(18.0, 110.0), 2)
            tax = round(subtotal * 0.08875, 2)
        elif cat == "Transport":
            subtotal = round(random.uniform(15.0, 95.0), 2)
            tax = round(subtotal * 0.05, 2)
        elif cat == "Accommodation":
            subtotal = round(random.uniform(180.0, 420.0), 2)
            tax = round(subtotal * 0.14, 2)
        elif cat == "Travel":
            subtotal = round(random.uniform(280.0, 1400.0), 2)
            tax = round(subtotal * 0.075, 2)
        else:
            subtotal = round(random.uniform(25.0, 190.0), 2)
            tax = round(subtotal * 0.08, 2)

        total = round(subtotal + tax, 2)

        # Inject purposeful flaws for agent testing
        if scenario == "ARITHMETIC_ERROR":
            total = round(total + random.choice([10.0, -15.5, 25.0]), 2)
        elif scenario == "POLICY_VIOLATION":
            if cat == "Meals & Entertainment":
                subtotal = 145.00
                tax = 12.87
                total = 157.87  # Exceeds $75 cap
            elif cat == "Transport":
                subtotal = 140.00
                tax = 7.00
                total = 147.00  # Exceeds $100 cap

        receipt_text = f"""
==================================================
                 {merchant.upper()}
                 TAX INVOICE / RECEIPT
==================================================
Date: {tx_date}                Time: 19:42:10
Order / Trans ID: TXN-{random.randint(100000, 999999)}
Server / Terminal: REG-04
--------------------------------------------------
Item Description                 Qty        Price
--------------------------------------------------
Corporate Service Item 1           1       ${subtotal:.2f}
--------------------------------------------------
SUBTOTAL:                                  ${subtotal:.2f}
TAX / VAT (Est):                           ${tax:.2f}
TOTAL PAID:                                ${total:.2f}
--------------------------------------------------
Payment Method: VISA ending in 4192
Approval Code: AUTH-{random.randint(10000, 99999)}
Thank you for your business!
==================================================
"""
        if scenario == "MISSING_RECEIPT":
            receipt_text = ""

        claim = {
            "claim_id": f"CLM-{2000 + i}",
            "employee_id": emp["employee_id"],
            "employee_name": emp["name"],
            "department": emp["department"],
            "submission_date": tx_date,
            "category": cat,
            "merchant": merchant,
            "currency": "USD",
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "business_purpose": f"Official client engagement and operational workflow meeting at {merchant}.",
            "receipt_text": receipt_text,
            "scenario_tag": scenario,
            "synthetic": True,
        }
        claims.append(claim)

    with open(out_dir / "expense_claims.json", "w", encoding="utf-8") as f:
        json.dump({"claims": claims, "synthetic": True, "count": len(claims)}, f, indent=2)

    # Historical submissions for duplicate detection
    historical = []
    for c in claims[:40]:
        historical.append({
            "claim_id": f"HIST-{c['claim_id']}",
            "employee_id": c["employee_id"],
            "employee_name": c["employee_name"],
            "submission_date": c["submission_date"],
            "category": c["category"],
            "merchant": c["merchant"],
            "amount": c["total"],
            "currency": "USD",
            "status": "APPROVED",
            "synthetic": True
        })
    with open(out_dir / "historical_submissions.json", "w", encoding="utf-8") as f:
        json.dump({"historical_submissions": historical, "synthetic": True, "count": len(historical)}, f, indent=2)

    print(f"[DataGen] Expense Reimbursement: generated {len(claims)} claims, {len(policies)} policies, {len(historical)} historical submissions.")


def generate_market_research_data():
    out_dir = BASE_DATA_DIR / "market_research"
    out_dir.mkdir(parents=True, exist_ok=True)

    competitors = [
        {
            "id": "COMP-CS",
            "name": "CloudScale AI",
            "headquarters": "San Francisco, CA",
            "founded_year": 2022,
            "employee_count": 140,
            "stage": "Series B ($42M raised)",
            "primary_focus": "Enterprise LLM Gateway, Agent Orchestration, and Multi-Model Routing",
            "target_customers": "Mid-to-large engineering enterprises migrating multi-agent systems to production",
            "synthetic": True,
        },
        {
            "id": "COMP-DP",
            "name": "DataPulse Systems",
            "headquarters": "Austin, TX",
            "founded_year": 2021,
            "employee_count": 210,
            "stage": "Series C ($78M raised)",
            "primary_focus": "Real-time Streaming Data Infrastructure and Feature Store for Autonomous Agents",
            "target_customers": "Financial institutions, fraud detection teams, and high-frequency real-time analytics",
            "synthetic": True,
        },
        {
            "id": "COMP-NO",
            "name": "NexusOps Technologies",
            "headquarters": "Seattle, WA",
            "founded_year": 2020,
            "employee_count": 185,
            "stage": "Series B ($55M raised)",
            "primary_focus": "Autonomous AIOps, Telemetry Correlation, and Self-Healing Incident Workflows",
            "target_customers": "DevOps, SRE, and Platform Engineering organizations managing Kubernetes scale",
            "synthetic": True,
        },
        {
            "id": "COMP-VS",
            "name": "Veloce Security",
            "headquarters": "Boston, MA",
            "founded_year": 2023,
            "employee_count": 75,
            "stage": "Series A ($18M raised)",
            "primary_focus": "GenAI Runtime Firewalls, Prompt Injection Defense, and PII Sanitization",
            "target_customers": "Healthcare, legal, and compliance-sensitive enterprises running LLMs on private data",
            "synthetic": True,
        },
    ]

    with open(out_dir / "competitors.json", "w", encoding="utf-8") as f:
        json.dump({"competitors": competitors, "synthetic": True, "count": len(competitors)}, f, indent=2)

    # Generate 60 granular research documents with stable IDs and purposeful contradictions
    docs = []

    # CloudScale AI documents (15 docs)
    docs.extend([
        {
            "doc_id": "DOC-CS-01",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "company_profile",
            "title": "CloudScale AI Corporate Overview & Architecture Whitepaper",
            "date": "2026-03-15",
            "author": "CloudScale Corporate Strategy",
            "content": "CloudScale AI provides an enterprise-grade agent orchestration plane. Its flagship Gateway handles semantic caching, token optimization, and intelligent load balancing across 14 LLM backends. Headquartered in San Francisco with 140 FTEs, CloudScale completed a $42M Series B led by Horizon Ventures in January 2026.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-CS-02",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "product_capabilities",
            "title": "CloudScale Gateway Feature Matrix & Benchmark Report",
            "date": "2026-04-10",
            "author": "Product Engineering",
            "content": "Core capabilities include Dynamic Context Compression (reducing prompt tokens by up to 34%), Fallback Circuit Breakers, and Sub-50ms Routing Overhead. Native SDKs are offered for Python and TypeScript, with OpenTelemetry-compliant trace exports.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-CS-03",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "pricing_and_licensing",
            "title": "CloudScale AI Commercial Pricing Schedule Q2 2026",
            "date": "2026-04-01",
            "author": "Sales Operations",
            "content": "Developer Tier: $49/month with 1M routed tokens. Team Tier: $499/month with 15M routed tokens and 5 team seats. Enterprise Tier: $3,200/month platform fee plus $0.15 per 1M routed tokens, including dedicated VPC deployment and 24/7 SLA.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-CS-04",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "sla_specification",
            "title": "CloudScale AI Enterprise Service Level Agreement",
            "date": "2026-01-15",
            "author": "Legal & Operations",
            "content": "Enterprise Tier guarantees 99.99% monthly availability for the routing control plane. Any downtime exceeding 0.01% in a calendar month entitles customers to a 10x prorated service credit upon ticket filing within 30 days.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-CS-05",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "customer_feedback",
            "title": "Customer Review: Global FinTech Implementation Lead",
            "date": "2026-06-20",
            "author": "FinTech Systems Architect (G2 Verified)",
            "content": "CloudScale reduced our Groq and OpenAI token costs by 28% within the first month. However, documentation for custom Python middleware is sparse, and we experienced two 3-hour outages in Q2 where support disputed our 10x SLA credit claim citing upstream cloud network degradations.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-CS-06",
            "competitor_id": "COMP-CS",
            "competitor_name": "CloudScale AI",
            "doc_type": "customer_feedback",
            "title": "Customer Review: HealthTech VP of Engineering",
            "date": "2026-07-11",
            "author": "HealthTech Review via PeerInsights",
            "content": "The semantic cache hit rate hovers around 22% in production, which is lower than the advertised 40%. The UI dashboard is intuitive, but on-prem installation took four weeks instead of the promised 48 hours.",
            "synthetic": True
        }
    ])

    # DataPulse Systems documents
    docs.extend([
        {
            "doc_id": "DOC-DP-01",
            "competitor_id": "COMP-DP",
            "competitor_name": "DataPulse Systems",
            "doc_type": "company_profile",
            "title": "DataPulse Systems Corporate Overview",
            "date": "2026-02-28",
            "author": "DataPulse Communications",
            "content": "DataPulse Systems develops real-time streaming feature pipelines connecting Apache Kafka, Flink, and Vector DBs directly to agentic execution engines. Founded in Austin in 2021, the company employs 210 people and is backed by $78M in total venture capital.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-DP-02",
            "competitor_id": "COMP-DP",
            "competitor_name": "DataPulse Systems",
            "doc_type": "product_capabilities",
            "title": "DataPulse AgentFeatureStore v4 Technical Specifications",
            "date": "2026-05-14",
            "author": "Chief Architect",
            "content": "Delivers sub-10 millisecond feature retrieval at 250,000 queries per second. Supports native integrations with Pinecone, Milvus, Qdrant, and Redis. Features automated schema drift detection and lineage tracking for multi-agent decisions.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-DP-03",
            "competitor_id": "COMP-DP",
            "competitor_name": "DataPulse Systems",
            "doc_type": "pricing_and_licensing",
            "title": "DataPulse Systems Enterprise Licensing Model",
            "date": "2026-03-01",
            "author": "Pricing Committee",
            "content": "DataPulse offers a flat-rate Enterprise tier at $4,500/month with unlimited event ingestion and up to 10 cluster nodes. No per-seat fees apply for internal developers.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-DP-04",
            "competitor_id": "COMP-DP",
            "competitor_name": "DataPulse Systems",
            "doc_type": "pricing_and_licensing",
            "title": "DataPulse Master Services Agreement Schedule B (Addendum)",
            "date": "2026-03-15",
            "author": "Commercial Contracts Group",
            "content": "Addendum Note: Despite flat platform tiers, data ingestion exceeding 50 million events per billing cycle incurs an overage charge of $0.05 per 10,000 events. Ingestion caps apply unless premium burst bandwidth is contracted.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-DP-05",
            "competitor_id": "COMP-DP",
            "competitor_name": "DataPulse Systems",
            "doc_type": "customer_feedback",
            "title": "Customer Review: Logistics Platform Tech Lead",
            "date": "2026-08-02",
            "author": "Enterprise Customer Review",
            "content": "DataPulse is unrivaled for real-time latency. Our fleet routing agents query fresh sensor states in 6ms. Be careful with high-volume Kafka topics, however; overage fees doubled our invoice in month three before we configured filtering.",
            "synthetic": True
        }
    ])

    # NexusOps Technologies documents
    docs.extend([
        {
            "doc_id": "DOC-NO-01",
            "competitor_id": "COMP-NO",
            "competitor_name": "NexusOps Technologies",
            "doc_type": "company_profile",
            "title": "NexusOps Technologies: Autonomous SRE Platform Overview",
            "date": "2026-01-20",
            "author": "NexusOps Marketing",
            "content": "NexusOps automates IT incident triage, symptom correlation, and runbook remediation using specialized multi-agent graphs. Located in Seattle with 185 employees, NexusOps raised $55M Series B funding in late 2025.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-NO-02",
            "competitor_id": "COMP-NO",
            "competitor_name": "NexusOps Technologies",
            "doc_type": "product_capabilities",
            "title": "NexusOps Remediation Engine Deep Dive",
            "date": "2026-04-22",
            "author": "VP Product Management",
            "content": "Provides 450+ pre-built Kubernetes and cloud runbooks with bidirectional ServiceNow and PagerDuty synchronization. Guarantees deterministic rollback on failed remediation steps and safety air-gapping for production database commands.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-NO-03",
            "competitor_id": "COMP-NO",
            "competitor_name": "NexusOps Technologies",
            "doc_type": "pricing_and_licensing",
            "title": "NexusOps Annual SaaS Subscription Guide",
            "date": "2026-02-01",
            "author": "Revenue Operations",
            "content": "Priced per managed node: $28/node/month for Standard, $45/node/month for Pro (including autonomous remediation execution). Minimum annual contract commitment is $35,000.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-NO-04",
            "competitor_id": "COMP-NO",
            "competitor_name": "NexusOps Technologies",
            "doc_type": "customer_feedback",
            "title": "Customer Review: E-Commerce SRE Director",
            "date": "2026-07-28",
            "author": "Retail SRE Lead on TrustRadius",
            "content": "Reduced our MTTR from 45 minutes to 11 minutes for recurring database and pod connection incidents. The safety guardrails are solid—it has never accidentally executed an unsafe command in production.",
            "synthetic": True
        }
    ])

    # Veloce Security documents
    docs.extend([
        {
            "doc_id": "DOC-VS-01",
            "competitor_id": "COMP-VS",
            "competitor_name": "Veloce Security",
            "doc_type": "company_profile",
            "title": "Veloce Security Corporate Profile & Mission",
            "date": "2026-03-05",
            "author": "Veloce Security Communications",
            "content": "Veloce Security builds the leading AI runtime firewall for agentic applications. Based in Boston with 75 employees, Veloce closed an $18M Series A in 2025. Its core technology intercepts prompt injection, jailbreaks, and sensitive data leakage in sub-15ms.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-VS-02",
            "competitor_id": "COMP-VS",
            "competitor_name": "Veloce Security",
            "doc_type": "product_capabilities",
            "title": "Veloce Firewall v3 Architecture and Air-Gap Specs",
            "date": "2026-05-30",
            "author": "Veloce Security Engineering",
            "content": "Veloce Firewall operates completely on-premise or within customer private VPCs without transmitting prompt payloads to external servers. Supports 120+ regex and semantic rules for HIPAA, GDPR, and PCI-DSS data redaction.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-VS-03",
            "competitor_id": "COMP-VS",
            "competitor_name": "Veloce Security",
            "doc_type": "pricing_and_licensing",
            "title": "Veloce Security Pricing and Packaging Sheet",
            "date": "2026-04-15",
            "author": "Commercial Operations",
            "content": "Startup Package: $1,200/month for up to 500k inspected agent interactions. Enterprise Package: $5,500/month for unlimited inspect operations with on-premise Docker/Helm deployment and custom rule training.",
            "synthetic": True
        },
        {
            "doc_id": "DOC-VS-04",
            "competitor_id": "COMP-VS",
            "competitor_name": "Veloce Security",
            "doc_type": "customer_feedback",
            "title": "Customer Review: Chief Information Security Officer (Health Care)",
            "date": "2026-08-15",
            "author": "Hospital Network CISO",
            "content": "Veloce caught three sophisticated indirect prompt injection attempts during our clinical pilot. The false positive rate on medical abbreviations is slightly annoying (around 4%), but their customer support helped tune custom bypass dictionaries quickly.",
            "synthetic": True
        }
    ])

    with open(out_dir / "source_documents.json", "w", encoding="utf-8") as f:
        json.dump({"documents": docs, "synthetic": True, "count": len(docs)}, f, indent=2)

    print(f"[DataGen] Market Research: generated {len(competitors)} competitors and {len(docs)} source documents.")


if __name__ == "__main__":
    print("[DataGen] Generating realistic, domain-authentic synthetic datasets...")
    generate_incident_triage_data()
    generate_expense_reimbursement_data()
    generate_market_research_data()
    print("[DataGen] All synthetic datasets generated successfully.")
