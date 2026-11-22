"""
TraceProfiler Visualization Pipeline.

Generates comprehensive, publication-grade analytical plots from workflow trace CSV files:
- Plot 1: Workflow-wise latency distribution (Box plot + jitter)
- Plot 2: Workflow-wise token distribution (Box plot + jitter)
- Plot 3: Workflow performance trends across 30 runs (3 separate 2x5 figures: Latency, Tokens, Cost)
- Plot 4: Agent-wise latency distribution across all workflows (2x5 grid, common Y-scale)
- Plot 5: Agent-wise token-consumption distribution (2x5 grid, common Y-scale)
- Plot 6: Agent prompt vs completion token comparison (2x5 grid, grouped horizontal barh)
- Plot 10: Agent latency percentile comparison (P50, P90, P95 grouped bars with cautious statistical note)

Saves all figures into visualisations/trace_analysis/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Set Matplotlib cache dir inside project to avoid permission warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".cache" / "matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

# Style configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "worklow_traces_csv"
OUTPUT_DIR = BASE_DIR / "visualisations" / "trace_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 Target Workflows in alphabetical / canonical catalog order
WORKFLOW_NAMES = [
    "contract_review",
    "customer_support",
    "expense_reimbursement",
    "incident_triage",
    "inventory_replenishment",
    "invoice_processing",
    "market_research",
    "procurement_supplier_selection",
    "recruitment_screening",
    "sales_lead_qualification",
]

PALETTE = sns.color_palette("tab10", n_colors=10)
WORKFLOW_COLORS = {wf: PALETTE[i] for i, wf in enumerate(WORKFLOW_NAMES)}


def load_and_prepare_data():
    """
    Loads all workflow trace CSVs and prepares:
    1. agent_df: granular agent execution records
    2. workflow_df: de-duplicated workflow-level records (1 per unique run_id)
    """
    agent_dfs = []
    for wf in WORKFLOW_NAMES:
        csv_path = CSV_DIR / wf / f"{wf}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected trace CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        agent_dfs.append(df)

    agent_df = pd.concat(agent_dfs, ignore_index=True)

    # Cast numeric columns cleanly
    numeric_cols = [
        "workflow_duration_seconds",
        "agent_execution_count",
        "total_tokens",
        "workflow_prompt_tokens",
        "workflow_completion_tokens",
        "estimated_cost_usd",
        "agent_duration_seconds",
        "tokens",
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "order",
    ]
    for col in numeric_cols:
        if col in agent_df.columns:
            agent_df[col] = pd.to_numeric(agent_df[col], errors="coerce")

    # Workflow-level de-duplicated dataframe
    workflow_df = agent_df.drop_duplicates(subset=["run_id"]).copy()
    workflow_df["timestamp_dt"] = pd.to_datetime(workflow_df["timestamp"], errors="coerce")

    # Validation
    for wf in WORKFLOW_NAMES:
        sub_wf = workflow_df[workflow_df["workflow_name"] == wf]
        sub_ag = agent_df[agent_df["workflow_name"] == wf]
        runs_count = len(sub_wf)
        agent_count = sub_ag["agent_id"].nunique()
        assert runs_count == 30, f"Workflow {wf} has {runs_count} runs (expected 30)"
        print(f"[Data Prep] {wf:<32}: {runs_count} runs, {agent_count} unique agents, {len(sub_ag)} execution rows")

    return workflow_df, agent_df


# ==============================================================================
# PLOT 1: Workflow-wise latency distribution
# ==============================================================================
def plot_1_workflow_latency_distribution(workflow_df: pd.DataFrame):
    print("[Plot 1] Generating Workflow-wise Latency Distribution...")
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Prepare labels showing workflow name + run count
    labels = []
    for wf in WORKFLOW_NAMES:
        n_runs = len(workflow_df[workflow_df["workflow_name"] == wf])
        labels.append(f"{wf}\n(n={n_runs})")

    sns.boxplot(
        data=workflow_df,
        x="workflow_name",
        y="workflow_duration_seconds",
        order=WORKFLOW_NAMES,
        palette=WORKFLOW_COLORS,
        ax=ax,
        width=0.55,
        boxprops=dict(alpha=0.75, edgecolor="black", linewidth=1.2),
        medianprops=dict(color="crimson", linewidth=2.0),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="D", color="red", markersize=6, alpha=0.8),
    )

    # Jittered strip plot overlay
    sns.stripplot(
        data=workflow_df,
        x="workflow_name",
        y="workflow_duration_seconds",
        order=WORKFLOW_NAMES,
        color="black",
        alpha=0.55,
        size=5,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("End-to-End Workflow Latency Distribution Across 10 Workflows", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Workflow Name (with Run Count)", fontsize=12, labelpad=10)
    ax.set_ylabel("Workflow Duration (Seconds)", fontsize=12, labelpad=10)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Descriptive Legend
    legend_elements = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Interquartile Range (IQR: Q1 - Q3)"),
        mlines.Line2D([], [], color="crimson", linewidth=2.0, label="Median Latency"),
        mlines.Line2D([], [], color="black", linewidth=1.2, label="Whiskers (1.5 × IQR)"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="red", markersize=7, label="Statistical Outliers (> 1.5 × IQR)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="black", markersize=6, alpha=0.6, label="Individual Workflow Run (n=30 each)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", shadow=True)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "plot1_workflow_latency_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


# ==============================================================================
# PLOT 2: Workflow-wise token distribution
# ==============================================================================
def plot_2_workflow_token_distribution(workflow_df: pd.DataFrame):
    print("[Plot 2] Generating Workflow-wise Token Distribution...")
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    labels = []
    for wf in WORKFLOW_NAMES:
        n_runs = len(workflow_df[workflow_df["workflow_name"] == wf])
        labels.append(f"{wf}\n(n={n_runs})")

    sns.boxplot(
        data=workflow_df,
        x="workflow_name",
        y="total_tokens",
        order=WORKFLOW_NAMES,
        palette=WORKFLOW_COLORS,
        ax=ax,
        width=0.55,
        boxprops=dict(alpha=0.75, edgecolor="black", linewidth=1.2),
        medianprops=dict(color="navy", linewidth=2.0),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="D", color="purple", markersize=6, alpha=0.8),
    )

    sns.stripplot(
        data=workflow_df,
        x="workflow_name",
        y="total_tokens",
        order=WORKFLOW_NAMES,
        color="black",
        alpha=0.55,
        size=5,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("Total Token Consumption Distribution Across 10 Workflows", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Workflow Name (with Run Count)", fontsize=12, labelpad=10)
    ax.set_ylabel("Total Tokens", fontsize=12, labelpad=10)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Format Y axis with comma thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    legend_elements = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Interquartile Range (IQR: Q1 - Q3)"),
        mlines.Line2D([], [], color="navy", linewidth=2.0, label="Median Token Consumption"),
        mlines.Line2D([], [], color="black", linewidth=1.2, label="Whiskers (1.5 × IQR)"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="purple", markersize=7, label="Outliers (> 1.5 × IQR)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="black", markersize=6, alpha=0.6, label="Individual Run Tokens (n=30 each)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", shadow=True)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "plot2_workflow_token_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


# ==============================================================================
# PLOT 3: Workflow performance trends across 30 runs (A, B, C)
# ==============================================================================
def plot_3_workflow_trends(workflow_df: pd.DataFrame, agent_df: pd.DataFrame):
    print("[Plot 3] Generating Performance Trends (Latency, Tokens, Cost)...")

    # Compute agent counts per workflow
    agent_counts = {wf: agent_df[agent_df["workflow_name"] == wf]["agent_id"].nunique() for wf in WORKFLOW_NAMES}

    # Prepare chronologically sorted runs (1..30) per workflow
    wf_runs_data = {}
    for wf in WORKFLOW_NAMES:
        sub = workflow_df[workflow_df["workflow_name"] == wf].copy()
        # Sort by timestamp
        sub = sub.sort_values(by="timestamp_dt").reset_index(drop=True)
        sub["run_num"] = range(1, len(sub) + 1)
        wf_runs_data[wf] = sub

    def _render_trend_grid(metric_col: str, y_label: str, fig_title: str, file_name: str, y_formatter=None):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 11), dpi=300, sharex=True)
        axes = axes.flatten()

        for idx, wf in enumerate(WORKFLOW_NAMES):
            ax = axes[idx]
            sub = wf_runs_data[wf]
            color = WORKFLOW_COLORS[wf]
            n_agents = agent_counts[wf]
            n_runs = len(sub)

            y_vals = sub[metric_col].values
            x_vals = sub["run_num"].values

            # Line plot
            ax.plot(x_vals, y_vals, color=color, linewidth=1.8, marker="o", markersize=4, label="Execution Run")

            # Mean reference line
            mean_val = np.mean(y_vals)
            ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1.0, alpha=0.75, label=f"Mean: {mean_val:.2f}")

            # Outlier / Unusual points detection (beyond mean +/- 2 std)
            std_val = np.std(y_vals)
            threshold_high = mean_val + 2 * std_val
            threshold_low = mean_val - 2 * std_val

            unusual_mask = (y_vals > threshold_high) | (y_vals < threshold_low)
            if np.any(unusual_mask):
                ax.scatter(
                    x_vals[unusual_mask],
                    y_vals[unusual_mask],
                    color="crimson",
                    s=60,
                    zorder=5,
                    marker="^",
                    label="Unusual (±2σ)",
                )

            # Subplot title format: [Workflow Name] | [Agent Count] Agents | [Run Count] Runs
            ax.set_title(f"{wf}\n{n_agents} Agents | {n_runs} Runs", fontsize=10.5, weight="bold", pad=8)
            ax.set_xlabel("Workflow Run Number (1-30)", fontsize=9.5)
            ax.set_ylabel(y_label, fontsize=9.5)
            ax.set_xticks(range(5, 31, 5))
            ax.grid(True, linestyle=":", alpha=0.6)

            if y_formatter:
                ax.yaxis.set_major_formatter(y_formatter)

            ax.legend(loc="best", fontsize=8, framealpha=0.85)

        fig.suptitle(fig_title, fontsize=16, weight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out_path = OUTPUT_DIR / file_name
        plt.savefig(out_path)
        plt.close()
        print(f"  -> Saved: {out_path}")

    # Figure 3A: Latency
    _render_trend_grid(
        metric_col="workflow_duration_seconds",
        y_label="Workflow Duration (Seconds)",
        fig_title="Workflow Latency Trends Across 30 Executions",
        file_name="plot3a_workflow_latency_trend.png",
    )

    # Figure 3B: Tokens
    _render_trend_grid(
        metric_col="total_tokens",
        y_label="Total Tokens",
        fig_title="Workflow Token Consumption Trends Across 30 Executions",
        file_name="plot3b_workflow_token_trend.png",
        y_formatter=plt.FuncFormatter(lambda x, p: f"{int(x):,}"),
    )

    # Figure 3C: Cost
    _render_trend_grid(
        metric_col="estimated_cost_usd",
        y_label="Estimated Cost (USD)",
        fig_title="Workflow Execution Cost Trends Across 30 Executions",
        file_name="plot3c_workflow_cost_trend.png",
        y_formatter=plt.FuncFormatter(lambda x, p: f"${x:.4f}"),
    )


# ==============================================================================
# PLOT 4: Agent-wise latency distribution across all workflows
# ==============================================================================
def plot_4_agent_latency_distribution(agent_df: pd.DataFrame):
    print("[Plot 4] Generating Agent-wise Latency Distribution (2x5 Consolidated Grid)...")

    # Common Y-scale for cross-workflow comparison
    max_lat = agent_df["agent_duration_seconds"].max() * 1.08

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 13), dpi=300, sharey=True)
    axes = axes.flatten()

    for idx, wf in enumerate(WORKFLOW_NAMES):
        ax = axes[idx]
        sub = agent_df[agent_df["workflow_name"] == wf].copy()

        # Sort agents by execution 'order'
        agent_order_map = sub.groupby("agent_id")["order"].min().to_dict()
        sorted_agents = sorted(agent_order_map.keys(), key=lambda a: agent_order_map[a])

        n_agents = len(sorted_agents)
        n_runs = sub["run_id"].nunique()

        sns.boxplot(
            data=sub,
            x="agent_id",
            y="agent_duration_seconds",
            order=sorted_agents,
            ax=ax,
            width=0.6,
            color=WORKFLOW_COLORS[wf],
            boxprops=dict(alpha=0.65, edgecolor="black", linewidth=1.0),
            medianprops=dict(color="crimson", linewidth=1.8),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black", markersize=4),
            flierprops=dict(marker="D", color="red", markersize=4, alpha=0.7),
        )

        sns.stripplot(
            data=sub,
            x="agent_id",
            y="agent_duration_seconds",
            order=sorted_agents,
            color="black",
            alpha=0.45,
            size=3.5,
            jitter=0.2,
            ax=ax,
        )

        ax.set_title(f"{wf} | {n_agents} Agents | {n_runs} Runs", fontsize=10.5, weight="bold", pad=8)
        ax.set_xlabel("Agent ID (Execution Order)", fontsize=9.5)
        ax.set_ylabel("Agent Execution Latency (Seconds)", fontsize=9.5)
        ax.set_ylim(0, max_lat)
        ax.set_xticklabels(sorted_agents, rotation=45, ha="right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Legend for the box plot elements
    legend_elements = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Interquartile Range (Q1 - Q3)"),
        mlines.Line2D([], [], color="crimson", linewidth=1.8, label="Median Latency"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="red", markeredgecolor="black", markersize=6, label="Mean Latency"),
        mlines.Line2D([], [], color="black", linewidth=1.0, label="Whiskers (1.5 × IQR)"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="red", markersize=6, label="Outliers (> 1.5 × IQR)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="black", markersize=4, alpha=0.5, label="Run Observations (n=30)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, 0.005), frameon=True, facecolor="white")

    fig.suptitle("Agent-wise Latency Distribution Across All Workflows (Common Y-Scale)", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = OUTPUT_DIR / "plot4_agent_latency_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


# ==============================================================================
# PLOT 5: Agent-wise token-consumption distribution
# ==============================================================================
def plot_5_agent_token_distribution(agent_df: pd.DataFrame):
    print("[Plot 5] Generating Agent-wise Token Distribution (2x5 Consolidated Grid)...")

    max_tok = agent_df["tokens"].max() * 1.08

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 13), dpi=300, sharey=True)
    axes = axes.flatten()

    for idx, wf in enumerate(WORKFLOW_NAMES):
        ax = axes[idx]
        sub = agent_df[agent_df["workflow_name"] == wf].copy()

        agent_order_map = sub.groupby("agent_id")["order"].min().to_dict()
        sorted_agents = sorted(agent_order_map.keys(), key=lambda a: agent_order_map[a])

        n_agents = len(sorted_agents)

        sns.boxplot(
            data=sub,
            x="agent_id",
            y="tokens",
            order=sorted_agents,
            ax=ax,
            width=0.6,
            color=WORKFLOW_COLORS[wf],
            boxprops=dict(alpha=0.65, edgecolor="black", linewidth=1.0),
            medianprops=dict(color="navy", linewidth=1.8),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="navy", markeredgecolor="black", markersize=4),
            flierprops=dict(marker="D", color="purple", markersize=4, alpha=0.7),
        )

        sns.stripplot(
            data=sub,
            x="agent_id",
            y="tokens",
            order=sorted_agents,
            color="black",
            alpha=0.45,
            size=3.5,
            jitter=0.2,
            ax=ax,
        )

        # Title required: [Workflow Name] | [Agent Count] Agents | [30 Runs]
        ax.set_title(f"{wf} | {n_agents} Agents | 30 Runs", fontsize=10.5, weight="bold", pad=8)
        ax.set_xlabel("Agent ID (Execution Order)", fontsize=9.5)
        ax.set_ylabel("Total Agent Tokens", fontsize=9.5)
        ax.set_ylim(0, max_tok)
        ax.set_xticklabels(sorted_agents, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    legend_elements = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Interquartile Range (Q1 - Q3)"),
        mlines.Line2D([], [], color="navy", linewidth=1.8, label="Median Tokens"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="navy", markeredgecolor="black", markersize=6, label="Mean Tokens"),
        mlines.Line2D([], [], color="black", linewidth=1.0, label="Whiskers (1.5 × IQR)"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="purple", markersize=6, label="Outliers (> 1.5 × IQR)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="black", markersize=4, alpha=0.5, label="Run Observations (n=30)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, 0.005), frameon=True, facecolor="white")

    fig.suptitle("Agent-wise Token Consumption Distribution Across Workflows (Common Y-Scale)", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = OUTPUT_DIR / "plot5_agent_token_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


# ==============================================================================
# PLOT 6: Agent prompt vs completion token comparison
# ==============================================================================
def plot_6_agent_prompt_vs_completion_tokens(agent_df: pd.DataFrame):
    print("[Plot 6] Generating Agent Prompt vs Completion Token Comparison (Grouped Barh)...")

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 13), dpi=300)
    axes = axes.flatten()

    for idx, wf in enumerate(WORKFLOW_NAMES):
        ax = axes[idx]
        sub = agent_df[agent_df["workflow_name"] == wf].copy()

        # Group by agent_id and compute median prompt tokens and completion tokens
        agent_order_map = sub.groupby("agent_id")["order"].min().to_dict()
        sorted_agents = sorted(agent_order_map.keys(), key=lambda a: agent_order_map[a])

        medians = sub.groupby("agent_id")[["agent_prompt_tokens", "agent_completion_tokens"]].median()
        medians = medians.loc[sorted_agents]

        n_agents = len(sorted_agents)
        n_runs = sub["run_id"].nunique()

        # Invert agents list so top execution order appears at the top of the horizontal bar chart
        plot_agents = list(reversed(sorted_agents))
        y_pos = np.arange(len(plot_agents))
        bar_height = 0.38

        prompt_vals = [medians.loc[a, "agent_prompt_tokens"] for a in plot_agents]
        comp_vals = [medians.loc[a, "agent_completion_tokens"] for a in plot_agents]

        # Grouped horizontal bars (prompt tokens top, completion tokens bottom)
        bars_p = ax.barh(y_pos + bar_height / 2, prompt_vals, height=bar_height, color="#1f77b4", edgecolor="black", alpha=0.85, label="Prompt Tokens")
        bars_c = ax.barh(y_pos - bar_height / 2, comp_vals, height=bar_height, color="#2ca02c", edgecolor="black", alpha=0.85, label="Completion Tokens")

        ax.set_title(f"{wf}\n{n_agents} Agents | {n_runs} Runs", fontsize=10.5, weight="bold", pad=8)
        ax.set_xlabel("Median Token Count", fontsize=9.5)
        ax.set_ylabel("Agent ID", fontsize=9.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_agents, fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Include legend in each subplot or globally
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle("Agent-wise Prompt and Completion Token Comparison", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = OUTPUT_DIR / "plot6_agent_prompt_vs_completion_tokens.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


# ==============================================================================
# PLOT 10: Agent latency percentile comparison (P50, P90, P95)
# ==============================================================================
def plot_10_agent_latency_percentiles(agent_df: pd.DataFrame):
    print("[Plot 10] Generating Agent Latency Percentile Comparison (P50, P90, P95)...")

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 13), dpi=300)
    axes = axes.flatten()

    for idx, wf in enumerate(WORKFLOW_NAMES):
        ax = axes[idx]
        sub = agent_df[agent_df["workflow_name"] == wf].copy()

        agent_order_map = sub.groupby("agent_id")["order"].min().to_dict()
        sorted_agents = sorted(agent_order_map.keys(), key=lambda a: agent_order_map[a])

        # Calculate percentiles per agent
        p50 = sub.groupby("agent_id")["agent_duration_seconds"].quantile(0.50).loc[sorted_agents]
        p90 = sub.groupby("agent_id")["agent_duration_seconds"].quantile(0.90).loc[sorted_agents]
        p95 = sub.groupby("agent_id")["agent_duration_seconds"].quantile(0.95).loc[sorted_agents]

        n_agents = len(sorted_agents)
        obs_count = len(sub)

        x = np.arange(len(sorted_agents))
        bar_width = 0.25

        ax.bar(x - bar_width, p50, width=bar_width, color="#4575b4", edgecolor="black", alpha=0.9, label="P50: Median Latency")
        ax.bar(x, p90, width=bar_width, color="#fdae61", edgecolor="black", alpha=0.9, label="P90: 90th Percentile Latency")
        ax.bar(x + bar_width, p95, width=bar_width, color="#d73027", edgecolor="black", alpha=0.9, label="P95: 95th Percentile Latency")

        ax.set_title(f"{wf}\n{n_agents} Agents | {obs_count} Observations", fontsize=10.5, weight="bold", pad=8)
        ax.set_xlabel("Agent ID (Execution Order)", fontsize=9.5)
        ax.set_ylabel("Latency (Seconds)", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_agents, rotation=45, ha="right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle("Agent-wise Latency Percentile Comparison Across Workflows", fontsize=16, weight="bold", y=0.99)

    # Footnote warning banner across the figure
    fig.text(
        0.5,
        0.01,
        "Note: P95 estimates are based on only 30 observations per agent and should be interpreted cautiously; P99 is omitted due to sample size instability.",
        ha="center",
        fontsize=10.5,
        style="italic",
        color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2f2", edgecolor="#ffcccc", linewidth=1.2),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = OUTPUT_DIR / "plot10_agent_latency_percentiles.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Saved: {out_path}")


def main():
    print(f"[TraceProfiler Visualizations] Starting pipeline...")
    print(f"  Source CSV Dir : {CSV_DIR}")
    print(f"  Output Plot Dir: {OUTPUT_DIR}\n")

    workflow_df, agent_df = load_and_prepare_data()

    plot_1_workflow_latency_distribution(workflow_df)
    plot_2_workflow_token_distribution(workflow_df)
    plot_3_workflow_trends(workflow_df, agent_df)
    plot_4_agent_latency_distribution(agent_df)
    plot_5_agent_token_distribution(agent_df)
    plot_6_agent_prompt_vs_completion_tokens(agent_df)
    plot_10_agent_latency_percentiles(agent_df)

    print("\n[TraceProfiler Visualizations] All 9 figures successfully generated!")


if __name__ == "__main__":
    main()

