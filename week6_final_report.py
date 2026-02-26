#!/usr/bin/env python3
"""
Week 6 Final Report Generator

Generates a comprehensive 10-15 page technical report including:
1. Executive Summary
2. EDA Findings with Visualizations
3. Feature Engineering Documentation
4. Model Performance Metrics
5. Flagged User Analysis
6. Recommendations

This script reads outputs from weeks 1-5 and generates the final deliverable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def format_number(n: float | int, decimals: int = 0) -> str:
    """Format a number with commas."""
    if pd.isna(n):
        return "N/A"
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def generate_executive_summary(scored_df: pd.DataFrame, breakdown: dict) -> str:
    """Generate the executive summary section."""
    lines = []
    lines.append("# Account Sharing Detection - Technical Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    
    total_users = len(scored_df)
    high_risk = int(breakdown["counts"].get("high", 0))
    medium_risk = int(breakdown["counts"].get("medium", 0))
    low_risk = int(breakdown["counts"].get("low", 0))
    
    flag_counts = breakdown.get("flag_counts", {})
    
    lines.append("### Key Findings")
    lines.append("")
    lines.append(f"We analyzed **{format_number(total_users)} user accounts** for potential account sharing behavior using five detection methods. Our analysis identified:")
    lines.append("")
    lines.append(f"| Risk Level | Count | % of Total |")
    lines.append(f"|------------|-------|------------|")
    lines.append(f"| **High Risk** | {format_number(high_risk)} | {100*high_risk/total_users:.2f}% |")
    lines.append(f"| **Medium Risk** | {format_number(medium_risk)} | {100*medium_risk/total_users:.2f}% |")
    lines.append(f"| Low Risk | {format_number(low_risk)} | {100*low_risk/total_users:.2f}% |")
    lines.append("")
    
    lines.append("### Detection Method Breakdown")
    lines.append("")
    lines.append("| Detection Method | Users Flagged | Description |")
    lines.append("|-----------------|---------------|-------------|")
    
    flag_descriptions = {
        "flag_impossible_travel": ("Geographic Impossibility", "Logins from locations requiring impossible travel speed (>500 mph)"),
        "flag_concurrent": ("Concurrent Activity", "Simultaneous activity from multiple distant IP addresses"),
        "flag_device_diversity": ("Device Diversity", "4+ different device types used on the account"),
        "flag_post_success": ("Post-Success Activity", "Continued usage after user reported passing their exam"),
        "flag_reset_shift": ("Password Reset Shift", "Significant behavior change after password reset"),
    }
    
    for flag_col, (name, desc) in flag_descriptions.items():
        count = flag_counts.get(flag_col, 0)
        lines.append(f"| {name} | {format_number(count)} | {desc} |")
    
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> High-risk accounts should be reviewed manually before any enforcement action.")
    lines.append("> Medium-risk accounts warrant monitoring and may benefit from verification prompts.")
    lines.append("")
    
    return "\n".join(lines)


def generate_eda_section(week1_summary: dict) -> str:
    """Generate the EDA findings section."""
    lines = []
    lines.append("## 2. Exploratory Data Analysis")
    lines.append("")
    lines.append("### 2.1 Dataset Overview")
    lines.append("")
    
    datasets = week1_summary.get("datasets", {})
    
    lines.append("| Dataset | Rows | Unique Users | Date Range |")
    lines.append("|---------|------|--------------|------------|")
    
    for name, info in datasets.items():
        profile = info.get("profile", {})
        rows = profile.get("rows", 0)
        users = info.get("user_event_stats", {}).get("unique_users", 0)
        
        date_ranges = info.get("date_ranges", {})
        date_range = "N/A"
        for col, dr in date_ranges.items():
            if dr.get("min") and dr.get("max"):
                min_date = dr["min"][:10] if isinstance(dr["min"], str) else str(dr["min"])[:10]
                max_date = dr["max"][:10] if isinstance(dr["max"], str) else str(dr["max"])[:10]
                date_range = f"{min_date} to {max_date}"
                break
        
        lines.append(f"| {name} | {format_number(rows)} | {format_number(users)} | {date_range} |")
    
    lines.append("")
    lines.append("### 2.2 Data Quality Notes")
    lines.append("")
    
    issues = []
    for name, info in datasets.items():
        checks = info.get("checks", [])
        for check in checks[:3]:  # Limit to top 3 per dataset
            issues.append(f"- **{name}**: {check}")
    
    if issues:
        for issue in issues[:10]:  # Limit to top 10 overall
            lines.append(issue)
    else:
        lines.append("- No significant data quality issues detected.")
    
    lines.append("")
    
    return "\n".join(lines)


def generate_feature_engineering_section(features_df: pd.DataFrame) -> str:
    """Generate the feature engineering documentation section."""
    lines = []
    lines.append("## 3. Feature Engineering")
    lines.append("")
    lines.append("### 3.1 Primary Detection Features")
    lines.append("")
    
    feature_docs = [
        ("Geographic Impossibility", [
            ("geo_max_speed_mph", "Maximum travel speed between consecutive logins (miles per hour)"),
            ("geo_impossible_travel_count", "Number of login pairs requiring >500 mph travel"),
            ("geo_max_distance_from_registration", "Maximum distance of logins from registration location"),
        ]),
        ("Concurrent Activity", [
            ("concurrent_max_distinct_ips", "Maximum distinct IPs seen within 10-minute windows"),
            ("concurrent_event_ratio", "Proportion of activity occurring with multiple simultaneous IPs"),
            ("content_unique_ips", "Total unique IPs used for content activity"),
        ]),
        ("Device Diversity", [
            ("unique_device_families", "Count of device types (iOS, Android, Windows, Mac, etc.)"),
            ("unique_browser_families", "Count of browser types used"),
            ("device_switches", "Number of device type changes across sessions"),
        ]),
        ("Account Handoff", [
            ("password_reset_count", "Total password resets for the account"),
            ("password_resets_after_positive", "Resets after user reported passing exam"),
            ("reset_device_diversity_change", "Change in device diversity after password reset"),
            ("post_positive_content_count", "Content activity after passing status"),
            ("post_positive_exam_count", "Exam attempts after passing status"),
        ]),
    ]
    
    for category, features in feature_docs:
        lines.append(f"#### {category}")
        lines.append("")
        lines.append("| Feature | Description | Median | P95 |")
        lines.append("|---------|-------------|--------|-----|")
        
        for feat_name, desc in features:
            if feat_name in features_df.columns:
                series = pd.to_numeric(features_df[feat_name], errors="coerce")
                median = series.median()
                p95 = series.quantile(0.95)
                lines.append(f"| `{feat_name}` | {desc} | {format_number(median, 2)} | {format_number(p95, 2)} |")
            else:
                lines.append(f"| `{feat_name}` | {desc} | N/A | N/A |")
        
        lines.append("")
    
    lines.append("### 3.2 Flag Thresholds")
    lines.append("")
    lines.append("The following thresholds were used to flag suspicious accounts:")
    lines.append("")
    lines.append("| Flag | Condition | Rationale |")
    lines.append("|------|-----------|-----------|")
    lines.append("| Impossible Travel | `geo_impossible_travel_count > 0` | Any instance of >500 mph travel is physically impossible |")
    lines.append("| Concurrent Activity | `concurrent_max_distinct_ips >= 2` AND `concurrent_event_ratio > 0.05` | Multiple IPs in short windows suggests sharing |")
    lines.append("| Device Diversity | `unique_device_families >= 4` | Using 4+ device types is unusual for single users |")
    lines.append("| Post-Success Activity | `post_positive_content_count > 0` OR `post_positive_exam_count > 0` | Activity after passing exam suggests handoff |")
    lines.append("| Reset Behavior Shift | `reset_device_diversity_change >= 2` | Major behavior change after password reset |")
    lines.append("")
    
    return "\n".join(lines)


def generate_model_section(breakdown: dict, scored_df: pd.DataFrame) -> str:
    """Generate the model performance section."""
    lines = []
    lines.append("## 4. Model & Scoring")
    lines.append("")
    lines.append("### 4.1 Ensemble Scoring Approach")
    lines.append("")
    lines.append("We combined rule-based heuristic flags with anomaly detection for robust scoring:")
    lines.append("")
    
    model_meta = breakdown.get("model_meta", {})
    weights = breakdown.get("weights", {})
    
    lines.append("```")
    lines.append(f"final_score = {weights.get('risk', 0.7):.0%} × heuristic_risk_score + {weights.get('model', 0.3):.0%} × anomaly_model_score")
    lines.append("```")
    lines.append("")
    
    lines.append(f"**Anomaly Model:** {model_meta.get('model', 'N/A')}")
    if model_meta.get("n_estimators"):
        lines.append(f"- Trees: {model_meta['n_estimators']}")
    lines.append("")
    
    lines.append("### 4.2 Risk Tier Thresholds")
    lines.append("")
    
    thresholds = breakdown.get("thresholds", {})
    lines.append(f"- **High Risk:** final_score ≥ {thresholds.get('high', 'N/A'):.4f}")
    lines.append(f"- **Medium Risk:** final_score ≥ {thresholds.get('medium', 'N/A'):.4f}")
    lines.append(f"- **Low Risk:** below medium threshold")
    lines.append("")
    
    lines.append("### 4.3 Score Distribution")
    lines.append("")
    
    if "final_score" in scored_df.columns:
        final_scores = pd.to_numeric(scored_df["final_score"], errors="coerce")
        lines.append(f"- **Minimum:** {final_scores.min():.4f}")
        lines.append(f"- **Median:** {final_scores.median():.4f}")
        lines.append(f"- **Mean:** {final_scores.mean():.4f}")
        lines.append(f"- **P95:** {final_scores.quantile(0.95):.4f}")
        lines.append(f"- **Maximum:** {final_scores.max():.4f}")
    lines.append("")
    
    return "\n".join(lines)


def generate_flagged_users_section(scored_df: pd.DataFrame, breakdown: dict) -> str:
    """Generate the flagged user analysis section."""
    lines = []
    lines.append("## 5. Flagged User Analysis")
    lines.append("")
    
    # Top high-risk users
    lines.append("### 5.1 Top High-Risk Users")
    lines.append("")
    lines.append("The following users have the highest risk scores:")
    lines.append("")
    
    top_users = scored_df.sort_values("final_score", ascending=False).head(15)
    
    lines.append("| User ID | Risk Tier | Final Score | Flags |")
    lines.append("|---------|-----------|-------------|-------|")
    
    for _, row in top_users.iterrows():
        user_id = str(row.get("user_id", ""))[:36]  # Truncate UUID
        tier = row.get("risk_tier", "")
        score = row.get("final_score", 0)
        flags = row.get("flags", "-")
        lines.append(f"| `{user_id}` | {tier} | {score:.4f} | {flags} |")
    
    lines.append("")
    
    # Flag co-occurrence analysis
    lines.append("### 5.2 Flag Co-occurrence")
    lines.append("")
    lines.append("Users with multiple flags are more likely to be sharing accounts:")
    lines.append("")
    
    flag_cols = [c for c in scored_df.columns if c.startswith("flag_")]
    if flag_cols:
        flag_counts = scored_df[flag_cols].sum(axis=1)
        lines.append("| Number of Flags | Users |")
        lines.append("|-----------------|-------|")
        for n in range(5):
            count = (flag_counts == n).sum()
            lines.append(f"| {n} flags | {format_number(count)} |")
        count_5plus = (flag_counts >= 5).sum()
        if count_5plus > 0:
            lines.append(f"| 5+ flags | {format_number(count_5plus)} |")
    
    lines.append("")
    
    return "\n".join(lines)


def generate_recommendations_section(scored_df: pd.DataFrame, breakdown: dict) -> str:
    """Generate the recommendations section."""
    lines = []
    lines.append("## 6. Recommendations")
    lines.append("")
    
    lines.append("### 6.1 Immediate Actions")
    lines.append("")
    lines.append("1. **Manual Review of High-Risk Accounts**")
    lines.append("   - Review the top high-risk users for verification")
    lines.append("   - Look for corroborating evidence (payment info, support tickets)")
    lines.append("   - Consider reaching out to users before enforcement")
    lines.append("")
    lines.append("2. **Implement Real-time Monitoring**")
    lines.append("   - Flag impossible travel events in real-time")
    lines.append("   - Alert on password resets followed by geographic shifts")
    lines.append("")
    
    lines.append("### 6.2 Policy Interventions")
    lines.append("")
    lines.append("| Intervention | Target Users | Expected Impact |")
    lines.append("|--------------|--------------|-----------------|")
    lines.append("| Device limit (max 3 devices) | High device diversity | Prevents casual sharing |")
    lines.append("| Geographic verification prompts | Impossible travel detected | Catches credential sharing |")
    lines.append("| Post-exam account lockout | Post-success activity | Prevents account handoffs |")
    lines.append("| Concurrent session limits | Multiple simultaneous IPs | Blocks real-time sharing |")
    lines.append("")
    
    lines.append("### 6.3 Additional Data Recommendations")
    lines.append("")
    lines.append("The following additional data would improve detection accuracy:")
    lines.append("")
    lines.append("1. **Device fingerprints** - More reliable than user agent strings")
    lines.append("2. **Payment method changes** - Strong signal for account handoffs")
    lines.append("3. **Support ticket context** - May reveal legitimate account transfers")
    lines.append("4. **Time-of-day patterns** - Different work schedules suggest different users")
    lines.append("")
    
    lines.append("### 6.4 False Positive Considerations")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> Users with legitimate VPN usage or frequent travel may trigger false positives.")
    lines.append("> Consider implementing an appeals process for flagged accounts.")
    lines.append("")
    lines.append("Common false positive scenarios:")
    lines.append("- Business travelers using VPNs")
    lines.append("- Users with multiple personal devices (phone, tablet, laptop)")
    lines.append("- Shared family computers")
    lines.append("- Corporate proxy servers")
    lines.append("")
    
    return "\n".join(lines)


def generate_appendix(features_df: pd.DataFrame) -> str:
    """Generate the technical appendix."""
    lines = []
    lines.append("## Appendix A: Technical Details")
    lines.append("")
    lines.append("### A.1 Risk Score Weights")
    lines.append("")
    lines.append("| Flag | Weight |")
    lines.append("|------|--------|")
    lines.append("| Impossible Travel | 1.5 |")
    lines.append("| Concurrent Activity | 1.2 |")
    lines.append("| Device Diversity | 1.0 |")
    lines.append("| Post-Success Activity | 1.0 |")
    lines.append("| Password Reset Shift | 0.8 |")
    lines.append("")
    
    lines.append("### A.2 Geographic Calculations")
    lines.append("")
    lines.append("- Distance: Haversine formula (great-circle distance)")
    lines.append("- Speed: distance / time between consecutive signIn events")
    lines.append("- Impossible threshold: 500 mph (faster than commercial aircraft)")
    lines.append("")
    
    lines.append("### A.3 Concurrent Activity Detection")
    lines.append("")
    lines.append("- Window size: 10 minutes (sliding window)")
    lines.append("- Algorithm: Two-pointer scan for distinct IP counts")
    lines.append("- Data source: Content activity events (1.35M events)")
    lines.append("")
    
    return "\n".join(lines)


def generate_visualizations_section(viz_paths: list[str]) -> str:
    """Generate a section that embeds visualization PNGs into the report."""
    if not viz_paths:
        return ""

    lines: list[str] = []
    lines.append("## 7. Visualizations")
    lines.append("")
    lines.append("The following figures summarize key aspects of the risk scoring results.")
    lines.append("")

    # Try to give nicer titles based on filenames
    nice_titles = {
        "risk_tier_distribution.png": "Distribution of Users by Risk Tier",
        "score_distribution.png": "Distribution of Final Risk Scores",
        "flag_correlation.png": "Flag Co-occurrence Correlation Heatmap",
    }

    for p in viz_paths:
        name = Path(p).name
        title = nice_titles.get(name, name)
        lines.append(f"### {title}")
        lines.append("")
        # Standard markdown image embed; keep relative path as written to disk
        lines.append(f"![{title}]({p})")
        lines.append("")

    return "\n".join(lines)

def generate_visualizations(scored_df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Generate visualization files and return their paths."""
    if not PLOTTING_AVAILABLE:
        return []
    
    paths = []
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Risk tier distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    tier_counts = scored_df["risk_tier"].value_counts()
    colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}
    tier_order = ["low", "medium", "high"]
    tier_counts = tier_counts.reindex(tier_order)
    bars = ax.bar(tier_counts.index, tier_counts.values, color=[colors.get(t, "#3498db") for t in tier_counts.index])
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Users by Risk Tier")
    for bar, count in zip(bars, tier_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f"{int(count):,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path1 = output_dir / "risk_tier_distribution.png"
    plt.savefig(path1, dpi=150)
    plt.close()
    paths.append(str(path1))
    
    # 2. Final score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    final_scores = pd.to_numeric(scored_df["final_score"], errors="coerce").dropna()
    ax.hist(final_scores, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(scored_df["final_score"].quantile(0.98), color="#e74c3c", linestyle="--", label="98th percentile")
    ax.axvline(scored_df["final_score"].quantile(0.995), color="#c0392b", linestyle="-", label="99.5th percentile")
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Final Risk Scores")
    ax.legend()
    plt.tight_layout()
    path2 = output_dir / "score_distribution.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    paths.append(str(path2))
    
    # 3. Flag heatmap (flag co-occurrence)
    flag_cols = [c for c in scored_df.columns if c.startswith("flag_") and c != "flag_impossible_travel"]
    # Include impossible_travel if it has any true values
    if "flag_impossible_travel" in scored_df.columns and scored_df["flag_impossible_travel"].any():
        flag_cols = ["flag_impossible_travel"] + flag_cols
    
    if len(flag_cols) >= 2:
        flag_data = scored_df[flag_cols].astype(int)
        corr = flag_data.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0, ax=ax,
                   xticklabels=[c.replace("flag_", "") for c in flag_cols],
                   yticklabels=[c.replace("flag_", "") for c in flag_cols])
        ax.set_title("Flag Co-occurrence Correlation")
        plt.tight_layout()
        path3 = output_dir / "flag_correlation.png"
        plt.savefig(path3, dpi=150)
        plt.close()
        paths.append(str(path3))
    
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Week 6 final technical report.")
    parser.add_argument("--week1-summary", default="week1_output/week1_summary.json")
    parser.add_argument("--features-csv", default="week2_3_output/features_per_user.csv")
    parser.add_argument("--scored-csv", default="week4_5_output/scored_users.csv")
    parser.add_argument("--breakdown-json", default="week4_5_output/method_breakdown.json")
    parser.add_argument("--output-dir", default="week6_output")
    parser.add_argument("--generate-plots", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    week1_summary = load_json(Path(args.week1_summary))
    features_df = pd.read_csv(args.features_csv)
    scored_df = pd.read_csv(args.scored_csv)
    breakdown = load_json(Path(args.breakdown_json))
    
    print(f"  Loaded {len(features_df):,} users with features")
    print(f"  Loaded {len(scored_df):,} scored users")
    
    # Generate visualizations if requested
    viz_paths = []
    if args.generate_plots:
        print("Generating visualizations...")
        if PLOTTING_AVAILABLE:
            viz_paths = generate_visualizations(scored_df, output_dir)
            print(f"  Generated {len(viz_paths)} visualization(s)")
        else:
            print("  Warning: matplotlib/seaborn not available, skipping plots")
    
    # Generate report sections
    print("Generating report sections...")
    sections = []
    
    sections.append(generate_executive_summary(scored_df, breakdown))
    sections.append(generate_eda_section(week1_summary))
    sections.append(generate_feature_engineering_section(features_df))
    sections.append(generate_model_section(breakdown, scored_df))
    sections.append(generate_flagged_users_section(scored_df, breakdown))
    sections.append(generate_recommendations_section(scored_df, breakdown))
    # If visualizations were generated, embed them directly in the report
    if viz_paths:
        sections.append(generate_visualizations_section(viz_paths))
    sections.append(generate_appendix(features_df))
    
    # Combine and write report
    report = "\n".join(sections)
    
    report_path = output_dir / "technical_report.md"
    report_path.write_text(report)
    
    # Generate flagged users CSV
    high_risk = scored_df[scored_df["risk_tier"] == "high"].sort_values("final_score", ascending=False)
    high_risk_path = output_dir / "high_risk_users.csv"
    high_risk.to_csv(high_risk_path, index=False)
    
    medium_risk = scored_df[scored_df["risk_tier"] == "medium"].sort_values("final_score", ascending=False)
    medium_risk_path = output_dir / "medium_risk_users.csv"
    medium_risk.to_csv(medium_risk_path, index=False)
    
    print()
    print(f"Generated outputs:")
    print(f"  - Technical report: {report_path}")
    print(f"  - High risk users: {high_risk_path} ({len(high_risk):,} users)")
    print(f"  - Medium risk users: {medium_risk_path} ({len(medium_risk):,} users)")
    for vp in viz_paths:
        print(f"  - Visualization: {vp}")


if __name__ == "__main__":
    main()
