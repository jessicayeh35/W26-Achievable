#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


try:
    from sklearn.ensemble import IsolationForest  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None  # type: ignore
    SKLEARN_AVAILABLE = False


FLAG_COLS = [
    "flag_impossible_travel",
    "flag_concurrent",
    "flag_device_diversity",
    "flag_post_success",
    "flag_reset_shift",
]

FLAG_WEIGHTS = {
    "flag_impossible_travel": 1.5,
    "flag_concurrent": 1.2,
    "flag_device_diversity": 1.0,
    "flag_post_success": 1.0,
    "flag_reset_shift": 0.8,
}


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype("string").str.lower().isin(["true", "1", "t", "yes", "y"])


def robust_zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series([0.0] * len(series), index=series.index)
    return 0.6745 * (series - median) / mad


def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def percentile_norm(series: pd.Series) -> pd.Series:
    # Rank-based normalization to [0,1], robust to heavy tails.
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return s.rank(pct=True, method="average")


def select_model_columns(df: pd.DataFrame) -> list[str]:
    base_candidates = [
        # device/auth
        "auth_events",
        "unique_device_families",
        "unique_os_families",
        "unique_browser_families",
        "device_switches",
        # content/concurrency
        "content_events",
        "content_unique_ips",
        "concurrent_max_distinct_ips",
        "concurrent_event_ratio",
        # password resets / shift
        "password_reset_count",
        "avg_days_since_exam_at_reset",
        "avg_days_since_high_readiness_at_reset",
        "password_resets_after_positive",
        "reset_device_diversity_change",
        # post-success activity
        "post_positive_content_count",
        "post_positive_exam_count",
        # geo (optional)
        "geo_max_speed_mph",
        "geo_impossible_travel_count",
        "geo_max_distance_from_registration",
    ]

    derived = []
    if {"post_positive_content_count", "post_positive_exam_count"}.issubset(df.columns):
        df["post_positive_total_activity"] = (
            pd.to_numeric(df["post_positive_content_count"], errors="coerce").fillna(0)
            + pd.to_numeric(df["post_positive_exam_count"], errors="coerce").fillna(0)
        )
        derived.append("post_positive_total_activity")

    # Indicators to prevent “0-filled missing” ambiguity
    indicators = []
    if "password_reset_count" in df.columns:
        df["has_reset"] = pd.to_numeric(df["password_reset_count"], errors="coerce").fillna(0) > 0
        indicators.append("has_reset")
    if "positive_at" in df.columns:
        df["has_positive"] = pd.to_datetime(df["positive_at"], errors="coerce").notna()
        indicators.append("has_positive")
    if "geo_max_speed_mph" in df.columns:
        df["has_geo"] = pd.to_numeric(df["geo_max_speed_mph"], errors="coerce").notna()
        indicators.append("has_geo")

    cols = [c for c in base_candidates if c in df.columns]
    cols += derived
    cols += indicators
    return cols


def build_design_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = df[cols].copy()

    # Convert boolean indicators to int
    for c in cols:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)

    # Fill NaNs with 0 for count-like columns; keep indicators.
    fill0_prefixes = (
        "auth_",
        "unique_",
        "device_",
        "content_",
        "concurrent_",
        "password_",
        "reset_",
        "post_positive_",
        "geo_",
    )
    for c in cols:
        if x[c].dtype == object:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        if any(c.startswith(p) for p in fill0_prefixes):
            x[c] = x[c].fillna(0)
        else:
            x[c] = x[c].fillna(0)

    health = {
        "model_columns": cols,
        "n_rows": len(x),
        "n_cols": len(cols),
        "missing_after_fill": int(x.isna().sum().sum()),
    }
    return x, health


def anomaly_score_robust_z(x: pd.DataFrame) -> pd.Series:
    z = pd.DataFrame({c: robust_zscore(x[c]) for c in x.columns})
    score = z.abs().sum(axis=1)
    return score


def anomaly_score_isolation_forest(x: pd.DataFrame, random_state: int, n_estimators: int) -> tuple[pd.Series, dict[str, Any]]:
    if not SKLEARN_AVAILABLE or IsolationForest is None:
        raise RuntimeError("scikit-learn is not available; cannot run IsolationForest.")

    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x.values)
    # score_samples: higher = less anomalous, so invert.
    raw = -pd.Series(model.score_samples(x.values), index=x.index)
    meta = {
        "model": "isolation_forest",
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    return raw, meta


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series([0.0] * len(df), index=df.index)
    for col, w in FLAG_WEIGHTS.items():
        if col in df.columns:
            score += to_bool_series(df[col]).astype(int) * float(w)
    return score


def determine_threshold_by_count(scores: pd.Series, top_n: int) -> float:
    if top_n <= 0:
        return float("inf")
    top_n = min(top_n, len(scores))
    return float(scores.sort_values(ascending=False).iloc[top_n - 1])


def determine_threshold_by_percentile(scores: pd.Series, percentile: float) -> float:
    p = percentile / 100.0
    p = min(max(p, 0.0), 1.0)
    return float(scores.quantile(p))


def flags_to_string(row: pd.Series) -> str:
    parts = []
    for c in FLAG_COLS:
        if c in row.index and bool(row[c]):
            parts.append(c.replace("flag_", ""))
    return ",".join(parts) if parts else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4–5: anomaly model + ensemble scoring + thresholds.")
    parser.add_argument(
        "--features-csv",
        default="week2_3_output/features_per_user.csv",
        help="Input features CSV from week2_3_features.py",
    )
    parser.add_argument("--output-dir", default="week4_5_output", help="Directory to write outputs.")
    parser.add_argument(
        "--model",
        choices=["auto", "isolation_forest", "robust_z"],
        default="auto",
        help="Anomaly model to use.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--w-risk", type=float, default=0.70, help="Weight for heuristic risk score.")
    parser.add_argument("--w-model", type=float, default=0.30, help="Weight for anomaly model score.")

    parser.add_argument("--high-count", type=int, default=0, help="If set, high-risk = top N users.")
    parser.add_argument("--medium-count", type=int, default=0, help="If set, medium-risk = top N users (includes high).")
    parser.add_argument("--high-percentile", type=float, default=99.5, help="High-risk percentile cutoff (0-100).")
    parser.add_argument("--medium-percentile", type=float, default=98.0, help="Medium-risk percentile cutoff (0-100).")

    parser.add_argument("--review-top", type=int, default=50, help="How many high-risk users to include in review sample.")
    parser.add_argument("--review-medium", type=int, default=50, help="How many medium-risk users to include in review sample.")
    parser.add_argument("--review-low", type=int, default=50, help="How many low-risk users to include in review sample.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    if "user_id" not in df.columns:
        raise ValueError("features CSV must include user_id")

    # Normalize flag columns to bool
    for c in FLAG_COLS:
        if c in df.columns:
            df[c] = to_bool_series(df[c])
        else:
            df[c] = False

    risk_score = compute_risk_score(df)
    max_risk = float(sum(FLAG_WEIGHTS.values()))
    risk_norm = risk_score / max_risk if max_risk else 0.0

    cols = select_model_columns(df)
    x, x_health = build_design_matrix(df, cols)

    model_choice = args.model
    model_meta: dict[str, Any] = {"model": "robust_z"}  # default
    if model_choice == "auto":
        model_choice = "isolation_forest" if SKLEARN_AVAILABLE else "robust_z"

    if model_choice == "isolation_forest":
        raw_model_score, meta = anomaly_score_isolation_forest(x, args.random_state, args.n_estimators)
        model_meta = meta
        model_norm = percentile_norm(raw_model_score)
    else:
        raw_model_score = anomaly_score_robust_z(x)
        model_meta = {"model": "robust_z"}
        model_norm = percentile_norm(raw_model_score)

    w_risk = float(args.w_risk)
    w_model = float(args.w_model)
    if w_risk < 0 or w_model < 0 or (w_risk + w_model) == 0:
        raise ValueError("w-risk and w-model must be non-negative and not both zero.")
    w_sum = w_risk + w_model
    w_risk /= w_sum
    w_model /= w_sum

    final_score = w_risk * risk_norm + w_model * model_norm

    scored = df.copy()
    scored["risk_score"] = risk_score
    scored["risk_score_norm"] = risk_norm
    scored["model_score_raw"] = raw_model_score
    scored["model_score_norm"] = model_norm
    scored["final_score"] = final_score

    # Thresholds
    if args.high_count > 0:
        high_thresh = determine_threshold_by_count(scored["final_score"], args.high_count)
    else:
        high_thresh = determine_threshold_by_percentile(scored["final_score"], args.high_percentile)

    if args.medium_count > 0:
        medium_thresh = determine_threshold_by_count(scored["final_score"], args.medium_count)
    else:
        medium_thresh = determine_threshold_by_percentile(scored["final_score"], args.medium_percentile)

    # Ensure ordering: high >= medium
    medium_thresh = min(medium_thresh, high_thresh)

    def _tier(s: float) -> str:
        if s >= high_thresh:
            return "high"
        if s >= medium_thresh:
            return "medium"
        return "low"

    scored["risk_tier"] = scored["final_score"].apply(_tier)
    scored["flags"] = scored.apply(flags_to_string, axis=1)
    scored["any_flag"] = scored[[c for c in FLAG_COLS if c in scored.columns]].any(axis=1)

    # Outputs
    scored_out = out / "scored_users.csv"
    scored.to_csv(scored_out, index=False)

    # Manual review sample
    rng = np.random.default_rng(args.random_state)
    high = scored[scored["risk_tier"] == "high"].sort_values("final_score", ascending=False)
    medium = scored[scored["risk_tier"] == "medium"].sort_values("final_score", ascending=False)
    low = scored[scored["risk_tier"] == "low"]

    sample_parts = []
    if not high.empty and args.review_top > 0:
        sample_parts.append(high.head(args.review_top))
    if not medium.empty and args.review_medium > 0:
        sample_parts.append(medium.sample(n=min(args.review_medium, len(medium)), random_state=args.random_state))
    if not low.empty and args.review_low > 0:
        sample_parts.append(low.sample(n=min(args.review_low, len(low)), random_state=args.random_state))

    if sample_parts:
        review = pd.concat(sample_parts, ignore_index=True)
        keep_cols = [
            "user_id",
            "risk_tier",
            "final_score",
            "risk_score",
            "model_score_norm",
            "flags",
            "auth_events",
            "unique_device_families",
            "device_switches",
            "content_events",
            "content_unique_ips",
            "concurrent_max_distinct_ips",
            "concurrent_event_ratio",
            "password_reset_count",
            "password_resets_after_positive",
            "reset_device_diversity_change",
            "post_positive_content_count",
            "post_positive_exam_count",
        ]
        keep_cols = [c for c in keep_cols if c in review.columns]
        review = review[keep_cols]
        review.to_csv(out / "manual_review_sample.csv", index=False)

    # Method breakdown
    breakdown = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_meta": model_meta,
        "weights": {"risk": w_risk, "model": w_model},
        "thresholds": {"high": high_thresh, "medium": medium_thresh},
        "counts": scored["risk_tier"].value_counts().to_dict(),
        "flag_counts": {c: int(scored[c].sum()) for c in FLAG_COLS if c in scored.columns},
    }
    (out / "method_breakdown.json").write_text(json.dumps(breakdown, indent=2, default=str))

    # Markdown report
    report_lines: list[str] = []
    report_lines.append("# Week 4–5 Modeling Report")
    report_lines.append("")
    report_lines.append(f"Generated: {breakdown['generated_at']}")
    report_lines.append(f"Features input: `{args.features_csv}`")
    report_lines.append(f"Model: `{model_meta.get('model')}` (sklearn available: {SKLEARN_AVAILABLE})")
    report_lines.append("")
    report_lines.append("## Thresholds")
    report_lines.append("")
    report_lines.append(f"- High tier threshold: **{high_thresh:.4f}**")
    report_lines.append(f"- Medium tier threshold: **{medium_thresh:.4f}**")
    report_lines.append("")
    report_lines.append("## Counts by tier")
    report_lines.append("")
    for tier, cnt in breakdown["counts"].items():
        report_lines.append(f"- **{tier}**: {cnt:,}")
    report_lines.append("")
    report_lines.append("## Flag counts")
    report_lines.append("")
    for c, cnt in breakdown["flag_counts"].items():
        report_lines.append(f"- **{c}**: {cnt:,}")
    report_lines.append("")
    top10 = scored.sort_values("final_score", ascending=False).head(10)
    report_lines.append("## Top 10 users by final score")
    report_lines.append("")
    report_lines.append("| user_id | tier | final_score | risk_score | model_score_norm | flags |")
    report_lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in top10.iterrows():
        report_lines.append(
            f"| {row['user_id']} | {row['risk_tier']} | {row['final_score']:.4f} | {row['risk_score']:.2f} | {row['model_score_norm']:.3f} | {row['flags']} |"
        )
    (out / "week4_5_report.md").write_text("\n".join(report_lines))

    print(f"Wrote scored users: {scored_out}")
    print(f"Wrote report: {out / 'week4_5_report.md'}")
    print(f"Wrote breakdown: {out / 'method_breakdown.json'}")
    if (out / "manual_review_sample.csv").exists():
        print(f"Wrote manual review sample: {out / 'manual_review_sample.csv'}")


if __name__ == "__main__":
    main()

