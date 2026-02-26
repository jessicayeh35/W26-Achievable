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

NULL_VALUES = ["NULL", "null", ""]
IPV4_RE = r"^(?:\d{1,3}\.){3}\d{1,3}$"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FMT = "%Y-%m-%d"


@dataclass
class DatasetConfig:
    name: str
    filename: str
    columns: list[str]
    date_cols: list[str]
    date_formats: dict[str, str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    ip_col: str | None = None


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="auth_events",
        filename="auth_events.csv",
        columns=["user_id", "event_type", "event_timestamp", "ip_address", "user_agent"],
        date_cols=["event_timestamp"],
        date_formats={"event_timestamp": DATETIME_FMT},
        numeric_cols=[],
        categorical_cols=["event_type"],
        ip_col="ip_address",
    ),
    DatasetConfig(
        name="content_activity",
        filename="content_activity.csv",
        columns=["user_id", "activity_timestamp", "ip_address", "user_agent"],
        date_cols=["activity_timestamp"],
        date_formats={"activity_timestamp": DATETIME_FMT},
        numeric_cols=[],
        categorical_cols=[],
        ip_col="ip_address",
    ),
    DatasetConfig(
        name="exam_sessions",
        filename="exam_sessions.csv",
        columns=["user_id", "exam_session_id", "started_at", "finished_at", "score", "status"],
        date_cols=["started_at", "finished_at"],
        date_formats={"started_at": DATETIME_FMT, "finished_at": DATETIME_FMT},
        numeric_cols=["score"],
        categorical_cols=["status"],
    ),
    DatasetConfig(
        name="subscriptions",
        filename="subscriptions.csv",
        columns=["user_id", "subscription_start", "subscription_end", "is_active"],
        date_cols=["subscription_start", "subscription_end"],
        date_formats={"subscription_start": DATE_FMT, "subscription_end": DATE_FMT},
        numeric_cols=["is_active"],
        categorical_cols=[],
    ),
    DatasetConfig(
        name="user_metadata",
        filename="user_metadata.csv",
        columns=[
            "user_id",
            "registration_country",
            "registration_region",
            "registration_lat",
            "registration_lng",
            "timezone",
            "registered_at",
            "first_seen_at",
            "acquisition_source",
        ],
        date_cols=["registered_at", "first_seen_at"],
        date_formats={"registered_at": DATETIME_FMT, "first_seen_at": DATETIME_FMT},
        numeric_cols=["registration_lat", "registration_lng"],
        categorical_cols=[
            "registration_country",
            "registration_region",
            "timezone",
            "acquisition_source",
        ],
    ),
    DatasetConfig(
        name="study_plans",
        filename="study_plans.csv",
        columns=[
            "user_id",
            "target_exam_date",
            "readiness_score",
            "study_score",
            "status",
            "created_at",
            "updated_at",
        ],
        date_cols=["target_exam_date", "created_at", "updated_at"],
        date_formats={
            "target_exam_date": DATE_FMT,
            "created_at": DATETIME_FMT,
            "updated_at": DATETIME_FMT,
        },
        numeric_cols=["readiness_score", "study_score"],
        categorical_cols=["status"],
    ),
]


def _read_fixed_width_csv(path: Path, expected_columns: list[str], sample_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Robust reader for exports that look like CSV but may contain unquoted commas
    in the LAST column (e.g., user_agent).

    Strategy: split each line on the first (N-1) commas so any extra commas end
    up in the last field, avoiding silent column shifts.
    """
    expected_n = len(expected_columns)
    expected_commas = expected_n - 1

    parsing: dict[str, Any] = {
        "expected_columns": expected_columns,
        "header_columns": None,
        "schema_match": None,
        "rows_parsed": 0,
        "rows_skipped_too_few_fields": 0,
        "rows_repaired_extra_commas": 0,
        "rows_with_trailing_extra_field": 0,
        "skipped_line_numbers_sample": [],
        "repaired_line_numbers_sample": [],
        "sample_rows": sample_rows,
        "sample_method": "head" if sample_rows else None,
    }

    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Empty file: {path}")
        header = header_line.rstrip("\r\n").split(",")
        parsing["header_columns"] = header
        if header != expected_columns:
            raise ValueError(
                "Schema mismatch for "
                f"{path}.\nExpected header:\n  {expected_columns}\nGot header:\n  {header}\n"
                "This usually means the export format changed."
            )
        parsing["schema_match"] = True

        for line_num, line in enumerate(f, start=2):
            if sample_rows is not None and parsing["rows_parsed"] >= sample_rows:
                break

            raw = line.rstrip("\r\n")
            if not raw:
                continue

            comma_count = raw.count(",")
            if comma_count < expected_commas:
                parsing["rows_skipped_too_few_fields"] += 1
                if len(parsing["skipped_line_numbers_sample"]) < 25:
                    parsing["skipped_line_numbers_sample"].append(line_num)
                continue

            if comma_count > expected_commas:
                parsing["rows_repaired_extra_commas"] += 1
                if len(parsing["repaired_line_numbers_sample"]) < 25:
                    parsing["repaired_line_numbers_sample"].append(line_num)

            parts = raw.split(",", expected_commas)
            if len(parts) != expected_n:
                parsing["rows_skipped_too_few_fields"] += 1
                if len(parsing["skipped_line_numbers_sample"]) < 25:
                    parsing["skipped_line_numbers_sample"].append(line_num)
                continue

            # Handle a common nuisance: an extra trailing comma creating an empty extra field.
            if comma_count == expected_n and raw.endswith(","):
                parts[-1] = parts[-1].rstrip(",")
                parsing["rows_with_trailing_extra_field"] += 1

            rows.append(parts)
            parsing["rows_parsed"] += 1

    df = pd.DataFrame(rows, columns=expected_columns)
    return df, parsing


def load_dataset(path: Path, config: DatasetConfig, sample_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, parsing = _read_fixed_width_csv(path, config.columns, sample_rows)

    # Normalize nulls (including empty strings) to NaN
    df = df.replace(NULL_VALUES, np.nan)

    numeric_health: dict[str, Any] = {}
    for col in config.numeric_cols:
        if col not in df.columns:
            continue
        raw = df[col]
        raw_nonnull = int(raw.notna().sum())
        parsed = pd.to_numeric(raw, errors="coerce")
        parsed_nonnull = int(parsed.notna().sum())
        numeric_health[col] = {
            "raw_nonnull": raw_nonnull,
            "parsed_nonnull": parsed_nonnull,
            "parse_failures": raw_nonnull - parsed_nonnull,
        }
        df[col] = parsed

    datetime_health: dict[str, Any] = {}
    for col in config.date_cols:
        if col not in df.columns:
            continue
        fmt = config.date_formats.get(col)
        raw = df[col]
        raw_nonnull = int(raw.notna().sum())
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        parsed_nonnull = int(parsed.notna().sum())
        datetime_health[col] = {
            "format": fmt,
            "raw_nonnull": raw_nonnull,
            "parsed_nonnull": parsed_nonnull,
            "parse_failures": raw_nonnull - parsed_nonnull,
            "parsed_pct_of_rows": float(parsed_nonnull / len(df)) if len(df) else 0.0,
        }
        df[col] = parsed

    parsing["numeric_parsing"] = numeric_health
    parsing["datetime_parsing"] = datetime_health
    parsing["actual_columns"] = list(df.columns)
    parsing["actual_column_count"] = len(df.columns)
    parsing["expected_column_count"] = len(config.columns)

    return df, parsing


def split_ip_addresses(series: pd.Series) -> dict[str, pd.Series]:
    if series is None:
        return {"public": pd.Series(dtype="string"), "private": pd.Series(dtype="string")}

    has_pipe = series.astype("string").str.contains("|", regex=False)
    parts = series.astype("string").str.split("|", n=1, expand=True)

    public = parts[0].where(series.notna(), np.nan)
    if parts.shape[1] > 1:
        private = parts[1].where(has_pipe & series.notna(), np.nan)
    else:
        private = pd.Series([np.nan] * len(series), index=series.index)

    return {"public": public, "private": private, "has_pipe": has_pipe}


def basic_profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    missing = df.isna().sum()
    missing_pct = (missing / row_count).replace([np.inf, -np.inf], np.nan).fillna(0) if row_count else missing * 0
    duplicates = int(df.duplicated().sum())

    return {
        "rows": row_count,
        "columns": len(df.columns),
        "duplicate_rows": duplicates,
        "missingness": {
            col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
            for col in df.columns
        },
    }


def date_ranges(df: pd.DataFrame, date_cols: list[str]) -> dict[str, dict[str, Any]]:
    ranges = {}
    for col in date_cols:
        if col not in df.columns:
            continue
        series = df[col]
        ranges[col] = {
            "min": series.min(),
            "max": series.max(),
            "missing": int(series.isna().sum()),
        }
    return ranges


def categorical_tops(df: pd.DataFrame, categorical_cols: list[str], n: int = 8) -> dict[str, list[dict[str, Any]]]:
    tops: dict[str, list[dict[str, Any]]] = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=True).head(n)
        tops[col] = [{"value": str(idx), "count": int(count)} for idx, count in counts.items()]
    return tops


def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            summary[col] = {"min": None, "max": None, "mean": None, "median": None}
        else:
            summary[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
            }
    return summary


def user_event_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "user_id" not in df.columns:
        return {}

    counts = df["user_id"].value_counts(dropna=True)
    if counts.empty:
        return {}

    return {
        "unique_users": int(counts.shape[0]),
        "mean_events": float(counts.mean()),
        "median_events": float(counts.median()),
        "p95_events": float(counts.quantile(0.95)),
        "max_events": int(counts.max()),
    }


def ip_stats(df: pd.DataFrame, ip_col: str) -> dict[str, Any]:
    if ip_col not in df.columns:
        return {}

    parts = split_ip_addresses(df[ip_col])
    public = parts["public"]
    private = parts["private"]
    has_pipe = parts["has_pipe"]

    public_invalid = public.notna() & ~public.astype("string").str.match(IPV4_RE)
    private_invalid = private.notna() & ~private.astype("string").str.match(IPV4_RE)

    return {
        "total": int(df[ip_col].notna().sum()),
        "has_pipe": int(has_pipe.sum()),
        "public_invalid": int(public_invalid.sum()),
        "private_invalid": int(private_invalid.sum()),
    }


def dataset_checks(name: str, df: pd.DataFrame) -> list[str]:
    checks: list[str] = []

    if name == "auth_events":
        expected = {"signIn", "signInFailed", "signOut", "forgotPassword", "resetPassword"}
        if "event_type" in df.columns:
            invalid = df["event_type"].dropna()
            invalid = invalid[~invalid.isin(expected)]
            if not invalid.empty:
                checks.append(f"Unexpected event_type values: {invalid.unique()[:5].tolist()}")

    if name == "exam_sessions":
        if "started_at" in df.columns and "finished_at" in df.columns:
            mask = df["started_at"].notna() & df["finished_at"].notna()
            durations = (df.loc[mask, "finished_at"] - df.loc[mask, "started_at"]).dt.total_seconds() / 60
            if durations.notna().any():
                negative = durations[durations < 0]
                if not negative.empty:
                    checks.append(f"Negative exam durations (only where finished_at present): {int(negative.shape[0])}")
                long = durations[durations > 300]
                if not long.empty:
                    checks.append(f"Very long exam durations >300 min (only where finished_at present): {int(long.shape[0])}")

        if "score" in df.columns:
            score = pd.to_numeric(df["score"], errors="coerce")
            max_score = score.max()
            if pd.notna(max_score):
                if max_score <= 1.0:
                    checks.append("Scores appear to be proportions (0-1).")
                else:
                    checks.append("Scores appear to be percentages (0-100).")

    if name == "study_plans":
        if "created_at" in df.columns and "updated_at" in df.columns:
            invalid = df["updated_at"] < df["created_at"]
            if invalid.any():
                checks.append(f"updated_at < created_at: {int(invalid.sum())}")

        for col in ["readiness_score", "study_score"]:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                out_of_range = series[(series < 0) | (series > 1)]
                if not out_of_range.empty:
                    checks.append(f"{col} out of [0, 1]: {int(out_of_range.shape[0])}")
        if "target_exam_date" in df.columns:
            extreme = df["target_exam_date"].notna() & (df["target_exam_date"] > pd.Timestamp("2035-12-31"))
            if extreme.any():
                checks.append(f"target_exam_date appears extreme (>2035-12-31): {int(extreme.sum())}")

    if name == "subscriptions":
        if "subscription_start" in df.columns and "subscription_end" in df.columns:
            invalid = df["subscription_end"] < df["subscription_start"]
            if invalid.any():
                checks.append(f"subscription_end < subscription_start: {int(invalid.sum())}")
        if "is_active" in df.columns:
            invalid = ~df["is_active"].isin([0, 1])
            if invalid.any():
                checks.append(f"is_active not in {{0,1}}: {int(invalid.sum())}")

    if name == "user_metadata":
        if "registration_lat" in df.columns:
            invalid = (df["registration_lat"] < -90) | (df["registration_lat"] > 90)
            if invalid.any():
                checks.append(f"registration_lat outside [-90, 90]: {int(invalid.sum())}")
        if "registration_lng" in df.columns:
            invalid = (df["registration_lng"] < -180) | (df["registration_lng"] > 180)
            if invalid.any():
                checks.append(f"registration_lng outside [-180, 180]: {int(invalid.sum())}")
        if "first_seen_at" in df.columns and "registered_at" in df.columns:
            invalid = df["first_seen_at"] < df["registered_at"]
            if invalid.any():
                checks.append(f"first_seen_at < registered_at: {int(invalid.sum())}")

    # Generic sanity check: if user_id exists, unique users should be plausible.
    if "user_id" in df.columns:
        unique_users = int(df["user_id"].nunique(dropna=True))
        if len(df) >= 10_000 and unique_users < 100:
            checks.append(
                f"Unique users implausibly low ({unique_users} for {len(df):,} rows) — likely parsing/schema issue."
            )

    return checks


def dataset_metrics(name: str, df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if name == "exam_sessions" and {"started_at", "finished_at"}.issubset(df.columns):
        mask = df["started_at"].notna() & df["finished_at"].notna()
        durations = (df.loc[mask, "finished_at"] - df.loc[mask, "started_at"]).dt.total_seconds() / 60
        durations = durations.dropna()
        if not durations.empty:
            metrics["exam_duration_minutes"] = {
                "n_with_finished_at": int(mask.sum()),
                "p50": float(durations.quantile(0.50)),
                "p95": float(durations.quantile(0.95)),
                "p99": float(durations.quantile(0.99)),
                "negative": int((durations < 0).sum()),
                "gt_300": int((durations > 300).sum()),
                "gt_1440": int((durations > 1440).sum()),
            }
            if "status" in df.columns:
                finished_mask = mask & (df["status"] == "finished")
                finished_dur = (
                    (df.loc[finished_mask, "finished_at"] - df.loc[finished_mask, "started_at"]).dt.total_seconds() / 60
                ).dropna()
                if not finished_dur.empty:
                    metrics["exam_duration_minutes"]["finished_only_p50"] = float(finished_dur.quantile(0.50))
                    metrics["exam_duration_minutes"]["finished_only_p95"] = float(finished_dur.quantile(0.95))
                    metrics["exam_duration_minutes"]["finished_only_p99"] = float(finished_dur.quantile(0.99))

    return metrics


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_datetime(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def build_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 1 EDA Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append(f"Data directory: `{summary['data_dir']}`")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Some exports contain **unquoted commas** inside the last column (notably `user_agent`).")
    lines.append("- This runner repairs those rows by splitting on the first (N-1) commas and treating the remainder as the last field.")
    lines.append("")

    overview_rows = []
    for name, info in summary["datasets"].items():
        profile = info["profile"]
        users = info.get("user_event_stats", {})
        overview_rows.append(
            [
                name,
                f"{profile['rows']:,}",
                f"{profile['columns']}",
                f"{users.get('unique_users', 0):,}",
                f"{profile['duplicate_rows']:,}",
            ]
        )

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(
        markdown_table(
            ["dataset", "rows", "columns", "unique users", "duplicate rows"],
            overview_rows,
        )
    )
    lines.append("")

    if summary.get("user_coverage"):
        coverage_rows = []
        for row in summary["user_coverage"]:
            coverage_rows.append(
                [
                    row["dataset"],
                    f"{row['unique_users']:,}",
                    f"{row['missing_from_metadata']:,}",
                    f"{row['missing_from_subscriptions']:,}",
                ]
            )
        lines.append("## User Coverage")
        lines.append("")
        lines.append(
            markdown_table(
                ["dataset", "unique users", "missing metadata", "missing subscriptions"],
                coverage_rows,
            )
        )
        lines.append("")

    lines.append("## Dataset Details")
    lines.append("")

    for name, info in summary["datasets"].items():
        lines.append(f"### {name}")
        lines.append("")

        parsing = info.get("parsing_health", {})
        if parsing:
            lines.append("**Parsing health**")
            lines.append("")
            ph_rows = [
                ["expected columns", str(parsing.get("expected_column_count", "-"))],
                ["actual columns", str(parsing.get("actual_column_count", "-"))],
                ["header matches expected", str(bool(parsing.get("schema_match")))],
                ["rows parsed", str(parsing.get("rows_parsed", "-"))],
                ["rows skipped (too few fields)", str(parsing.get("rows_skipped_too_few_fields", "-"))],
                ["rows repaired (extra commas merged into last field)", str(parsing.get("rows_repaired_extra_commas", "-"))],
            ]
            if parsing.get("rows_with_trailing_extra_field", 0):
                ph_rows.append(["rows w/ trailing extra field", str(parsing["rows_with_trailing_extra_field"])])
            lines.append(markdown_table(["metric", "value"], ph_rows))
            lines.append("")

            dt_health = parsing.get("datetime_parsing", {})
            if dt_health:
                dt_rows = []
                for col, vals in dt_health.items():
                    dt_rows.append(
                        [
                            col,
                            str(vals.get("format", "-")),
                            f"{vals.get('parsed_pct_of_rows', 0.0):.2%}",
                            str(vals.get("parse_failures", 0)),
                        ]
                    )
                lines.append("_Datetime parsing_")
                lines.append("")
                lines.append(markdown_table(["column", "format", "parsed % of rows", "parse failures"], dt_rows))
                lines.append("")

            if parsing.get("rows_skipped_too_few_fields", 0):
                lines.append(
                    f"- Skipped line numbers (sample): {parsing.get('skipped_line_numbers_sample', [])}"
                )
                lines.append("")
            # Only show repaired examples when it's not essentially the entire dataset.
            if 0 < parsing.get("rows_repaired_extra_commas", 0) < 5000:
                lines.append(
                    f"- Repaired line numbers (sample): {parsing.get('repaired_line_numbers_sample', [])}"
                )
                lines.append("")

        ranges = info.get("date_ranges", {})
        if ranges:
            range_rows = [
                [col, format_datetime(vals["min"]), format_datetime(vals["max"]), str(vals["missing"])]
                for col, vals in ranges.items()
            ]
            lines.append("**Date ranges**")
            lines.append("")
            lines.append(markdown_table(["column", "min", "max", "missing"], range_rows))
            lines.append("")

        metrics = info.get("metrics", {})
        if name == "exam_sessions" and metrics.get("exam_duration_minutes"):
            dur = metrics["exam_duration_minutes"]
            lines.append("**Exam duration (minutes) — only where finished_at is present**")
            lines.append("")
            dur_rows = [
                ["n with finished_at", str(dur.get("n_with_finished_at", "-"))],
                ["p50", f"{dur.get('p50', 0):.2f}"],
                ["p95", f"{dur.get('p95', 0):.2f}"],
                ["p99", f"{dur.get('p99', 0):.2f}"],
                ["negative durations", str(dur.get("negative", "-"))],
                ["> 300 min", str(dur.get("gt_300", "-"))],
                ["> 1440 min (24h)", str(dur.get("gt_1440", "-"))],
            ]
            if "finished_only_p50" in dur:
                dur_rows.extend(
                    [
                        ["finished-only p50", f"{dur.get('finished_only_p50', 0):.2f}"],
                        ["finished-only p95", f"{dur.get('finished_only_p95', 0):.2f}"],
                        ["finished-only p99", f"{dur.get('finished_only_p99', 0):.2f}"],
                    ]
                )
            lines.append(markdown_table(["metric", "value"], dur_rows))
            lines.append("")

        missingness = info["profile"]["missingness"]
        missing_rows = sorted(
            missingness.items(), key=lambda item: item[1]["count"], reverse=True
        )[:8]
        missing_table = [
            [col, str(vals["count"]), f"{vals['pct']:.2%}"] for col, vals in missing_rows
        ]
        lines.append("**Top missing columns**")
        lines.append("")
        lines.append(markdown_table(["column", "missing", "pct"], missing_table))
        lines.append("")

        categorical = info.get("categorical_tops", {})
        if categorical:
            lines.append("**Top categorical values**")
            lines.append("")
            for col, entries in categorical.items():
                rows = [[entry["value"], str(entry["count"])] for entry in entries]
                lines.append(f"_{col}_")
                lines.append("")
                lines.append(markdown_table(["value", "count"], rows))
                lines.append("")

        ip_info = info.get("ip_stats")
        if ip_info:
            rows = [
                ["total ip rows", str(ip_info["total"])],
                ["has public|private", str(ip_info["has_pipe"])],
                ["public invalid", str(ip_info["public_invalid"])],
                ["private invalid", str(ip_info["private_invalid"])],
            ]
            lines.append("**IP parsing checks**")
            lines.append("")
            lines.append(markdown_table(["metric", "value"], rows))
            lines.append("")

        checks = info.get("checks", [])
        if checks:
            lines.append("**Potential data quality flags**")
            lines.append("")
            for check in checks:
                lines.append(f"- {check}")
            lines.append("")

        # Add lightweight, actionable notes for a few common anomalies.
        notes: list[str] = []
        if name == "user_metadata" and any("first_seen_at < registered_at" in c for c in checks):
            notes.append("Consider treating `first_seen_at` as unreliable (or clamp it to `registered_at`) before time-based analyses.")
        if name == "subscriptions" and any("subscription_end < subscription_start" in c for c in checks):
            notes.append("Rows where `subscription_end < subscription_start` should be excluded or corrected upstream.")
        if name == "study_plans" and any("target_exam_date appears extreme" in c for c in checks):
            notes.append("Extreme `target_exam_date` values (e.g. far-future years) are likely invalid placeholders; consider filtering beyond a cutoff (e.g. 2035).")
        if notes:
            lines.append("**Notes / suggested resolution**")
            lines.append("")
            for n in notes:
                lines.append(f"- {n}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 EDA and data quality checks.")
    parser.add_argument(
        "--data-dir",
        default="account-sharing-export",
        help="Directory containing CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        default="week1_output",
        help="Directory to write reports.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional per-file sample size for faster runs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "datasets": {},
    }

    user_sets: dict[str, set[str]] = {}

    for config in DATASETS:
        path = data_dir / config.filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        df, parsing = load_dataset(path, config, args.sample_rows)
        profile = basic_profile(df)
        ranges = date_ranges(df, config.date_cols)
        cats = categorical_tops(df, config.categorical_cols)
        nums = numeric_summary(df, config.numeric_cols)
        users = user_event_stats(df)
        ip_info = ip_stats(df, config.ip_col) if config.ip_col else {}
        checks = dataset_checks(config.name, df)
        metrics = dataset_metrics(config.name, df)

        if "user_id" in df.columns:
            user_sets[config.name] = set(df["user_id"].dropna().astype(str).unique())

        summary["datasets"][config.name] = {
            "parsing_health": parsing,
            "profile": profile,
            "date_ranges": ranges,
            "categorical_tops": cats,
            "numeric_summary": nums,
            "user_event_stats": users,
            "ip_stats": ip_info,
            "checks": checks,
            "metrics": metrics,
        }

    if user_sets:
        metadata_users = user_sets.get("user_metadata", set())
        subscription_users = user_sets.get("subscriptions", set())
        coverage = []
        for name, users in user_sets.items():
            coverage.append(
                {
                    "dataset": name,
                    "unique_users": len(users),
                    "missing_from_metadata": len(users - metadata_users),
                    "missing_from_subscriptions": len(users - subscription_users),
                }
            )
        summary["user_coverage"] = coverage

    report_text = build_report(summary)
    report_path = output_dir / "week1_report.md"
    report_path.write_text(report_text)

    json_path = output_dir / "week1_summary.json"
    json_path.write_text(json.dumps(summary, default=str, indent=2))

    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {json_path}")


if __name__ == "__main__":
    main()
