#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from urllib.request import urlopen

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


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="auth_events",
        filename="auth_events.csv",
        columns=["user_id", "event_type", "event_timestamp", "ip_address", "user_agent"],
        date_cols=["event_timestamp"],
        date_formats={"event_timestamp": DATETIME_FMT},
        numeric_cols=[],
    ),
    DatasetConfig(
        name="content_activity",
        filename="content_activity.csv",
        columns=["user_id", "activity_timestamp", "ip_address", "user_agent"],
        date_cols=["activity_timestamp"],
        date_formats={"activity_timestamp": DATETIME_FMT},
        numeric_cols=[],
    ),
    DatasetConfig(
        name="exam_sessions",
        filename="exam_sessions.csv",
        columns=["user_id", "exam_session_id", "started_at", "finished_at", "score", "status"],
        date_cols=["started_at", "finished_at"],
        date_formats={"started_at": DATETIME_FMT, "finished_at": DATETIME_FMT},
        numeric_cols=["score"],
    ),
    DatasetConfig(
        name="subscriptions",
        filename="subscriptions.csv",
        columns=["user_id", "subscription_start", "subscription_end", "is_active"],
        date_cols=["subscription_start", "subscription_end"],
        date_formats={"subscription_start": DATE_FMT, "subscription_end": DATE_FMT},
        numeric_cols=["is_active"],
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
    ),
]


def _read_fixed_width_csv(
    path: Path, expected_columns: list[str], sample_rows: int | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_n = len(expected_columns)
    expected_commas = expected_n - 1

    parsing: dict[str, Any] = {
        "expected_columns": expected_columns,
        "header_columns": None,
        "schema_match": None,
        "rows_parsed": 0,
        "rows_skipped_too_few_fields": 0,
        "rows_repaired_extra_commas": 0,
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

            rows.append(parts)
            parsing["rows_parsed"] += 1

    df = pd.DataFrame(rows, columns=expected_columns)
    return df, parsing


def load_dataset(path: Path, config: DatasetConfig, sample_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, parsing = _read_fixed_width_csv(path, config.columns, sample_rows)

    df = df.replace(NULL_VALUES, np.nan)

    for col in config.numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config.date_cols:
        if col in df.columns:
            fmt = config.date_formats.get(col)
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")

    return df, parsing


def split_ip_addresses(series: pd.Series) -> dict[str, pd.Series]:
    if series is None:
        empty = pd.Series(dtype="string")
        return {"public": empty, "private": empty, "has_pipe": empty}

    s = series.astype("string")
    has_pipe = s.str.contains("|", regex=False)
    parts = s.str.split("|", n=1, expand=True)
    public = parts[0].where(series.notna(), np.nan)
    if parts.shape[1] > 1:
        private = parts[1].where(has_pipe & series.notna(), np.nan)
    else:
        private = pd.Series([np.nan] * len(series), index=series.index)
    return {"public": public, "private": private, "has_pipe": has_pipe}


def normalize_ip(ip: str | float | None) -> str | None:
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return None
    ip_str = str(ip).strip()
    if not ip_str:
        return None
    # Strip port if present (e.g., 1.2.3.4:1234)
    if ":" in ip_str and ip_str.count(".") == 3:
        ip_str = ip_str.split(":", 1)[0]
    return ip_str


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def ua_to_device_family(ua: str | float | None) -> str:
    if ua is None or (isinstance(ua, float) and np.isnan(ua)):
        return "unknown"
    s = str(ua)
    if "iPhone" in s or "iPad" in s:
        return "ios"
    if "Android" in s:
        return "android"
    if "Windows" in s:
        return "windows"
    if "Macintosh" in s or "Mac OS" in s:
        return "mac"
    if "CrOS" in s:
        return "chromeos"
    if "Linux" in s:
        return "linux"
    return "other"


def ua_to_os_family(ua: str | float | None) -> str:
    return ua_to_device_family(ua)


def ua_to_browser_family(ua: str | float | None) -> str:
    if ua is None or (isinstance(ua, float) and np.isnan(ua)):
        return "unknown"
    s = str(ua)
    if "Edg/" in s:
        return "edge"
    if "OPR/" in s:
        return "opera"
    if "Firefox/" in s:
        return "firefox"
    if "Chrome/" in s:
        return "chrome"
    if "Safari/" in s:
        return "safari"
    return "other"


def groupby_apply(df: pd.DataFrame, key: str, func) -> pd.DataFrame:
    try:
        return df.groupby(key).apply(func, include_groups=False).reset_index()
    except TypeError:
        return df.groupby(key).apply(func).reset_index()


class GeoLocator:
    """IP geolocation using a cache file or live API."""
    
    def __init__(self, source: str, cache_path: Path, sleep_s: float) -> None:
        self.source = source
        self.cache_path = cache_path
        self.sleep_s = sleep_s
        self.cache: dict[str, tuple[float, float]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        df = pd.read_csv(self.cache_path)
        for _, row in df.iterrows():
            ip = str(row.get("ip"))
            lat = row.get("lat")
            lon = row.get("lon")
            success = row.get("success", True)  # Support new cache format
            if pd.notna(lat) and pd.notna(lon) and (success is True or success == "True" or success == 1):
                self.cache[ip] = (float(lat), float(lon))
        print(f"  Loaded {len(self.cache):,} IP geolocations from cache")

    def _save_cache(self) -> None:
        rows = [{"ip": ip, "lat": lat, "lon": lon} for ip, (lat, lon) in self.cache.items()]
        pd.DataFrame(rows).to_csv(self.cache_path, index=False)

    def resolve(self, ip: str | None) -> tuple[float, float] | None:
        if not ip:
            return None
        if ip in self.cache:
            return self.cache[ip]
        # In cache-only mode, don't make API calls
        if self.source in ("none", "cache"):
            return None
        if self.source == "ip-api":
            return self._resolve_ip_api(ip)
        raise ValueError(f"Unknown geo source: {self.source}")

    def _resolve_ip_api(self, ip: str) -> tuple[float, float] | None:
        url = f"http://ip-api.com/json/{ip}?fields=status,lat,lon"
        try:
            with urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        if payload.get("status") != "success":
            return None
        lat = payload.get("lat")
        lon = payload.get("lon")
        if lat is None or lon is None:
            return None
        coords = (float(lat), float(lon))
        self.cache[ip] = coords
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return coords

    def flush(self) -> None:
        if self.source not in ("none", "cache"):
            self._save_cache()



def compute_device_features(auth_df: pd.DataFrame) -> pd.DataFrame:
    auth_df = auth_df.copy()
    auth_df["device_family"] = auth_df["user_agent"].apply(ua_to_device_family)
    auth_df["os_family"] = auth_df["user_agent"].apply(ua_to_os_family)
    auth_df["browser_family"] = auth_df["user_agent"].apply(ua_to_browser_family)

    def _device_stats(group: pd.DataFrame) -> pd.Series:
        devices = group["device_family"].tolist()
        switches = sum(1 for i in range(1, len(devices)) if devices[i] != devices[i - 1])
        return pd.Series(
            {
                "auth_events": len(group),
                "unique_device_families": group["device_family"].nunique(dropna=True),
                "unique_os_families": group["os_family"].nunique(dropna=True),
                "unique_browser_families": group["browser_family"].nunique(dropna=True),
                "device_switches": switches,
            }
        )

    return groupby_apply(auth_df.sort_values("event_timestamp"), "user_id", _device_stats)


def compute_concurrent_activity(content_df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    content_df = content_df.copy()
    content_df = content_df[content_df["activity_timestamp"].notna()]
    content_df["ip_public"] = split_ip_addresses(content_df["ip_address"])["public"].apply(normalize_ip)

    window = np.timedelta64(window_minutes, "m")

    def _stats(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("activity_timestamp")
        times = group["activity_timestamp"].values
        ips = group["ip_public"].fillna("unknown").values

        left = 0
        counts: Counter[str] = Counter()
        distinct = 0
        max_distinct = 0
        events_in_multi = 0

        for right in range(len(times)):
            ip = ips[right]
            if counts[ip] == 0:
                distinct += 1
            counts[ip] += 1

            while times[right] - times[left] > window:
                ip_left = ips[left]
                counts[ip_left] -= 1
                if counts[ip_left] == 0:
                    distinct -= 1
                left += 1

            if distinct >= 2:
                events_in_multi += 1
            max_distinct = max(max_distinct, distinct)

        return pd.Series(
            {
                "content_events": len(group),
                "content_unique_ips": group["ip_public"].nunique(dropna=True),
                "concurrent_max_distinct_ips": int(max_distinct),
                "concurrent_event_ratio": float(events_in_multi / len(group)) if len(group) else 0.0,
            }
        )

    return groupby_apply(content_df, "user_id", _stats)


def compute_geo_features(auth_df: pd.DataFrame, metadata_df: pd.DataFrame, geolocator: GeoLocator) -> pd.DataFrame:
    auth_df = auth_df.copy()
    auth_df = auth_df[auth_df["event_type"] == "signIn"]
    auth_df["ip_public"] = split_ip_addresses(auth_df["ip_address"])["public"].apply(normalize_ip)

    # Resolve locations for unique IPs
    unique_ips = auth_df["ip_public"].dropna().unique().tolist()
    ip_to_coords: dict[str, tuple[float, float]] = {}
    for ip in unique_ips:
        coords = geolocator.resolve(ip)
        if coords:
            ip_to_coords[ip] = coords

    def _impossible_travel(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("event_timestamp")
        speeds: list[float] = []
        for prev, curr in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            ip_prev = prev.ip_public
            ip_curr = curr.ip_public
            if ip_prev not in ip_to_coords or ip_curr not in ip_to_coords:
                continue
            lat1, lon1 = ip_to_coords[ip_prev]
            lat2, lon2 = ip_to_coords[ip_curr]
            miles = haversine_miles(lat1, lon1, lat2, lon2)
            hours = (curr.event_timestamp - prev.event_timestamp).total_seconds() / 3600.0
            if hours <= 0:
                continue
            speeds.append(miles / hours)
        speeds_arr = np.array(speeds)
        return pd.Series(
            {
                "geo_max_speed_mph": float(speeds_arr.max()) if speeds_arr.size else np.nan,
                "geo_impossible_travel_count": int((speeds_arr > 500).sum()) if speeds_arr.size else 0,
            }
        )

    travel = groupby_apply(auth_df, "user_id", _impossible_travel)

    # Registration-login distance
    metadata = metadata_df[["user_id", "registration_lat", "registration_lng"]].dropna()
    meta_lookup = metadata.set_index("user_id")[["registration_lat", "registration_lng"]].to_dict("index")

    def _reg_distance(group: pd.DataFrame) -> pd.Series:
        user_id = group.name
        if user_id not in meta_lookup:
            return pd.Series({"geo_max_distance_from_registration": np.nan})
        reg_lat = meta_lookup[user_id]["registration_lat"]
        reg_lng = meta_lookup[user_id]["registration_lng"]
        distances: list[float] = []
        for ip in group["ip_public"].dropna().unique():
            if ip not in ip_to_coords:
                continue
            lat, lon = ip_to_coords[ip]
            distances.append(haversine_miles(reg_lat, reg_lng, lat, lon))
        return pd.Series(
            {"geo_max_distance_from_registration": float(max(distances)) if distances else np.nan}
        )

    reg_dist = groupby_apply(auth_df, "user_id", _reg_distance)
    return travel.merge(reg_dist, on="user_id", how="outer")


def compute_password_reset_features(
    auth_df: pd.DataFrame, exam_df: pd.DataFrame, plans_df: pd.DataFrame
) -> pd.DataFrame:
    resets = auth_df[auth_df["event_type"].isin(["resetPassword", "forgotPassword"])].copy()
    if resets.empty:
        return pd.DataFrame(columns=["user_id", "password_reset_count"])

    resets = resets.sort_values("event_timestamp")

    exam_df = exam_df.copy()
    exam_df = exam_df[exam_df["finished_at"].notna()]

    plans_df = plans_df.copy()
    plans_df = plans_df[plans_df["updated_at"].notna()]

    def _per_user(group: pd.DataFrame) -> pd.Series:
        user_id = group.name
        user_resets = group
        reset_count = len(user_resets)

        user_exams = exam_df[exam_df["user_id"] == user_id]
        user_plans = plans_df[plans_df["user_id"] == user_id]

        days_since_exam: list[float] = []
        days_since_readiness: list[float] = []
        after_positive = 0

        for ts in user_resets["event_timestamp"]:
            if not user_exams.empty:
                prior = user_exams[user_exams["finished_at"] <= ts]["finished_at"]
                if not prior.empty:
                    days_since_exam.append((ts - prior.max()).days)
            if not user_plans.empty:
                high_ready = user_plans[user_plans["readiness_score"] >= 0.8]
                prior_ready = high_ready[high_ready["updated_at"] <= ts]["updated_at"]
                if not prior_ready.empty:
                    days_since_readiness.append((ts - prior_ready.max()).days)
                positive = user_plans[user_plans["status"] == "positive"]
                if not positive.empty and (positive["updated_at"] <= ts).any():
                    after_positive += 1

        return pd.Series(
            {
                "password_reset_count": reset_count,
                "avg_days_since_exam_at_reset": float(np.mean(days_since_exam)) if days_since_exam else np.nan,
                "avg_days_since_high_readiness_at_reset": float(np.mean(days_since_readiness)) if days_since_readiness else np.nan,
                "password_resets_after_positive": after_positive,
            }
        )

    return groupby_apply(resets, "user_id", _per_user)


def compute_reset_behavior_shift(auth_df: pd.DataFrame) -> pd.DataFrame:
    resets = auth_df[auth_df["event_type"].isin(["resetPassword", "forgotPassword"])]
    if resets.empty:
        return pd.DataFrame(columns=["user_id", "reset_device_diversity_change"])

    auth_df = auth_df.copy()
    auth_df["device_family"] = auth_df["user_agent"].apply(ua_to_device_family)

    def _per_user(group: pd.DataFrame) -> pd.Series:
        user_id = group.name
        user_resets = resets[resets["user_id"] == user_id]
        if user_resets.empty:
            return pd.Series({"reset_device_diversity_change": np.nan})
        last_reset = user_resets["event_timestamp"].max()
        pre = group[(group["event_timestamp"] < last_reset) & (group["event_timestamp"] >= last_reset - pd.Timedelta(days=30))]
        post = group[(group["event_timestamp"] > last_reset) & (group["event_timestamp"] <= last_reset + pd.Timedelta(days=30))]
        pre_devices = pre["device_family"].nunique(dropna=True) if not pre.empty else 0
        post_devices = post["device_family"].nunique(dropna=True) if not post.empty else 0
        return pd.Series({"reset_device_diversity_change": post_devices - pre_devices})

    return groupby_apply(auth_df, "user_id", _per_user)


def compute_post_success_activity(
    content_df: pd.DataFrame, exam_df: pd.DataFrame, plans_df: pd.DataFrame
) -> pd.DataFrame:
    plans = plans_df[plans_df["status"] == "positive"].copy()
    if plans.empty:
        return pd.DataFrame(columns=["user_id", "post_positive_content_count", "post_positive_exam_count"])

    plans = plans.sort_values("updated_at")
    first_positive = plans.groupby("user_id")["updated_at"].min().reset_index()

    content_df = content_df.copy()
    exam_df = exam_df.copy()

    def _content_count(row: pd.Series) -> int:
        uid = row["user_id"]
        ts = row["updated_at"]
        return int(content_df[(content_df["user_id"] == uid) & (content_df["activity_timestamp"] > ts)].shape[0])

    def _exam_count(row: pd.Series) -> int:
        uid = row["user_id"]
        ts = row["updated_at"]
        return int(exam_df[(exam_df["user_id"] == uid) & (exam_df["started_at"] > ts)].shape[0])

    first_positive["post_positive_content_count"] = first_positive.apply(_content_count, axis=1)
    first_positive["post_positive_exam_count"] = first_positive.apply(_exam_count, axis=1)
    return first_positive.rename(columns={"updated_at": "positive_at"})


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series([0.0] * len(series), index=series.index)
    return 0.6745 * (series - median) / mad


def build_feature_report(output_dir: Path, features: pd.DataFrame, flags: pd.DataFrame, geo_source: str) -> None:
    lines: list[str] = []
    lines.append("# Week 2–3 Feature Engineering Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Geo source: `{geo_source}`")
    lines.append("")
    lines.append("## Feature Table Overview")
    lines.append("")
    lines.append(f"- Users with features: {len(features):,}")
    lines.append(f"- Columns: {len(features.columns)}")
    lines.append("")

    lines.append("## Flagged Users Overview")
    lines.append("")
    lines.append(f"- Users flagged: {int(flags['any_flag'].sum()):,}")
    lines.append(f"- Users with high score (>=2.5): {int((flags['risk_score'] >= 2.5).sum()):,}")
    lines.append("")

    top = flags.sort_values("risk_score", ascending=False).head(10)
    if not top.empty:
        lines.append("## Top 10 Users by Risk Score")
        lines.append("")
        lines.append("| user_id | risk_score | flags |")
        lines.append("| --- | --- | --- |")
        for _, row in top.iterrows():
            flag_list = ", ".join([f for f in ["impossible_travel", "concurrent", "device_diversity", "post_success", "reset_shift"] if row.get(f"flag_{f}")])
            lines.append(f"| {row['user_id']} | {row['risk_score']:.2f} | {flag_list} |")

    report_path = output_dir / "week2_3_report.md"
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2–3 feature engineering and initial detection signals.")
    parser.add_argument("--data-dir", default="account-sharing-export", help="Directory containing CSV exports.")
    parser.add_argument("--output-dir", default="week2_3_output", help="Directory to write outputs.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional per-file sample size for faster runs.")
    parser.add_argument("--window-minutes", type=int, default=10, help="Time window for concurrent activity.")
    parser.add_argument("--geo-source", choices=["none", "cache", "ip-api"], default="cache", help="IP geolocation source. 'cache' uses pre-computed ip_geo_cache.csv.")
    parser.add_argument("--geo-cache", default="ip_geo_cache.csv", help="Path to IP geolocation cache CSV.")
    parser.add_argument("--geo-sleep", type=float, default=1.5, help="Seconds to sleep between geo API calls.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, pd.DataFrame] = {}
    parsing: dict[str, Any] = {}

    for cfg in DATASETS:
        df, parse = load_dataset(data_dir / cfg.filename, cfg, args.sample_rows)
        loaded[cfg.name] = df
        parsing[cfg.name] = parse

    auth_df = loaded["auth_events"]
    content_df = loaded["content_activity"]
    exam_df = loaded["exam_sessions"]
    meta_df = loaded["user_metadata"]
    plans_df = loaded["study_plans"]

    # Feature engineering
    print("Computing device features...")
    device_features = compute_device_features(auth_df)
    print("Computing concurrent activity features...")
    concurrent_features = compute_concurrent_activity(content_df, args.window_minutes)

    # Geo features - use cache if available
    geo_cache_path = Path(args.geo_cache)
    geo_source = args.geo_source
    if geo_source == "cache" and not geo_cache_path.exists():
        print(f"Warning: geo cache not found at {geo_cache_path}, skipping geo features")
        print("  Run geolite2_lookup.py first to generate the cache")
        geo_source = "none"
    
    if geo_source != "none":
        print(f"Computing geo features (source: {geo_source})...")
        geolocator = GeoLocator(geo_source, geo_cache_path, args.geo_sleep)
        geo_features = compute_geo_features(auth_df, meta_df, geolocator)
        geolocator.flush()
    else:
        geo_features = pd.DataFrame(columns=["user_id"])

    reset_features = compute_password_reset_features(auth_df, exam_df, plans_df)
    reset_shift = compute_reset_behavior_shift(auth_df)
    post_success = compute_post_success_activity(content_df, exam_df, plans_df)

    # Merge all features
    features = device_features.merge(concurrent_features, on="user_id", how="outer")
    features = features.merge(geo_features, on="user_id", how="outer")
    features = features.merge(reset_features, on="user_id", how="outer")
    features = features.merge(reset_shift, on="user_id", how="outer")
    features = features.merge(post_success, on="user_id", how="outer")

    # Flagging rules
    features["flag_impossible_travel"] = (features.get("geo_impossible_travel_count", 0) > 0)
    features["flag_concurrent"] = (features.get("concurrent_max_distinct_ips", 0) >= 2) & (
        features.get("concurrent_event_ratio", 0) > 0.05
    )
    features["flag_device_diversity"] = features.get("unique_device_families", 0) >= 4
    features["flag_post_success"] = (features.get("post_positive_exam_count", 0) > 0) | (
        features.get("post_positive_content_count", 0) > 0
    )
    features["flag_reset_shift"] = features.get("reset_device_diversity_change", 0) >= 2

    # Simple risk score
    flags = features[["user_id"]].copy()
    for flag in ["impossible_travel", "concurrent", "device_diversity", "post_success", "reset_shift"]:
        flags[f"flag_{flag}"] = features[f"flag_{flag}"].fillna(False)
    flags["risk_score"] = (
        flags["flag_impossible_travel"].astype(int) * 1.5
        + flags["flag_concurrent"].astype(int) * 1.2
        + flags["flag_device_diversity"].astype(int) * 1.0
        + flags["flag_post_success"].astype(int) * 1.0
        + flags["flag_reset_shift"].astype(int) * 0.8
    )

    # Add a simple anomaly score on selected numeric features
    for col in ["concurrent_event_ratio", "unique_device_families", "device_switches"]:
        if col in features.columns:
            features[f"{col}_z"] = robust_zscore(features[col].fillna(0))
    if {"concurrent_event_ratio_z", "unique_device_families_z", "device_switches_z"}.issubset(features.columns):
        features["anomaly_score"] = (
            features["concurrent_event_ratio_z"].abs()
            + features["unique_device_families_z"].abs()
            + features["device_switches_z"].abs()
        )

    flags["any_flag"] = flags[[c for c in flags.columns if c.startswith("flag_")]].any(axis=1)

    # Outputs
    features.to_csv(output_dir / "features_per_user.csv", index=False)
    flags.to_csv(output_dir / "flagged_users.csv", index=False)
    build_feature_report(output_dir, features, flags, args.geo_source)

    parsing_path = output_dir / "parsing_health.json"
    parsing_path.write_text(json.dumps(parsing, default=str, indent=2))

    print(f"Wrote features: {output_dir / 'features_per_user.csv'}")
    print(f"Wrote flags: {output_dir / 'flagged_users.csv'}")
    print(f"Wrote report: {output_dir / 'week2_3_report.md'}")
    print(f"Wrote parsing health: {parsing_path}")


if __name__ == "__main__":
    main()
