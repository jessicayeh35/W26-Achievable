#!/usr/bin/env python3
"""
IP Geolocation Lookup Module - Batch Version

This module provides IP geolocation functionality using:
1. MaxMind GeoLite2 database (preferred - free, fast, no rate limits)
2. ip-api.com BATCH API (100 IPs per request, 15 requests/minute free)

For 90,000 IPs, the batch API approach takes ~60 minutes vs 33 hours with single requests.
"""
from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

# Try to import geoip2 for MaxMind database support
try:
    import geoip2.database
    import geoip2.errors
    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False
    geoip2 = None


@dataclass
class GeoResult:
    """Result of IP geolocation lookup."""
    ip: str
    lat: float | None
    lon: float | None
    city: str | None
    country: str | None
    success: bool


class GeoLite2Locator:
    """IP geolocation using MaxMind GeoLite2 database."""
    
    def __init__(self, db_path: Path) -> None:
        if not GEOIP2_AVAILABLE:
            raise RuntimeError(
                "geoip2 package not installed. Install with: pip install geoip2"
            )
        if not db_path.exists():
            raise FileNotFoundError(
                f"GeoLite2 database not found at {db_path}. "
                "Download from https://dev.maxmind.com/geoip/geolite2-free-geolocation-data"
            )
        self.reader = geoip2.database.Reader(str(db_path))
    
    def lookup(self, ip: str) -> GeoResult:
        try:
            response = self.reader.city(ip)
            return GeoResult(
                ip=ip,
                lat=response.location.latitude,
                lon=response.location.longitude,
                city=response.city.name,
                country=response.country.iso_code,
                success=True,
            )
        except (geoip2.errors.AddressNotFoundError, ValueError):
            return GeoResult(ip=ip, lat=None, lon=None, city=None, country=None, success=False)
    
    def lookup_batch(self, ips: list[str]) -> list[GeoResult]:
        return [self.lookup(ip) for ip in ips]
    
    def close(self) -> None:
        self.reader.close()


class IpApiBatchLocator:
    """IP geolocation using ip-api.com BATCH API (100 IPs per request, 15 req/min)."""
    
    BATCH_SIZE = 100
    REQUESTS_PER_MINUTE = 15
    
    def __init__(self) -> None:
        self.request_count = 0
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting: 15 requests per minute."""
        now = time.time()
        min_interval = 60.0 / self.REQUESTS_PER_MINUTE  # 4 seconds
        elapsed = now - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def lookup_batch(self, ips: list[str]) -> list[GeoResult]:
        """Lookup up to 100 IPs in a single request."""
        if len(ips) > self.BATCH_SIZE:
            raise ValueError(f"Batch size cannot exceed {self.BATCH_SIZE}")
        
        self._rate_limit()
        
        url = "http://ip-api.com/batch?fields=status,query,lat,lon,city,countryCode"
        payload = json.dumps([{"query": ip} for ip in ips]).encode("utf-8")
        
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        
        try:
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            # Return failures for all IPs on error
            return [GeoResult(ip=ip, lat=None, lon=None, city=None, country=None, success=False) 
                    for ip in ips]
        
        self.request_count += 1
        
        results = []
        for item in data:
            ip = item.get("query", "")
            if item.get("status") == "success":
                results.append(GeoResult(
                    ip=ip,
                    lat=item.get("lat"),
                    lon=item.get("lon"),
                    city=item.get("city"),
                    country=item.get("countryCode"),
                    success=True,
                ))
            else:
                results.append(GeoResult(
                    ip=ip, lat=None, lon=None, city=None, country=None, success=False
                ))
        
        return results
    
    def close(self) -> None:
        pass


def is_valid_public_ip(ip_str: str) -> bool:
    """Check if IP is a valid public IPv4 address."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.version == 4 and ip.is_global
    except ValueError:
        return False


def normalize_ip(ip: str | float | None) -> str | None:
    """Normalize IP address, handling public|private format and ports."""
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return None
    ip_str = str(ip).strip()
    if not ip_str or ip_str.lower() in ("null", "none", ""):
        return None
    
    # Handle "public|private" format - take the public part
    if "|" in ip_str:
        ip_str = ip_str.split("|")[0]
    
    # Strip port if present (e.g., 1.2.3.4:1234)
    if ":" in ip_str and ip_str.count(".") == 3:
        ip_str = ip_str.split(":")[0]
    
    return ip_str if ip_str else None


def extract_unique_ips(data_dir: Path) -> set[str]:
    """Extract all unique public IPs from auth_events and content_activity."""
    unique_ips: set[str] = set()
    
    files = [
        ("auth_events.csv", "ip_address"),
        ("content_activity.csv", "ip_address"),
    ]
    
    for filename, ip_col in files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue
        
        print(f"  Reading {filename}...")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_ip = row.get(ip_col, "")
                ip = normalize_ip(raw_ip)
                if ip and is_valid_public_ip(ip):
                    unique_ips.add(ip)
    
    return unique_ips


def load_existing_cache(cache_path: Path) -> dict[str, GeoResult]:
    """Load existing geolocation cache."""
    cache: dict[str, GeoResult] = {}
    if not cache_path.exists():
        return cache
    
    df = pd.read_csv(cache_path)
    for _, row in df.iterrows():
        ip = str(row["ip"])
        cache[ip] = GeoResult(
            ip=ip,
            lat=row.get("lat") if pd.notna(row.get("lat")) else None,
            lon=row.get("lon") if pd.notna(row.get("lon")) else None,
            city=row.get("city") if pd.notna(row.get("city")) else None,
            country=row.get("country") if pd.notna(row.get("country")) else None,
            success=bool(row.get("success", False)),
        )
    return cache


def save_cache(cache: dict[str, GeoResult], cache_path: Path) -> None:
    """Save geolocation cache to CSV."""
    rows = [
        {
            "ip": r.ip,
            "lat": r.lat,
            "lon": r.lon,
            "city": r.city,
            "country": r.country,
            "success": r.success,
        }
        for r in cache.values()
    ]
    pd.DataFrame(rows).to_csv(cache_path, index=False)


def batch_list(items: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-geolocate all IPs from auth_events and content_activity datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="account-sharing-export",
        help="Directory containing CSV exports.",
    )
    parser.add_argument(
        "--output-cache",
        default="ip_geo_cache.csv",
        help="Output path for IP geolocation cache.",
    )
    parser.add_argument(
        "--geolite2-db",
        default=None,
        help="Path to MaxMind GeoLite2-City.mmdb database (preferred).",
    )
    parser.add_argument(
        "--use-batch-api",
        action="store_true",
        help="Use ip-api.com batch API (100 IPs/request, 15 requests/min).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for testing).",
    )
    parser.add_argument(
        "--sample-ips",
        type=int,
        default=None,
        help="Only process a sample of IPs (for testing).",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    cache_path = Path(args.output_cache)
    
    print(f"IP Geolocation Lookup (Batch Version)")
    print(f"=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Cache output: {cache_path}")
    print()
    
    # Extract unique IPs
    print("Step 1: Extracting unique IPs from datasets...")
    unique_ips = extract_unique_ips(data_dir)
    print(f"  Found {len(unique_ips):,} unique public IPs")
    print()
    
    # Load existing cache
    print("Step 2: Loading existing cache...")
    cache = load_existing_cache(cache_path)
    print(f"  Loaded {len(cache):,} cached entries")
    
    # Determine IPs that need lookup
    ips_to_lookup = list(unique_ips - set(cache.keys()))
    print(f"  Need to lookup {len(ips_to_lookup):,} new IPs")
    print()
    
    if not ips_to_lookup:
        print("No new IPs to lookup. Cache is complete.")
        save_cache(cache, cache_path)
        print(f"Saved cache to {cache_path}")
        return
    
    # Sample if requested
    if args.sample_ips and len(ips_to_lookup) > args.sample_ips:
        import random
        random.shuffle(ips_to_lookup)
        ips_to_lookup = ips_to_lookup[:args.sample_ips]
        print(f"  Sampled down to {len(ips_to_lookup):,} IPs")
    
    # Initialize locator
    locator = None
    batch_size = 100
    
    if args.geolite2_db:
        db_path = Path(args.geolite2_db)
        if db_path.exists():
            print(f"Step 3: Using GeoLite2 database at {db_path}")
            try:
                locator = GeoLite2Locator(db_path)
                batch_size = 1000  # Can process faster with local db
            except Exception as e:
                print(f"  Error loading GeoLite2: {e}")
                locator = None
    
    if locator is None and args.use_batch_api:
        print(f"Step 3: Using ip-api.com batch API")
        print(f"  100 IPs per request, 15 requests per minute")
        total_batches = (len(ips_to_lookup) + 99) // 100
        estimated_minutes = total_batches / 15
        print(f"  Estimated time: {estimated_minutes:.1f} minutes for {total_batches} batches")
        locator = IpApiBatchLocator()
        batch_size = 100
    
    if locator is None:
        print()
        print("ERROR: No geolocation source available!")
        print()
        print("Options:")
        print("  1. Download GeoLite2-City.mmdb from MaxMind and use --geolite2-db")
        print("     https://dev.maxmind.com/geoip/geolite2-free-geolocation-data")
        print("     pip install geoip2")
        print()
        print("  2. Use --use-batch-api flag for ip-api.com batch endpoint")
        print()
        return
    
    # Perform batch lookups
    print()
    print(f"Step 4: Looking up {len(ips_to_lookup):,} IPs in batches of {batch_size}...")
    
    start_time = time.time()
    total_success = 0
    processed = 0
    batches_processed = 0
    
    for batch in batch_list(ips_to_lookup, batch_size):
        if args.max_batches and batches_processed >= args.max_batches:
            print(f"  Reached max batch limit ({args.max_batches}), stopping")
            break
        
        results = locator.lookup_batch(batch)
        for result in results:
            cache[result.ip] = result
            if result.success:
                total_success += 1
        
        processed += len(batch)
        batches_processed += 1
        
        if batches_processed % 10 == 0 or processed >= len(ips_to_lookup):
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            pct = 100 * processed / len(ips_to_lookup)
            print(f"  Processed {processed:,}/{len(ips_to_lookup):,} ({pct:.1f}%) - {rate:.1f} IPs/sec - {total_success:,} successful")
        
        # Save periodically
        if batches_processed % 50 == 0:
            save_cache(cache, cache_path)
    
    locator.close()
    
    # Final save
    print()
    print("Step 5: Saving cache...")
    save_cache(cache, cache_path)
    
    # Summary
    total_in_cache = len(cache)
    total_successful = sum(1 for r in cache.values() if r.success)
    print()
    print(f"Summary:")
    print(f"  Total IPs in cache: {total_in_cache:,}")
    print(f"  Successfully geolocated: {total_successful:,}")
    print(f"  Success rate: {100*total_successful/total_in_cache:.1f}%")
    print(f"  Saved to: {cache_path}")
    
    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
