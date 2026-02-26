# Account Sharing Detection Research Project

A comprehensive system for detecting potential account sharing among subscribers of an online exam preparation platform (FINRA SIE).

## Overview

This project implements five primary detection methods:

1. **Geographic Impossibility Detection** - Identifies physically impossible travel between login locations
2. **Concurrent Activity Detection** - Flags simultaneous activity from different IP addresses
3. **Device/Browser Diversity Analysis** - Detects users with suspiciously diverse device profiles
4. **Account Handoff Detection** - Identifies accounts passed between users after exam completion
5. **Password Reset Behavior Shift** - Detects behavior changes after password resets

## Project Structure

```
.
├── account-sharing-export/      # Raw data files (CSV)
│   ├── auth_events.csv          # Authentication events (~108K rows)
│   ├── content_activity.csv     # Study activity (~1.35M rows)
│   ├── exam_sessions.csv        # Practice exams (~387K rows)
│   ├── study_plans.csv          # User study plans (~189K rows)
│   ├── subscriptions.csv        # Subscription info (~32K rows)
│   └── user_metadata.csv        # Registration data (~31K rows)
│
├── week1_eda.py                 # Week 1: Exploratory Data Analysis
├── week1_output/                # EDA outputs
│   ├── week1_report.md          # EDA findings report
│   └── week1_summary.json       # Detailed statistics
│
├── geolite2_lookup.py           # IP Geolocation batch processor
├── ip_geo_cache.csv             # Cached IP geolocation results
│
├── week2_3_features.py          # Weeks 2-3: Feature Engineering
├── week2_3_output/              # Feature outputs
│   ├── features_per_user.csv    # Computed features for each user
│   ├── flagged_users.csv        # Users with detection flags
│   └── week2_3_report.md        # Feature engineering report
│
├── week4_5_modeling.py          # Weeks 4-5: Anomaly Detection & Scoring
├── week4_5_output/              # Modeling outputs
│   ├── scored_users.csv         # All users with risk scores
│   ├── manual_review_sample.csv # Sample for manual validation
│   └── method_breakdown.json    # Model configuration & thresholds
│
├── week6_final_report.py        # Week 6: Final Report Generator
├── week6_output/                # Final deliverables
│   ├── technical_report.md      # 10-15 page technical report
│   ├── high_risk_users.csv      # High-risk user list
│   ├── medium_risk_users.csv    # Medium-risk user list
│   └── *.png                    # Visualizations
│
└── requirements.txt             # Python dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install geoip2 for MaxMind database support
pip install geoip2
```

## Usage

### Quick Start (Full Pipeline)

```bash
# Step 1: Run EDA (Week 1)
python week1_eda.py

# Step 2: Pre-geolocate IP addresses (~60 minutes)
python geolite2_lookup.py --use-batch-api

# Step 3: Run feature engineering (Weeks 2-3)
python week2_3_features.py

# Step 4: Run modeling and scoring (Weeks 4-5)
python week4_5_modeling.py

# Step 5: Generate final report (Week 6)
python week6_final_report.py --generate-plots
```

### Individual Script Options

#### Week 1: EDA
```bash
python week1_eda.py \
    --data-dir account-sharing-export \
    --output-dir week1_output \
    --sample-rows 50000  # Optional: faster testing
```

#### IP Geolocation
```bash
# Using ip-api.com batch API (free, ~60 min for 90K IPs)
python geolite2_lookup.py --use-batch-api

# Using MaxMind GeoLite2 database (faster, requires download)
python geolite2_lookup.py --geolite2-db /path/to/GeoLite2-City.mmdb

# Test with small sample
python geolite2_lookup.py --use-batch-api --max-batches 5
```

#### Weeks 2-3: Feature Engineering
```bash
python week2_3_features.py \
    --data-dir account-sharing-export \
    --output-dir week2_3_output \
    --geo-cache ip_geo_cache.csv \
    --window-minutes 10  # Concurrent activity window
```

#### Weeks 4-5: Modeling
```bash
python week4_5_modeling.py \
    --features-csv week2_3_output/features_per_user.csv \
    --output-dir week4_5_output \
    --model auto  # Uses Isolation Forest if sklearn available
    --w-risk 0.70 \
    --w-model 0.30
```

#### Week 6: Final Report
```bash
python week6_final_report.py \
    --output-dir week6_output \
    --generate-plots  # Creates visualizations
```

## Detection Methods

### 1. Geographic Impossibility (Highest Confidence)
- Calculates travel speed between consecutive logins using Haversine distance
- Flags any user with travel speed > 500 mph (physically impossible)
- Uses IP geolocation for coordinate mapping

### 2. Concurrent Activity
- Uses sliding window (10 min) to detect multiple IPs active simultaneously
- Flags users with ≥2 distinct IPs in same window AND >5% of activity concurrent
- Based on 1.35M content activity events

### 3. Device Diversity
- Parses user agent strings to extract device/OS/browser families
- Flags users with ≥4 different device families (iOS, Android, Windows, Mac, etc.)
- Also tracks device switching frequency

### 4. Account Handoff Detection
- Identifies activity after user reports passing their exam (`status = "positive"`)
- Tracks password resets that occur after high readiness scores
- Measures behavior shift (device diversity change) after password resets

### 5. Password Reset Behavior Shift
- Compares device usage patterns before/after password resets
- Flags users where device diversity increases by ≥2 after reset

## Scoring System

Final risk score combines:
- **70%** Heuristic risk score (weighted sum of flags)
- **30%** Anomaly model score (Isolation Forest)

### Flag Weights
| Flag | Weight |
|------|--------|
| Impossible Travel | 1.5 |
| Concurrent Activity | 1.2 |
| Device Diversity | 1.0 |
| Post-Success Activity | 1.0 |
| Password Reset Shift | 0.8 |

### Risk Tiers
- **High Risk**: Top 0.5% (99.5th percentile)
- **Medium Risk**: Top 2% (98th percentile)
- **Low Risk**: Below medium threshold

## Output Files

### Technical Report (`week6_output/technical_report.md`)
Comprehensive 10-15 page report including:
- Executive summary with key findings
- EDA visualizations and data quality notes
- Feature engineering documentation
- Model performance metrics
- Flagged user analysis
- Recommendations

### Flagged User Lists
- `high_risk_users.csv` - Users requiring immediate review
- `medium_risk_users.csv` - Users for monitoring
- `manual_review_sample.csv` - Stratified sample for validation

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0 (for Isolation Forest)
- matplotlib >= 3.7.0 (for visualizations)
- seaborn >= 0.12.0 (for visualizations)
- geoip2 >= 4.7.0 (optional, for MaxMind database)

## Notes

### Data Privacy
- All user identifiers are opaque UUIDs
- No PII (names, emails, payment info) is included
- Data must be deleted upon project completion

### Known Limitations
- IP geolocation accuracy varies (city-level, not street-level)
- VPN users may trigger false positives
- Device fingerprinting would be more accurate than user agent parsing

### False Positive Considerations
- Business travelers with VPN usage
- Users with multiple personal devices
- Shared family computers
- Corporate proxy servers

## License

This project is for research purposes only. Data handling must comply with applicable data protection regulations.
