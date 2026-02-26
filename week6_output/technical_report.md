# Account Sharing Detection - Technical Report

**Generated:** 2026-02-11 19:34 UTC

---

## 1. Executive Summary

### Key Findings

We analyzed **21,868 user accounts** for potential account sharing behavior using five detection methods. Our analysis identified:

| Risk Level | Count | % of Total |
|------------|-------|------------|
| **High Risk** | 110 | 0.50% |
| **Medium Risk** | 328 | 1.50% |
| Low Risk | 21,430 | 98.00% |

### Detection Method Breakdown

| Detection Method | Users Flagged | Description |
|-----------------|---------------|-------------|
| Geographic Impossibility | 572 | Logins from locations requiring impossible travel speed (>500 mph) |
| Concurrent Activity | 466 | Simultaneous activity from multiple distant IP addresses |
| Device Diversity | 130 | 4+ different device types used on the account |
| Post-Success Activity | 563 | Continued usage after user reported passing their exam |
| Password Reset Shift | 68 | Significant behavior change after password reset |

> [!IMPORTANT]
> High-risk accounts should be reviewed manually before any enforcement action.
> Medium-risk accounts warrant monitoring and may benefit from verification prompts.

## 2. Exploratory Data Analysis

### 2.1 Dataset Overview

| Dataset | Rows | Unique Users | Date Range |
|---------|------|--------------|------------|
| auth_events | 107,943 | 14,939 | 2025-01-26 to 2026-01-26 |
| content_activity | 1,350,964 | 15,267 | 2025-04-09 to 2026-01-26 |
| exam_sessions | 386,920 | 12,666 | 2025-01-26 to 2026-01-26 |
| subscriptions | 32,282 | 30,668 | 2019-02-26 to 2026-01-26 |
| user_metadata | 30,668 | 30,668 | 2019-01-30 to 2026-01-26 |
| study_plans | 188,682 | 26,968 | 2022-03-31 to 2124-11-10 |

### 2.2 Data Quality Notes

- **exam_sessions**: Negative exam durations (only where finished_at present): 4
- **exam_sessions**: Very long exam durations >300 min (only where finished_at present): 12828
- **exam_sessions**: Scores appear to be proportions (0-1).
- **subscriptions**: subscription_end < subscription_start: 80
- **user_metadata**: first_seen_at < registered_at: 6829
- **study_plans**: study_score out of [0, 1]: 381
- **study_plans**: target_exam_date appears extreme (>2035-12-31): 4

## 3. Feature Engineering

### 3.1 Primary Detection Features

#### Geographic Impossibility

| Feature | Description | Median | P95 |
|---------|-------------|--------|-----|
| `geo_max_speed_mph` | Maximum travel speed between consecutive logins (miles per hour) | 0.49 | 781.74 |
| `geo_impossible_travel_count` | Number of login pairs requiring >500 mph travel | 0.00 | 0.00 |
| `geo_max_distance_from_registration` | Maximum distance of logins from registration location | 84.66 | 2,390.28 |

#### Concurrent Activity

| Feature | Description | Median | P95 |
|---------|-------------|--------|-----|
| `concurrent_max_distinct_ips` | Maximum distinct IPs seen within 10-minute windows | 1.00 | 2.00 |
| `concurrent_event_ratio` | Proportion of activity occurring with multiple simultaneous IPs | 0.00 | 0.03 |
| `content_unique_ips` | Total unique IPs used for content activity | 3.00 | 18.00 |

#### Device Diversity

| Feature | Description | Median | P95 |
|---------|-------------|--------|-----|
| `unique_device_families` | Count of device types (iOS, Android, Windows, Mac, etc.) | 2.00 | 3.00 |
| `unique_browser_families` | Count of browser types used | 1.00 | 3.00 |
| `device_switches` | Number of device type changes across sessions | 1.00 | 6.00 |

#### Account Handoff

| Feature | Description | Median | P95 |
|---------|-------------|--------|-----|
| `password_reset_count` | Total password resets for the account | 2.00 | 6.00 |
| `password_resets_after_positive` | Resets after user reported passing exam | 0.00 | 2.00 |
| `reset_device_diversity_change` | Change in device diversity after password reset | 0.00 | 1.00 |
| `post_positive_content_count` | Content activity after passing status | 0.00 | 1.00 |
| `post_positive_exam_count` | Exam attempts after passing status | 0.00 | 1.00 |

### 3.2 Flag Thresholds

The following thresholds were used to flag suspicious accounts:

| Flag | Condition | Rationale |
|------|-----------|-----------|
| Impossible Travel | `geo_impossible_travel_count > 0` | Any instance of >500 mph travel is physically impossible |
| Concurrent Activity | `concurrent_max_distinct_ips >= 2` AND `concurrent_event_ratio > 0.05` | Multiple IPs in short windows suggests sharing |
| Device Diversity | `unique_device_families >= 4` | Using 4+ device types is unusual for single users |
| Post-Success Activity | `post_positive_content_count > 0` OR `post_positive_exam_count > 0` | Activity after passing exam suggests handoff |
| Reset Behavior Shift | `reset_device_diversity_change >= 2` | Major behavior change after password reset |

## 4. Model & Scoring

### 4.1 Ensemble Scoring Approach

We combined rule-based heuristic flags with anomaly detection for robust scoring:

```
final_score = 70% × heuristic_risk_score + 30% × anomaly_model_score
```

**Anomaly Model:** isolation_forest
- Trees: 300

### 4.2 Risk Tier Thresholds

- **High Risk:** final_score ≥ 0.5961
- **Medium Risk:** final_score ≥ 0.4708
- **Low Risk:** below medium threshold

### 4.3 Score Distribution

- **Minimum:** 0.0000
- **Median:** 0.1500
- **Mean:** 0.1626
- **P95:** 0.4124
- **Maximum:** 0.8980

## 5. Flagged User Analysis

### 5.1 Top High-Risk Users

The following users have the highest risk scores:

| User ID | Risk Tier | Final Score | Flags |
|---------|-----------|-------------|-------|
| `f56ac25c-dd37-4ecf-bd30-0926d7230fa5` | high | 0.8980 | impossible_travel,concurrent,device_diversity,post_success |
| `9f09f93f-b294-4a26-8adb-138c7d815684` | high | 0.8473 | impossible_travel,device_diversity,post_success,reset_shift |
| `c2f8c50f-da58-4e60-bb11-a092abc636f1` | high | 0.8473 | impossible_travel,device_diversity,post_success,reset_shift |
| `59fab9ac-a52d-4e93-bc44-981c57a83623` | high | 0.7707 | impossible_travel,concurrent,post_success |
| `77f081c5-d1e6-4116-ac48-221260c16e13` | high | 0.7454 | impossible_travel,device_diversity,post_success |
| `d90e88eb-cd62-44f2-ab0b-e794df80a540` | high | 0.7448 | impossible_travel,device_diversity,post_success |
| `786ca853-3cfb-4e29-a11c-19c0cf2097e0` | high | 0.7447 | impossible_travel,device_diversity,post_success |
| `e18fda1f-828e-4626-81b8-eea0240b9fd4` | high | 0.7437 | impossible_travel,device_diversity,post_success |
| `4dea8d4d-b549-4d9a-a3b5-c3560f2afbd9` | high | 0.7425 | impossible_travel,device_diversity,post_success |
| `ad956259-bd81-4f17-838f-4c566558baef` | high | 0.7197 | impossible_travel,device_diversity,reset_shift |
| `b9ad3a8f-d6a3-4591-9961-e413899a6609` | high | 0.7053 | concurrent,device_diversity,post_success |
| `96354ca7-5340-4579-b6c3-6197bbea0d5b` | high | 0.6804 | concurrent,device_diversity,reset_shift |
| `822e61cc-12dd-4716-aeb9-1958647e6ece` | high | 0.6433 | impossible_travel,concurrent |
| `f6a3ad7a-b85d-4e21-9c7c-cc5abe2785d5` | high | 0.6430 | impossible_travel,concurrent |
| `9d0dc421-4551-4ca7-999f-da7ca5f77f1a` | high | 0.6427 | impossible_travel,concurrent |

### 5.2 Flag Co-occurrence

Users with multiple flags are more likely to be sharing accounts:

| Number of Flags | Users |
|-----------------|-------|
| 0 flags | 20,246 |
| 1 flags | 1,460 |
| 2 flags | 150 |
| 3 flags | 9 |
| 4 flags | 3 |

## 6. Recommendations

### 6.1 Immediate Actions

1. **Manual Review of High-Risk Accounts**
   - Review the top high-risk users for verification
   - Look for corroborating evidence (payment info, support tickets)
   - Consider reaching out to users before enforcement

2. **Implement Real-time Monitoring**
   - Flag impossible travel events in real-time
   - Alert on password resets followed by geographic shifts

### 6.2 Policy Interventions

| Intervention | Target Users | Expected Impact |
|--------------|--------------|-----------------|
| Device limit (max 3 devices) | High device diversity | Prevents casual sharing |
| Geographic verification prompts | Impossible travel detected | Catches credential sharing |
| Post-exam account lockout | Post-success activity | Prevents account handoffs |
| Concurrent session limits | Multiple simultaneous IPs | Blocks real-time sharing |

### 6.3 Additional Data Recommendations

The following additional data would improve detection accuracy:

1. **Device fingerprints** - More reliable than user agent strings
2. **Payment method changes** - Strong signal for account handoffs
3. **Support ticket context** - May reveal legitimate account transfers
4. **Time-of-day patterns** - Different work schedules suggest different users

### 6.4 False Positive Considerations

> [!WARNING]
> Users with legitimate VPN usage or frequent travel may trigger false positives.
> Consider implementing an appeals process for flagged accounts.

Common false positive scenarios:
- Business travelers using VPNs
- Users with multiple personal devices (phone, tablet, laptop)
- Shared family computers
- Corporate proxy servers

## Appendix A: Technical Details

### A.1 Risk Score Weights

| Flag | Weight |
|------|--------|
| Impossible Travel | 1.5 |
| Concurrent Activity | 1.2 |
| Device Diversity | 1.0 |
| Post-Success Activity | 1.0 |
| Password Reset Shift | 0.8 |

### A.2 Geographic Calculations

- Distance: Haversine formula (great-circle distance)
- Speed: distance / time between consecutive signIn events
- Impossible threshold: 500 mph (faster than commercial aircraft)

### A.3 Concurrent Activity Detection

- Window size: 10 minutes (sliding window)
- Algorithm: Two-pointer scan for distinct IP counts
- Data source: Content activity events (1.35M events)
