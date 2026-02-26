# Week 1 EDA Report

Generated: 2026-02-02T21:35:39.914136+00:00
Data directory: `account-sharing-export`

Notes:
- Some exports contain **unquoted commas** inside the last column (notably `user_agent`).
- This runner repairs those rows by splitting on the first (N-1) commas and treating the remainder as the last field.

## Dataset Overview

| dataset | rows | columns | unique users | duplicate rows |
| --- | --- | --- | --- | --- |
| auth_events | 107,943 | 5 | 14,939 | 5 |
| content_activity | 1,350,964 | 4 | 15,267 | 22 |
| exam_sessions | 386,920 | 6 | 12,666 | 0 |
| subscriptions | 32,282 | 4 | 30,668 | 4 |
| user_metadata | 30,668 | 9 | 30,668 | 0 |
| study_plans | 188,682 | 7 | 26,968 | 46 |

## User Coverage

| dataset | unique users | missing metadata | missing subscriptions |
| --- | --- | --- | --- |
| auth_events | 14,939 | 0 | 0 |
| content_activity | 15,267 | 1,236 | 1,236 |
| exam_sessions | 12,666 | 19 | 19 |
| subscriptions | 30,668 | 0 | 0 |
| user_metadata | 30,668 | 0 | 0 |
| study_plans | 26,968 | 2,814 | 2,814 |

## Dataset Details

### auth_events

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 5 |
| actual columns | 5 |
| header matches expected | True |
| rows parsed | 107943 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 106100 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| event_timestamp | %Y-%m-%d %H:%M:%S | 100.00% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| event_timestamp | 2025-01-26T17:18:23 | 2026-01-26T16:53:02 | 0 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| ip_address | 35 | 0.03% |
| user_id | 0 | 0.00% |
| event_type | 0 | 0.00% |
| event_timestamp | 0 | 0.00% |
| user_agent | 0 | 0.00% |

**Top categorical values**

_event_type_

| value | count |
| --- | --- |
| signIn | 84631 |
| signOut | 14791 |
| signInFailed | 3796 |
| forgotPassword | 2504 |
| resetPassword | 2221 |

**IP parsing checks**

| metric | value |
| --- | --- |
| total ip rows | 107908 |
| has public|private | 107907 |
| public invalid | 74 |
| private invalid | 11606 |

### content_activity

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 4 |
| actual columns | 4 |
| header matches expected | True |
| rows parsed | 1350964 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 1326111 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| activity_timestamp | %Y-%m-%d %H:%M:%S | 100.00% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| activity_timestamp | 2025-04-09T11:51:29 | 2026-01-26T16:54:45 | 0 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| ip_address | 19 | 0.00% |
| user_id | 0 | 0.00% |
| activity_timestamp | 0 | 0.00% |
| user_agent | 0 | 0.00% |

**IP parsing checks**

| metric | value |
| --- | --- |
| total ip rows | 1350945 |
| has public|private | 1350945 |
| public invalid | 900 |
| private invalid | 110181 |

### exam_sessions

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 6 |
| actual columns | 6 |
| header matches expected | True |
| rows parsed | 386920 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 0 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| started_at | %Y-%m-%d %H:%M:%S | 100.00% | 0 |
| finished_at | %Y-%m-%d %H:%M:%S | 87.71% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| started_at | 2025-01-26T17:18:05 | 2026-01-26T17:17:16 | 0 |
| finished_at | 2025-01-26T17:26:30 | 2026-01-26T17:17:03 | 47541 |

**Exam duration (minutes) — only where finished_at is present**

| metric | value |
| --- | --- |
| n with finished_at | 339379 |
| p50 | 15.48 |
| p95 | 135.35 |
| p99 | 7609.57 |
| negative durations | 4 |
| > 300 min | 12828 |
| > 1440 min (24h) | 7466 |
| finished-only p50 | 15.50 |
| finished-only p95 | 129.74 |
| finished-only p99 | 7211.73 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| score | 47542 | 12.29% |
| finished_at | 47541 | 12.29% |
| user_id | 0 | 0.00% |
| exam_session_id | 0 | 0.00% |
| started_at | 0 | 0.00% |
| status | 0 | 0.00% |

**Top categorical values**

_status_

| value | count |
| --- | --- |
| finished | 327816 |
| started | 37459 |
| discarded | 16155 |
| archived | 5489 |
| ready | 1 |

**Potential data quality flags**

- Negative exam durations (only where finished_at present): 4
- Very long exam durations >300 min (only where finished_at present): 12828
- Scores appear to be proportions (0-1).

### subscriptions

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 4 |
| actual columns | 4 |
| header matches expected | True |
| rows parsed | 32282 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 0 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| subscription_start | %Y-%m-%d | 100.00% | 0 |
| subscription_end | %Y-%m-%d | 100.00% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| subscription_start | 2019-02-26T00:00:00 | 2026-01-26T00:00:00 | 0 |
| subscription_end | 2020-02-26T00:00:00 | 2099-01-01T00:00:00 | 0 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| user_id | 0 | 0.00% |
| subscription_start | 0 | 0.00% |
| subscription_end | 0 | 0.00% |
| is_active | 0 | 0.00% |

**Potential data quality flags**

- subscription_end < subscription_start: 80

**Notes / suggested resolution**

- Rows where `subscription_end < subscription_start` should be excluded or corrected upstream.

### user_metadata

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 9 |
| actual columns | 9 |
| header matches expected | True |
| rows parsed | 30668 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 0 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| registered_at | %Y-%m-%d %H:%M:%S | 100.00% | 0 |
| first_seen_at | %Y-%m-%d %H:%M:%S | 22.34% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| registered_at | 2019-01-30T16:38:41 | 2026-01-26T16:57:06 | 0 |
| first_seen_at | 2022-08-03T11:44:35 | 2026-01-25T19:47:54 | 23818 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| timezone | 30590 | 99.75% |
| first_seen_at | 23818 | 77.66% |
| acquisition_source | 10891 | 35.51% |
| registration_region | 966 | 3.15% |
| registration_country | 946 | 3.08% |
| registration_lng | 903 | 2.94% |
| registration_lat | 901 | 2.94% |
| user_id | 0 | 0.00% |

**Top categorical values**

_registration_country_

| value | count |
| --- | --- |
| US | 28952 |
| CA | 119 |
| GB | 93 |
| IN | 56 |
| BR | 37 |
| MX | 32 |
| PR | 26 |
| KR | 21 |

_registration_region_

| value | count |
| --- | --- |
| CA | 3547 |
| NY | 3231 |
| FL | 2529 |
| TX | 2464 |
| NJ | 1258 |
| IL | 1178 |
| PA | 1164 |
| GA | 1032 |

_timezone_

| value | count |
| --- | --- |
| America/New_York | 33 |
| America/Chicago | 25 |
| America/Los_Angeles | 13 |
| America/Denver | 3 |
| Pacific/Guam | 1 |
| America/Toronto | 1 |
| Asia/Tokyo | 1 |
| Asia/Kolkata | 1 |

_acquisition_source_

| value | count |
| --- | --- |
| google-ads | 7537 |
| enrollment | 4290 |
| Series7examtutor | 1043 |
| a10 | 1002 |
| capital-advantage | 829 |
| bing | 634 |
| Crush | 596 |
| promo-mailer | 570 |

**Potential data quality flags**

- first_seen_at < registered_at: 6829

**Notes / suggested resolution**

- Consider treating `first_seen_at` as unreliable (or clamp it to `registered_at`) before time-based analyses.

### study_plans

**Parsing health**

| metric | value |
| --- | --- |
| expected columns | 7 |
| actual columns | 7 |
| header matches expected | True |
| rows parsed | 188682 |
| rows skipped (too few fields) | 0 |
| rows repaired (extra commas merged into last field) | 0 |

_Datetime parsing_

| column | format | parsed % of rows | parse failures |
| --- | --- | --- | --- |
| target_exam_date | %Y-%m-%d | 100.00% | 3 |
| created_at | %Y-%m-%d %H:%M:%S | 100.00% | 0 |
| updated_at | %Y-%m-%d %H:%M:%S | 100.00% | 0 |

**Date ranges**

| column | min | max | missing |
| --- | --- | --- | --- |
| target_exam_date | 2022-03-31T00:00:00 | 2124-11-10T00:00:00 | 3 |
| created_at | 2022-03-30T19:20:27 | 2026-01-26T17:04:05 | 0 |
| updated_at | 2022-03-30T21:24:20 | 2026-01-26T17:05:41 | 0 |

**Top missing columns**

| column | missing | pct |
| --- | --- | --- |
| readiness_score | 154233 | 81.74% |
| study_score | 23076 | 12.23% |
| target_exam_date | 3 | 0.00% |
| user_id | 0 | 0.00% |
| status | 0 | 0.00% |
| created_at | 0 | 0.00% |
| updated_at | 0 | 0.00% |

**Top categorical values**

_status_

| value | count |
| --- | --- |
| inactive | 151537 |
| feedback | 26207 |
| positive | 5959 |
| active | 3680 |
| negative | 1299 |

**Potential data quality flags**

- study_score out of [0, 1]: 381
- target_exam_date appears extreme (>2035-12-31): 4

**Notes / suggested resolution**

- Extreme `target_exam_date` values (e.g. far-future years) are likely invalid placeholders; consider filtering beyond a cutoff (e.g. 2035).
