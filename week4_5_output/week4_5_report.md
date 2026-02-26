# Week 4–5 Modeling Report

Generated: 2026-02-05T01:11:11.319871+00:00
Features input: `week2_3_output/features_per_user.csv`
Model: `isolation_forest` (sklearn available: True)

## Thresholds

- High tier threshold: **0.5961**
- Medium tier threshold: **0.4708**

## Counts by tier

- **low**: 21,430
- **medium**: 328
- **high**: 110

## Flag counts

- **flag_impossible_travel**: 572
- **flag_concurrent**: 466
- **flag_device_diversity**: 130
- **flag_post_success**: 563
- **flag_reset_shift**: 68

## Top 10 users by final score

| user_id | tier | final_score | risk_score | model_score_norm | flags |
| --- | --- | --- | --- | --- | --- |
| f56ac25c-dd37-4ecf-bd30-0926d7230fa5 | high | 0.8980 | 4.70 | 0.999 | impossible_travel,concurrent,device_diversity,post_success |
| 9f09f93f-b294-4a26-8adb-138c7d815684 | high | 0.8473 | 4.30 | 1.000 | impossible_travel,device_diversity,post_success,reset_shift |
| c2f8c50f-da58-4e60-bb11-a092abc636f1 | high | 0.8473 | 4.30 | 1.000 | impossible_travel,device_diversity,post_success,reset_shift |
| 59fab9ac-a52d-4e93-bc44-981c57a83623 | high | 0.7707 | 3.70 | 0.999 | impossible_travel,concurrent,post_success |
| 77f081c5-d1e6-4116-ac48-221260c16e13 | high | 0.7454 | 3.50 | 1.000 | impossible_travel,device_diversity,post_success |
| d90e88eb-cd62-44f2-ab0b-e794df80a540 | high | 0.7448 | 3.50 | 0.998 | impossible_travel,device_diversity,post_success |
| 786ca853-3cfb-4e29-a11c-19c0cf2097e0 | high | 0.7447 | 3.50 | 0.998 | impossible_travel,device_diversity,post_success |
| e18fda1f-828e-4626-81b8-eea0240b9fd4 | high | 0.7437 | 3.50 | 0.994 | impossible_travel,device_diversity,post_success |
| 4dea8d4d-b549-4d9a-a3b5-c3560f2afbd9 | high | 0.7425 | 3.50 | 0.990 | impossible_travel,device_diversity,post_success |
| ad956259-bd81-4f17-838f-4c566558baef | high | 0.7197 | 3.30 | 0.999 | impossible_travel,device_diversity,reset_shift |