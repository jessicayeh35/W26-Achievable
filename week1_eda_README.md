# Week 1 EDA Runner

This script implements Week 1 tasks:
- Load all exported datasets
- Run basic EDA
- Identify data quality issues
- Produce a Markdown report and JSON summary

## Requirements
- Python 3.10+
- pandas
- numpy

Install dependencies (if needed):
```
python -m pip install pandas numpy
```

## Run
From the repo root:
```
python week1_eda.py --data-dir account-sharing-export --output-dir week1_output
```

Optional: run on a sample for faster iteration:
```
python week1_eda.py --data-dir account-sharing-export --output-dir week1_output --sample-rows 50000
```

## Outputs
The script writes:
- `week1_output/week1_report.md`
- `week1_output/week1_summary.json`
