# Data partitions

- `development_set.csv`: 6,748 rows for analysis and internal validation.
- `reporting_set.csv`: 1,687 rows reserved for final evaluation. Do not preview, explore, transform, or tune against this file.
- `split_manifest.json`: fixed split policy, checksums, row counts, and package versions. No reporting-set outcome summary is included.

Both CSVs preserve the 34 original columns and their values, including blanks; no imputation, encoding, feature selection, or filtering was performed. Their membership is disjoint and covers every source record. Known repeated player IDs stay in one partition.

Source: [U. T. Pedersen, NFL Draft 1985–2015, Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015). The source's terms apply to its data; no new data license is asserted here.

Run `python scripts/verify_reporting_integrity.py` from the repository root to verify the reporting checksum without parsing records. Read the main README for the predeclared modeling cohort, prior-exposure disclosure, and final-evaluation policy.
