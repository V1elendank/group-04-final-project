"""One-time reporting-set reservation. Never imports or fits a model."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    data = ROOT / "Data"
    data.mkdir(exist_ok=True)
    outputs = [data / name for name in
               ("development_set.csv", "reporting_set.csv", "split_manifest.json")]
    if any(path.exists() for path in outputs):
        raise SystemExit("Split already exists. It must not be regenerated or overwritten.")

    # Preserve the original column values and blanks in both exported files.
    raw = pd.read_csv(args.source, dtype=str, keep_default_na=False)
    if len(raw) != 8435 or raw.shape[1] != 34:
        raise ValueError("This frozen protocol is for the supplied 8,435-row, 34-column CSV.")
    if raw.duplicated(["year", "pick"]).any():
        raise ValueError("Duplicate draft-selection keys require investigation before splitting.")
    pb = pd.to_numeric(raw["pb"], errors="raise")
    y = pb.ge(1).astype(int)
    # Empty IDs represent different records, not one large group.
    groups = pd.Series([
        "player:" + pid if pid else "selection:" + year + ":" + pick
        for pid, year, pick in zip(raw.player_id, raw.year, raw.pick)
    ])

    # Protocol fixed before reservation: first of five stratified grouped folds,
    # seed 42. Do not try different seeds/folds based on outcomes or performance.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    dev_idx, report_idx = next(splitter.split(np.zeros((len(raw), 1)), y, groups))
    assert len(dev_idx) == 6748 and len(report_idx) == 1687
    assert set(dev_idx).isdisjoint(report_idx)
    assert set(dev_idx).union(report_idx) == set(range(len(raw)))
    assert set(groups.iloc[dev_idx]).isdisjoint(groups.iloc[report_idx])

    # File creation only: no reporting-set previews, descriptive analysis, or scores.
    raw.iloc[np.sort(dev_idx)].to_csv(outputs[0], index=False, lineterminator="\n")
    raw.iloc[np.sort(report_idx)].to_csv(outputs[1], index=False, lineterminator="\n")
    manifest = {
        "created_date": "2026-11-17",
        "source_url": "https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015",
        "source_sha256": sha256(args.source),
        "source_rows": len(raw), "source_columns": raw.shape[1],
        "target": "success = 1 if pb >= 1; 0 otherwise (recorded snapshot, not lifetime success)",
        "algorithm": "StratifiedGroupKFold; first fold reserved",
        "n_splits": 5, "shuffle": True, "random_state": 42,
        "group_key": "nonempty player_id; otherwise unique year/pick",
        "development_rows": len(dev_idx), "reporting_rows": len(report_idx),
        "reporting_fraction": len(report_idx) / len(raw),
        "known_player_overlap": 0,
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs[:2]},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "prior_exposure": "Full-file descriptive and aggregate outcome summaries preceded reservation; no models were fitted. This is a prospective freeze, not a never-viewed dataset.",
        "reporting_policy": "No reporting-set EDA, transformations, tuning, threshold selection, or model evaluation until the final phase. Hash verification and upload are permitted.",
    }
    outputs[2].write_text(json.dumps(manifest, indent=2) + "\n")
    print("Reserved 6,748 development rows and 1,687 reporting rows (exactly 80/20).")
    print("Verified complete, disjoint membership and no shared known player IDs.")
    print(f"Reporting file: {outputs[1].stat().st_size:,} bytes. Contents are now frozen.")


if __name__ == "__main__":
    main()
