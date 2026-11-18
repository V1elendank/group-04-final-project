"""Check the reporting file's bytes without parsing or displaying its records."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "Data/split_manifest.json").read_text())
path = root / "Data/reporting_set.csv"
expected = manifest["files"]["reporting_set.csv"]["sha256"]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("FAIL: reporting_set.csv differs from the frozen checksum.")
print("PASS: reporting_set.csv is unchanged. No records were parsed or displayed.")
