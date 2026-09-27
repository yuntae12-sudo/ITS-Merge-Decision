#!/usr/bin/env bash
# Sequential, resumable launcher for the MERGE v2 WOMD proto locator.
# Run this inside tmux. Re-running it is safe: the locator skips DONE shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/autonav/miniconda3/envs/its-merge/bin/python}"
OUTPUT_DIR="outputs/merge_v2_locator"
TARGET_CSV="data/manifests/v2/target_scenario_ids.csv"
LOCATOR_SCRIPT="scripts/build_womd_scenario_proto_locator.py"
RUN_MANIFEST="data/manifests/v2/womd_proto_locator_run_manifest.json"
PROGRESS_FILE="data/manifests/v2/womd_proto_scan_progress.jsonl"
LOCATOR_OUTPUT="data/manifests/v2/womd_scenario_proto_locator.csv"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the its-merge environment's Python." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# Record exact code/input provenance at the real scan start. This updates the
# prepared snapshot created during pre-scan validation.
"$PYTHON_BIN" - "$RUN_MANIFEST" "$TARGET_CSV" "$LOCATOR_SCRIPT" "${BASH_SOURCE[0]}" <<'PY'
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, target_path, locator_path, launcher_path = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with target_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
counts = {
    split: sum(row["source_split"] == split for row in rows)
    for split in ("training", "validation")
}
manifest = {
    "scan_start_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "git_status_porcelain": subprocess.check_output(
        ["git", "status", "--short"], text=True
    ).splitlines(),
    "target_scenario_ids_csv": str(target_path),
    "target_scenario_ids_sha256": sha256(target_path),
    "locator_script": str(locator_path),
    "locator_script_sha256": sha256(locator_path),
    "launcher_script": str(launcher_path),
    "launcher_script_sha256": sha256(launcher_path),
    "target_scenario_count": len(rows),
    "target_counts": counts,
    "expected_proto_shards": {"training": 1000, "validation": 150},
    "workers": 1,
    "progress_file": "data/manifests/v2/womd_proto_scan_progress.jsonl",
    "locator_output": "data/manifests/v2/womd_scenario_proto_locator.csv",
    "status": "running",
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary.replace(manifest_path)
print(f"[{manifest['scan_start_time_utc']}] wrote {manifest_path}", flush=True)
PY

run_split() {
  local split="$1"
  local log_file="$OUTPUT_DIR/${split}_full_scan.log"
  echo "[$(date --utc --iso-8601=seconds)] START split=$split workers=1" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PYTHON_BIN" -u "$LOCATOR_SCRIPT" \
    --split "$split" \
    --workers 1 \
    --target-scenario-ids "$TARGET_CSV" \
    --locator-output "$LOCATOR_OUTPUT" \
    --progress-file "$PROGRESS_FILE" \
    2>&1 | tee -a "$log_file"
  echo "[$(date --utc --iso-8601=seconds)] END split=$split status=success" | tee -a "$log_file"
}

run_split training
run_split validation
