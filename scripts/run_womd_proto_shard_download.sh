#!/usr/bin/env bash
# Batch launcher for scripts/fetch_womd_scenario_proto_shards.py.
#
# Reads the explicit required-shard-index files produced by
# scripts/summarize_required_proto_shards.py and downloads them in
# batches of at most MAX_SHARDS_PER_RUN (enforced inside the fetch
# script itself -- this launcher never raises or bypasses that guard).
# No wildcard/--all form exists here or in the fetch script.
#
# Resumable: shards already present locally (correct filename under the
# split's output dir) are skipped without re-invoking the fetch script.
# Re-running this launcher after a partial run or a kill only fetches
# what's still missing.
#
# Training runs to completion before validation starts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/autonav/miniconda3/envs/its-merge/bin/python}"
FETCH_SCRIPT="scripts/fetch_womd_scenario_proto_shards.py"
BATCH_SIZE=20
LOG_DIR="outputs/merge_v2_locator"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

_local_shard_path() {
  local split="$1" shard_index="$2"
  local total_shards prefix
  if [[ "$split" == "training" ]]; then total_shards=1000; prefix="training"; else total_shards=150; prefix="validation"; fi
  printf "data/womd/scenario_proto/%s/%s.tfrecord-%05d-of-%05d" \
    "$split" "$prefix" "$shard_index" "$total_shards"
}

run_split() {
  local split="$1"
  local required_file="$2"
  local log_file="$LOG_DIR/proto_download_${split}_$(date --utc +%Y%m%dT%H%M%S).log"

  mapfile -t all_indices < "$required_file"

  local missing=()
  for idx in "${all_indices[@]}"; do
    [[ -z "$idx" ]] && continue
    local path
    path="$(_local_shard_path "$split" "$idx")"
    if [[ -f "$path" ]]; then
      echo "[$split] shard $idx already present, skipping: $path" | tee -a "$log_file"
    else
      missing+=("$idx")
    fi
  done

  echo "[$split] $(date --utc --iso-8601=seconds) total=${#all_indices[@]} missing=${#missing[@]}" | tee -a "$log_file"

  local i=0
  local n=${#missing[@]}
  while (( i < n )); do
    local batch=("${missing[@]:i:BATCH_SIZE}")
    local joined
    joined="$(IFS=,; echo "${batch[*]}")"
    echo "[$split] $(date --utc --iso-8601=seconds) fetching batch (${#batch[@]} shards): $joined" | tee -a "$log_file"
    if ! CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u "$FETCH_SCRIPT" \
        --split "$split" --shards "$joined" --yes 2>&1 | tee -a "$log_file"; then
      echo "[$split] $(date --utc --iso-8601=seconds) BATCH FAILED, aborting: $joined" | tee -a "$log_file"
      exit 1
    fi
    i=$(( i + BATCH_SIZE ))
  done

  echo "[$split] $(date --utc --iso-8601=seconds) DONE" | tee -a "$log_file"
}

run_split training data/manifests/v2/required_scenario_proto_shards_training.txt
run_split validation data/manifests/v2/required_scenario_proto_shards_validation.txt
