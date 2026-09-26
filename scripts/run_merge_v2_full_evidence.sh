#!/usr/bin/env bash
# Sequential Full MERGE v2 Evidence Extraction launcher: training then
# validation, using scripts/extract_merge_v2_evidence.py's resume
# contract (append/skip-processed/flush+fsync) so a kill/reboot only
# costs the in-flight candidate, never the whole run.
#
# Uses only the explicit local Scenario-protobuf shards named in
# data/manifests/v2/required_scenario_proto_shards_{training,validation}.txt
# -- never a wildcard/glob over the full split.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/autonav/miniconda3/envs/its-merge/bin/python}"
LOG_DIR="outputs/merge_v2_full"
mkdir -p "$LOG_DIR"

GIT_SHA="$(git rev-parse HEAD)"

_proto_paths() {
  local split="$1" required_file="$2"
  local total_shards
  if [[ "$split" == "training" ]]; then total_shards=1000; else total_shards=150; fi
  local paths=()
  while IFS= read -r idx; do
    [[ -z "$idx" ]] && continue
    paths+=("data/womd/scenario_proto/${split}/${split}.tfrecord-$(printf '%05d' "$idx")-of-$(printf '%05d' "$total_shards")")
  done < "$required_file"
  printf '%s\n' "${paths[@]}"
}

run_split() {
  local split="$1" dataset_config="$2" candidates_csv="$3" required_file="$4"
  local output="data/manifests/v2/evidence_${split}.jsonl"
  local review_dir="outputs/merge_v2_review/${split}"
  local log_file="$LOG_DIR/${split}_full_extraction.log"

  mapfile -t proto_paths < <(_proto_paths "$split" "$required_file")

  {
    echo "[$split] $(date --utc --iso-8601=seconds) START"
    echo "[$split] git_sha=$GIT_SHA"
    echo "[$split] dataset_config=$dataset_config"
    echo "[$split] candidates_csv=$candidates_csv"
    echo "[$split] proto_shard_count=${#proto_paths[@]}"
    echo "[$split] output=$output"
    echo "[$split] review_dir=$review_dir"
  } | tee -a "$log_file"

  if ! CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PYTHON_BIN" -u scripts/extract_merge_v2_evidence.py \
      --dataset-config "$dataset_config" \
      --scenario-proto "${proto_paths[@]}" \
      --transitions-csv "$candidates_csv" \
      --phase1-config configs/phase1_merge.yaml \
      --output "$output" \
      --review-dir "$review_dir" \
      2>&1 | tee -a "$log_file"; then
    echo "[$split] $(date --utc --iso-8601=seconds) FAILED" | tee -a "$log_file"
    exit 1
  fi

  echo "[$split] $(date --utc --iso-8601=seconds) DONE" | tee -a "$log_file"
}

run_split training \
  outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml \
  outputs/phase1/stage_b0_verify/merge_candidates_training_10shard_postfix.csv \
  data/manifests/v2/required_scenario_proto_shards_training.txt

run_split validation \
  outputs/phase1/validation_6shard_pilot/dataset_validation_6shard.yaml \
  outputs/phase1/feature_reference_fix/merge_candidates_6shard_postfix.csv \
  data/manifests/v2/required_scenario_proto_shards_validation.txt

echo "[full] $(date --utc --iso-8601=seconds) ALL DONE git_sha=$GIT_SHA"
