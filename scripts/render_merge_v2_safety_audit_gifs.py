#!/usr/bin/env python3
"""Renders WOMD logged-trajectory GIFs for the 20 SAFETY_CRITICAL targeted-
audit candidates in outputs/merge_v2_decision_audit_v2/safety_critical_
targeted_audit.csv (none overlap with the existing 136-sample GIF set).

Same pure log-playback pattern as render_merge_v2_human_review_rollouts.py
(datatypes.update_state_by_log) -- no PPO/Frenet/MPC. Frame window widened
to cover the audited negative-gap run (run_frames in `notes`), not just the
commit-frame window, so the visual clip actually shows the audited
interval.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waymax import datatypes, visualization

from src.scenarios.dataset_builder import sanitize_candidate_id_for_filename
from src.scenarios.scenario_loader import build_waymax_config, iter_scenarios, load_dataset_config
import json

AUDIT_CSV = "outputs/merge_v2_decision_audit_v2/safety_critical_targeted_audit.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUTPUT_DIR = Path("outputs/merge_v2_decision_audit_v2/safety_review/gifs")
FRAME_MARGIN = 15
GIF_DURATION_MS = 100


def to_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def load_evidence_lookup(path):
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["candidate_id"]] = rec
    return lookup


def parse_run_frames(notes):
    m = re.search(r"run_frames=\[(\d+),(\d+)\]", notes)
    return int(m.group(1)), int(m.group(2))


def gap_ttc_at_frame(evidence_rec, frame):
    for sample in evidence_rec["gap_timeseries"]:
        if sample["frame"] == frame:
            return sample
    return None


def fmt(v, is_ttc=False):
    if v is None:
        return "n/a"
    if is_ttc and v == float("inf"):
        return "no closing"
    return f"{v:.1f}"


def render_candidate(row, evidence_rec, dataset_config, out_dir):
    candidate_id = row["candidate_id"]
    run_start, run_end = parse_run_frames(row["notes"])
    generator = iter_scenarios(
        dataset_config, start_index=evidence_rec["record_index"], limit=1,
        source_dataset=evidence_rec.get("source_dataset", "WOMD"),
        source_split=evidence_rec.get("source_split", "training"),
    )
    records = list(generator)
    if len(records) != 1:
        raise RuntimeError(f"could not load record_index={evidence_rec['record_index']}")
    record = records[0]
    total_timesteps = int(np.asarray(record.state.log_trajectory.valid).shape[1])

    start = max(0, run_start - FRAME_MARGIN)
    end = min(total_timesteps - 1, run_end + FRAME_MARGIN)

    state = record.state
    if start > 0:
        state = datatypes.update_state_by_log(state, num_steps=start)

    frames = []
    for _ in range(start, end + 1):
        current_timestep = int(state.timestep)
        image = visualization.plot_simulator_state(state, use_log_traj=True)
        pil_frame = Image.fromarray(to_uint8(image)).convert("RGB")

        sample = gap_ttc_at_frame(evidence_rec, current_timestep)
        front_gap = sample["front_gap_m"] if sample else None
        rear_gap = sample["rear_gap_m"] if sample else None
        front_ttc = sample["front_ttc_s"] if sample else None
        rear_ttc = sample["rear_ttc_s"] if sample else None
        in_run = run_start <= current_timestep <= run_end

        draw = ImageDraw.Draw(pil_frame)
        lines = [
            candidate_id,
            f"frame={current_timestep}  audited_run=[{run_start},{run_end}]{'  <-- IN RUN' if in_run else ''}",
            f"front_gap={fmt(front_gap)}m  rear_gap={fmt(rear_gap)}m",
            f"front_ttc={fmt(front_ttc, True)}s  rear_ttc={fmt(rear_ttc, True)}s",
            f"verdict={row['diagnostic_verdict']}  min_clearance={fmt(float(row['min_physical_clearance_m']) if row['min_physical_clearance_m'] else None)}m",
        ]
        y = 4
        for line in lines:
            fill = (255, 80, 80) if in_run else (255, 255, 0)
            draw.text((5, y), line, fill=fill)
            y += 14

        frames.append(pil_frame)
        if current_timestep >= end:
            break
        state = datatypes.update_state_by_log(state, num_steps=1)

    safe_id = sanitize_candidate_id_for_filename(candidate_id)
    gif_path = out_dir / f"{safe_id}.gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=GIF_DURATION_MS, loop=0)
    return gif_path, len(frames)


def main():
    with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    evidence_lookup = load_evidence_lookup(EVIDENCE_JSONL)
    dataset_expansion = load_dataset_config(DATASET_CONFIG)

    by_shard = defaultdict(list)
    for row in rows:
        ev = evidence_lookup[row["candidate_id"]]
        by_shard[ev["source_shard"]].append(row)

    rendered = {}
    failed = {}
    for shard, shard_rows in sorted(by_shard.items()):
        physical_path = f"data/womd/training/{shard}"
        dataset_config = build_waymax_config(dataset_expansion, physical_path)
        print(f"[shard {shard}] rendering {len(shard_rows)} candidate(s)")
        for row in shard_rows:
            cid = row["candidate_id"]
            ev = evidence_lookup[cid]
            try:
                gif_path, n_frames = render_candidate(row, ev, dataset_config, OUTPUT_DIR)
                rendered[cid] = str(gif_path)
                print(f"  {cid}: {n_frames} frames -> {gif_path}")
            except Exception as exc:  # noqa: BLE001
                failed[cid] = repr(exc)
                print(f"  FAILED {cid}: {exc!r}")

    print(f"rendered={len(rendered)} failed={len(failed)}")

    # patch review_gif_path into a new CSV (never overwrite the audit CSV in place silently)
    fieldnames = list(rows[0].keys())
    for row in rows:
        if row["candidate_id"] in rendered:
            row["review_gif_path"] = rendered[row["candidate_id"]]
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"updated {AUDIT_CSV} with review_gif_path")
    return rendered, failed


if __name__ == "__main__":
    main()
