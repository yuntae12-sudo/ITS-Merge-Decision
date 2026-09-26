#!/usr/bin/env python3
"""Renders WOMD logged-trajectory GIFs for the TRAIN human-calibration
review sample.

Analysis/visualization-only: no PPO policy, Frenet planner, or MPC is
involved. Every frame comes from ``datatypes.update_state_by_log`` --
the exact same pure log-playback pattern ``scripts/run_rollout.py``
uses -- so the GIF shows what actually happened in the WOMD recording,
never a learned or planned trajectory.

Reads ``outputs/merge_v2_calibration/human_review_training.csv`` (never
modified) plus ``data/manifests/v2/evidence_training.jsonl`` for the
per-candidate (source_shard, record_index, frame window, gap/TTC)
needed to render a focused clip, groups candidates by physical shard so
each shard is scanned sequentially exactly once, and writes one GIF per
candidate plus a local (no server, relative-path-only) HTML review
index.
"""

import csv
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waymax import datatypes, visualization

from src.scenarios.dataset_builder import sanitize_candidate_id_for_filename
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
)

HUMAN_REVIEW_CSV = "outputs/merge_v2_calibration/human_review_training.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUTPUT_ROOT = Path("outputs/merge_v2_calibration/human_review_rollouts")
DERIVED_CSV = "outputs/merge_v2_calibration/human_review_training_with_rollouts.csv"

FRAME_MARGIN_BEFORE = 20
FRAME_MARGIN_AFTER = 30
COMPLETION_MARGIN_AFTER = 15
GIF_DURATION_MS = 100  # ~10 FPS, matches scripts/run_rollout.py


def to_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def load_human_review_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_evidence_lookup(path):
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["candidate_id"]] = rec
    return lookup


def frame_window(evidence_rec, total_timesteps):
    ie = evidence_rec["interaction_evidence"]
    commit_frame = ie["commit_frame"]
    completion_frame = ie["completion_frame"]

    start = max(0, commit_frame - FRAME_MARGIN_BEFORE)
    end = min(total_timesteps - 1, max(
        commit_frame + FRAME_MARGIN_AFTER,
        completion_frame + COMPLETION_MARGIN_AFTER,
    ))
    return start, end


def gap_ttc_at_frame(evidence_rec, frame):
    for sample in evidence_rec["gap_timeseries"]:
        if sample["frame"] == frame:
            return sample
    return None


def fmt_ttc(v):
    if v is None:
        return "n/a"
    if v == float("inf"):
        return "no closing"
    return f"{v:.1f}s"


def fmt_gap(v):
    if v is None:
        return "n/a"
    return f"{v:.1f}m"


def render_candidate(review_row, evidence_rec, dataset_config, out_dir):
    candidate_id = review_row["candidate_id"]
    source_lane = evidence_rec["topology_evidence"]["source_lane_id"]
    target_lane = evidence_rec["topology_evidence"]["target_lane_id"]
    commit_frame = evidence_rec["interaction_evidence"]["commit_frame"]

    frames = []
    generator = iter_scenarios(
        dataset_config,
        start_index=evidence_rec["record_index"],
        limit=1,
        source_dataset=evidence_rec.get("source_dataset", "WOMD"),
        source_split=evidence_rec.get("source_split", "training"),
    )
    records = list(generator)
    if len(records) != 1:
        raise RuntimeError(f"could not load record_index={evidence_rec['record_index']}")
    record = records[0]

    total_timesteps = int(np.asarray(record.state.log_trajectory.valid).shape[1])
    start, end = frame_window(evidence_rec, total_timesteps)

    state = record.state
    # Advance to `start` using pure logged-trajectory playback (no
    # policy/planner) before recording any frames.
    if start > 0:
        state = datatypes.update_state_by_log(state, num_steps=start)

    for _ in range(start, end + 1):
        current_timestep = int(state.timestep)
        image = visualization.plot_simulator_state(state, use_log_traj=True)
        pil_frame = Image.fromarray(to_uint8(image)).convert("RGB")

        sample = gap_ttc_at_frame(evidence_rec, current_timestep)
        front_gap = sample["front_gap_m"] if sample else None
        rear_gap = sample["rear_gap_m"] if sample else None
        front_ttc = sample["front_ttc_s"] if sample else None
        rear_ttc = sample["rear_ttc_s"] if sample else None

        draw = ImageDraw.Draw(pil_frame)
        lines = [
            candidate_id,
            f"frame={current_timestep}  transition_frame={commit_frame}",
            f"source_lane={source_lane}  target_lane={target_lane}",
            f"decision={review_row['automatic_decision']}  reason={review_row['automatic_reason']}",
            f"front_gap={fmt_gap(front_gap)}  rear_gap={fmt_gap(rear_gap)}",
            f"front_ttc={fmt_ttc(front_ttc)}  rear_ttc={fmt_ttc(rear_ttc)}",
        ]
        y = 4
        for line in lines:
            draw.text((5, y), line, fill=(255, 255, 0))
            y += 14

        frames.append(pil_frame)

        if current_timestep >= end:
            break
        state = datatypes.update_state_by_log(state, num_steps=1)

    safe_id = sanitize_candidate_id_for_filename(candidate_id)
    gif_path = out_dir / f"{safe_id}.gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=GIF_DURATION_MS, loop=0,
    )
    return gif_path, len(frames)


def build_batch_html(batch_name, rows_with_paths, out_path):
    lines = [
        "<html><head><meta charset='utf-8'>",
        f"<title>{batch_name}</title></head><body>",
        f"<h1>{batch_name}</h1>",
        "<p><a href='../index.html'>&larr; back to master index</a></p>",
        "<table border=1 cellpadding=6 cellspacing=0>",
        "<tr><th>order</th><th>priority</th><th>candidate_id</th>"
        "<th>stratum</th><th>automatic_decision</th><th>automatic_reason</th>"
        "<th>thresholds</th><th>GIF</th><th>static PNG</th></tr>",
    ]
    for row in rows_with_paths:
        thresholds = (
            f"occ={row['source_occupancy_frames']} "
            f"stable={row['target_stable_frames']} "
            f"lon={row['longitudinal_progress_m']} "
            f"lat={row['lateral_displacement_m']} "
            f"gap_persist={row['gap_order_persistence_frames']}"
        )
        gif_rel = Path(row["review_rollout_path"]).name
        png_rel = "../../../../" + row["review_png_path"]
        lines.append(
            "<tr>"
            f"<td>{row['review_order']}</td><td>{row['review_priority']}</td>"
            f"<td>{row['candidate_id']}</td><td>{row['sampling_stratum']}</td>"
            f"<td>{row['automatic_decision']}</td><td>{row['automatic_reason']}</td>"
            f"<td>{thresholds}</td>"
            f"<td><img src='{gif_rel}' width='400'></td>"
            f"<td><a href='{png_rel}' target='_blank'>static PNG</a></td>"
            "</tr>"
        )
    lines.append("</table></body></html>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_master_html(batch_summaries, out_path):
    lines = [
        "<html><head><meta charset='utf-8'>",
        "<title>MERGE v2 TRAIN Human Review Rollouts</title></head><body>",
        "<h1>MERGE v2 TRAIN Human Review Rollouts</h1>",
        "<p>136 TRAIN calibration candidates, WOMD logged-trajectory GIFs "
        "(no policy/planner). See REVIEW_GUIDE.md for label definitions.</p>",
        "<table border=1 cellpadding=6 cellspacing=0>",
        "<tr><th>batch</th><th>candidates</th><th>P1</th><th>P2</th>"
        "<th>P3</th><th>P4</th><th>P5</th><th>P6</th><th>link</th></tr>",
    ]
    for name, count, pri_counts in batch_summaries:
        lines.append(
            "<tr>"
            f"<td>{name}</td><td>{count}</td>"
            + "".join(f"<td>{pri_counts.get(p, 0)}</td>" for p in ("P1", "P2", "P3", "P4", "P5", "P6"))
            + f"<td><a href='{name}/index.html'>open</a></td>"
            "</tr>"
        )
    lines.append("</table></body></html>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_human_review_rows(HUMAN_REVIEW_CSV)
    evidence_lookup = load_evidence_lookup(EVIDENCE_JSONL)
    dataset_expansion = load_dataset_config(DATASET_CONFIG)

    missing_evidence = [r["candidate_id"] for r in rows if r["candidate_id"] not in evidence_lookup]
    if missing_evidence:
        raise SystemExit(f"missing evidence for: {missing_evidence}")

    by_shard = defaultdict(list)
    for row in rows:
        ev = evidence_lookup[row["candidate_id"]]
        by_shard[ev["source_shard"]].append(row)

    print(f"input candidates: {len(rows)}, distinct shards: {len(by_shard)}")

    rendered = {}
    failed = {}
    for shard, shard_rows in sorted(by_shard.items()):
        physical_path = f"data/womd/training/{shard}"
        dataset_config = build_waymax_config(dataset_expansion, physical_path)
        print(f"[shard {shard}] rendering {len(shard_rows)} candidate(s)")
        for row in shard_rows:
            candidate_id = row["candidate_id"]
            ev = evidence_lookup[candidate_id]
            batch_dir = OUTPUT_ROOT / row["review_batch"]
            try:
                gif_path, n_frames = render_candidate(row, ev, dataset_config, batch_dir)
                rendered[candidate_id] = (gif_path, n_frames)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failed[candidate_id] = repr(exc)
                print(f"  FAILED {candidate_id}: {exc!r}")

    print(f"rendered={len(rendered)} failed={len(failed)}")
    if failed:
        for cid, reason in failed.items():
            print(f"  FAILED {cid}: {reason}")

    # derived CSV: original columns + review_rollout_path, human fields untouched
    fieldnames = list(rows[0].keys()) + ["review_rollout_path"]
    out_rows = []
    for row in rows:
        new_row = dict(row)
        if row["candidate_id"] in rendered:
            gif_path, _ = rendered[row["candidate_id"]]
            new_row["review_rollout_path"] = str(gif_path)
        else:
            new_row["review_rollout_path"] = ""
        out_rows.append(new_row)

    Path(DERIVED_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(DERIVED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"wrote {DERIVED_CSV}")

    # HTML index
    batch_summaries = []
    for batch_name in sorted(set(r["review_batch"] for r in rows)):
        batch_rows = [r for r in out_rows if r["review_batch"] == batch_name and r["review_rollout_path"]]
        batch_dir = OUTPUT_ROOT / batch_name
        build_batch_html(batch_name, batch_rows, batch_dir / "index.html")
        pri_counts = defaultdict(int)
        for r in batch_rows:
            pri_counts[r["review_priority"]] += 1
        batch_summaries.append((batch_name, len(batch_rows), pri_counts))

    build_master_html(batch_summaries, OUTPUT_ROOT / "index.html")
    print(f"wrote {OUTPUT_ROOT / 'index.html'}")

    return rendered, failed


if __name__ == "__main__":
    main()
