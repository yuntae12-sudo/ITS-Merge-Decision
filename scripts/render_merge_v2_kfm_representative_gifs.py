#!/usr/bin/env python3
"""Renders representative WOMD logged-trajectory GIFs (~30, not all 5545)
for Part C's new discriminability subsets: FOLLOW_vs_MERGE, KEEP_vs_MERGE,
REAR_PRESSURE, BOTH_SIDES_CONSTRAINED, PATH_FOLLOWING_LIKE.

Same pure log-playback pattern as the other MERGE v2 GIF renderers
(datatypes.update_state_by_log) -- no PPO/Frenet/MPC.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waymax import datatypes, visualization

from src.scenarios.dataset_builder import sanitize_candidate_id_for_filename
from src.scenarios.scenario_loader import build_waymax_config, iter_scenarios, load_dataset_config

SELECTION_CSV = "/tmp/claude-1000/-home-autonav-ITS-Merge-Decision/47b36172-6a32-4f68-a888-77daf4702ec4/scratchpad/partc_gif_selection.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
KFM_CSV = "outputs/merge_v2_decision_audit_v2/training_kfm_discriminability.csv"
DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUTPUT_ROOT = Path("outputs/merge_v2_decision_audit_v2/review/gifs")
FRAME_MARGIN_BEFORE = 20
FRAME_MARGIN_AFTER = 30
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


def load_kfm_lookup(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["candidate_id"]: r for r in csv.DictReader(f)}


def gap_ttc_at_frame(evidence_rec, frame):
    for sample in evidence_rec["gap_timeseries"]:
        if sample["frame"] == frame:
            return sample
    return None


def fmt(v):
    if v in (None, ""):
        return "n/a"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv == float("inf"):
        return "no closing"
    return f"{fv:.1f}"


def render_candidate(candidate_id, category, evidence_rec, kfm_row, dataset_config, out_dir):
    commit_frame = evidence_rec["interaction_evidence"]["commit_frame"]
    completion_frame = evidence_rec["interaction_evidence"]["completion_frame"]

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
    start = max(0, commit_frame - FRAME_MARGIN_BEFORE)
    end = min(total_timesteps - 1, max(commit_frame + FRAME_MARGIN_AFTER, completion_frame + 15))

    state = record.state
    if start > 0:
        state = datatypes.update_state_by_log(state, num_steps=start)

    frames = []
    for _ in range(start, end + 1):
        current_timestep = int(state.timestep)
        image = visualization.plot_simulator_state(state, use_log_traj=True)
        pil_frame = Image.fromarray(to_uint8(image)).convert("RGB")

        sample = gap_ttc_at_frame(evidence_rec, current_timestep)
        target_front_gap = sample["front_gap_m"] if sample else None
        target_rear_gap = sample["rear_gap_m"] if sample else None

        draw = ImageDraw.Draw(pil_frame)
        lines = [
            f"{candidate_id}  [{category}]",
            f"frame={current_timestep}  commit={commit_frame}",
            f"target_front_gap={fmt(target_front_gap)}m  target_rear_gap={fmt(target_rear_gap)}m",
            f"source_front_gap={fmt(kfm_row['source_front_gap_m'])}m  source_front_ttc={fmt(kfm_row['source_front_ttc_s'])}s",
            f"K={kfm_row['keep_affordance']} F={kfm_row['follow_affordance']} M={kfm_row['merge_affordance']}  archetype={kfm_row['archetype']}",
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
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=GIF_DURATION_MS, loop=0)
    return gif_path, len(frames)


def main():
    with open(SELECTION_CSV, newline="", encoding="utf-8") as f:
        selection = list(csv.DictReader(f))
    evidence_lookup = load_evidence_lookup(EVIDENCE_JSONL)
    kfm_lookup = load_kfm_lookup(KFM_CSV)
    dataset_expansion = load_dataset_config(DATASET_CONFIG)

    by_shard = defaultdict(list)
    for row in selection:
        ev = evidence_lookup[row["candidate_id"]]
        by_shard[ev["source_shard"]].append(row)

    print(f"selected {len(selection)} candidates across {len(by_shard)} shard(s)")

    rendered = {}
    failed = {}
    for shard, shard_rows in sorted(by_shard.items()):
        physical_path = f"data/womd/training/{shard}"
        dataset_config = build_waymax_config(dataset_expansion, physical_path)
        print(f"[shard {shard}] rendering {len(shard_rows)} candidate(s)")
        for row in shard_rows:
            cid = row["candidate_id"]
            category = row["category"]
            ev = evidence_lookup[cid]
            kfm_row = kfm_lookup[cid]
            out_dir = OUTPUT_ROOT / category
            try:
                gif_path, n_frames = render_candidate(cid, category, ev, kfm_row, dataset_config, out_dir)
                rendered[cid] = (category, gif_path, n_frames)
                print(f"  {cid}: {n_frames} frames")
            except Exception as exc:  # noqa: BLE001
                failed[cid] = repr(exc)
                print(f"  FAILED {cid}: {exc!r}")

    print(f"rendered={len(rendered)} failed={len(failed)}")

    # HTML index by category
    categories = sorted(set(row["category"] for row in selection))
    master_lines = [
        "<html><head><meta charset='utf-8'><title>MERGE v2 KFM Discriminability Representative Review</title></head><body>",
        "<h1>MERGE v2 KFM Discriminability Representative Review</h1>",
        "<p>Representative candidates only (not all 5545) -- see training_kfm_discriminability.csv for the full table.</p>",
        "<ul>",
    ]
    for cat in categories:
        master_lines.append(f"<li><a href='{cat}/index.html'>{cat}</a></li>")
    master_lines.append("</ul></body></html>")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT.parent / "index.html").write_text("\n".join(master_lines), encoding="utf-8")

    for cat in categories:
        cat_rows = [row for row in selection if row["category"] == cat and row["candidate_id"] in rendered]
        lines = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{cat}</title></head><body>",
            f"<h1>{cat}</h1>",
            "<p><a href='../../index.html'>&larr; back</a></p>",
            "<table border=1 cellpadding=6 cellspacing=0>",
            "<tr><th>candidate_id</th><th>GIF</th></tr>",
        ]
        for row in cat_rows:
            cid = row["candidate_id"]
            _, gif_path, _ = rendered[cid]
            lines.append(f"<tr><td>{cid}</td><td><img src='{gif_path.name}' width='420'></td></tr>")
        lines.append("</table></body></html>")
        (OUTPUT_ROOT / cat / "index.html").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUTPUT_ROOT.parent / 'index.html'}")
    return rendered, failed


if __name__ == "__main__":
    main()
