"""Phase 2 Stage C-1: reproducible per-maneuver dataset descriptor table.

For all 168 canonical maneuvers (both splits), resets the frozen
production ``MergeEnvironment`` (Stage B-2.8 semantics: episode starts
at ``decision_start_frame``) and extracts, at that exact reset frame:

  - the identical COMMON_POLICY_STATE the FSM/PPO will receive (the
    14D vector, reused verbatim -- never re-derived independently, per
    Stage C Section 29's consistency requirement), plus
  - DATASET_ANALYSIS_DESCRIPTOR-only fields that are NOT part of the
    frozen 14D state (traffic density, closing-speed derived fields,
    merge-region length) -- computed from the SAME per-frame inputs
    ``build_observation`` used, via the same
    ``scenario_features.extract_interaction_features``/
    ``compute_traffic_density`` helpers, so there is no second,
    divergent definition of the same physical concept.

Deterministic and reproducible: no randomness, TRAIN/VALIDATION
membership is joined from the canonical
``data/manifests/phase2_dataset_split.csv`` (never recomputed), and
the only external dependency is the already-frozen production
environment code.

Output: data/manifests/phase2_dataset_difficulty.csv (identifiers +
physical descriptors only -- interaction tags/difficulty tiers are
added by a separate, later analysis step once TRAIN thresholds are
frozen; see docs/ or the Stage C report for the two-pass design).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import src.environment.merge_environment as merge_environment_module
from src.environment.dataset_split import load_split_manifest
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.scenario_features import compute_traffic_density

DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUTPUT_PATH = "data/manifests/phase2_dataset_difficulty.csv"

FIELDNAMES = [
    "maneuver_id",
    "scene_key",
    "split",
    "source_shard",
    "record_index",
    "decision_start_frame",
    "merge_start_frame",
    "start_shift_frames",
    "episode_horizon",
    # COMMON_POLICY_STATE (verbatim from the frozen 14D observation)
    "v_e",
    "d_m",
    "target_front_present",
    "target_front_gap",
    "target_front_relative_speed",
    "target_front_ttc",
    "target_rear_present",
    "target_rear_gap",
    "target_rear_relative_speed",
    "target_rear_ttc",
    "source_front_present",
    "source_front_gap",
    "source_front_relative_speed",
    "source_front_ttc",
    # DATASET_ANALYSIS_DESCRIPTOR (not part of the frozen policy state)
    "front_closing_speed",  # = target_front_relative_speed, present-gated
    "rear_closing_speed",  # = target_rear_relative_speed, present-gated
    "traffic_density",
    "merge_region_length_m",
]

# Single-shard cache to avoid redundant TFRecord scans across
# maneuvers sharing a shard (same technique as Stage B-2.7/B-2.8
# scratchpad audits).
_real_iter_scenarios = merge_environment_module.iter_scenarios
_cache_key = None
_cache_records = []


def _cached_iter_scenarios(dataset_config, limit=None, source_dataset=None, source_split=None):
    global _cache_key, _cache_records
    key = (str(dataset_config), source_dataset, source_split)
    if key != _cache_key:
        _cache_key = key
        _cache_records = []
    if limit is not None and len(_cache_records) < limit:
        _cache_records = list(
            _real_iter_scenarios(
                dataset_config, limit=limit, source_dataset=source_dataset, source_split=source_split
            )
        )
    return list(_cache_records[:limit] if limit is not None else _cache_records)


merge_environment_module.iter_scenarios = _cached_iter_scenarios


def _row_for_maneuver(env, spec, split):
    observation, info = env.reset(spec)

    ctx = env._episode_context
    source_polyline = env._polylines_by_id[ctx.active_source_lane_id]
    target_polyline = env._polylines_by_id[ctx.active_target_lane_id]

    traj = env._state.current_sim_trajectory
    ego_x = float(np.asarray(traj.x)[env._sdc_index, 0])
    ego_y = float(np.asarray(traj.y)[env._sdc_index, 0])

    object_ids = np.asarray(env._state.object_metadata.ids)
    object_types = np.asarray(env._state.object_metadata.object_types)
    valid = np.asarray(traj.valid)[:, 0].astype(bool)

    traffic_density = compute_traffic_density(
        ego_x,
        ego_y,
        object_ids,
        object_types,
        valid,
        np.asarray(traj.x)[:, 0],
        np.asarray(traj.y)[:, 0],
        env._sdc_id,
        env._agent_selection_config,
    )

    # merge_region_length_m: static per-scene geometry -- source-lane
    # arc length from merge_start_s to merge_end_s (the merge-region
    # span used by d_m), independent of ego/frame.
    from src.scenarios.merge_detector import compute_merge_start_end_s

    merge_start_s, merge_end_s = compute_merge_start_end_s(
        source_polyline, target_polyline, env._merge_topology_config
    )
    merge_region_length_m = float(merge_end_s - merge_start_s)

    target_front_present = bool(observation[2] == 1.0)
    target_rear_present = bool(observation[6] == 1.0)

    front_closing_speed = float(observation[4]) if target_front_present else None
    rear_closing_speed = float(observation[8]) if target_rear_present else None

    return {
        "maneuver_id": spec.maneuver_id,
        "scene_key": ctx.scene_key,
        "split": split,
        "source_shard": spec.source_shard,
        "record_index": spec.record_index,
        "decision_start_frame": info["decision_start_frame"],
        "merge_start_frame": info["merge_start_frame"],
        "start_shift_frames": info["merge_start_frame"] - info["decision_start_frame"],
        "episode_horizon": info["episode_horizon"],
        "v_e": float(observation[0]),
        "d_m": float(observation[1]),
        "target_front_present": int(observation[2]),
        "target_front_gap": float(observation[3]),
        "target_front_relative_speed": float(observation[4]),
        "target_front_ttc": float(observation[5]),
        "target_rear_present": int(observation[6]),
        "target_rear_gap": float(observation[7]),
        "target_rear_relative_speed": float(observation[8]),
        "target_rear_ttc": float(observation[9]),
        "source_front_present": int(observation[10]),
        "source_front_gap": float(observation[11]),
        "source_front_relative_speed": float(observation[12]),
        "source_front_ttc": float(observation[13]),
        "front_closing_speed": front_closing_speed,
        "rear_closing_speed": rear_closing_speed,
        "traffic_density": traffic_density,
        "merge_region_length_m": merge_region_length_m,
    }


def main():
    split_by_id = {r.maneuver_id: r.split for r in load_split_manifest()}

    train_specs = load_maneuver_specs("train")
    validation_specs = load_maneuver_specs("validation")
    all_specs = sorted(
        train_specs + validation_specs,
        key=lambda s: (s.source_shard, s.record_index),
    )

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG)

    rows = []
    for i, spec in enumerate(all_specs):
        split = split_by_id[spec.maneuver_id]
        row = _row_for_maneuver(env, spec, split)
        rows.append(row)
        print(f"[{i+1}/{len(all_specs)}] {spec.maneuver_id} ({split})", flush=True)

    rows.sort(key=lambda r: r["maneuver_id"])

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
