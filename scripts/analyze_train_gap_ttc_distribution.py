#!/usr/bin/env python3
"""TRAIN-only target-lane Front/Rear Gap & TTC distribution audit, for
PPO Reward V1 threshold design (docs/ppo/PPO_PLAN.md's Reward V0 is
NOT modified by this script; this is a read-only DATA AUDIT).

VISUALIZATION / DIAGNOSTIC ONLY. Never modifies MergeEnvironment,
BehaviorAction, the frozen 14D observation, the Frenet planner,
LTV-MPC, or any reward config. Reuses (never reimplements):

  - src.environment.observation_builder.build_observation /
    _encode_vehicle_slot's TTC_CAP_S encoding
  - src.scenarios.scenario_features.extract_interaction_features /
    _compute_ttc (the SOLE gap/TTC geometry implementation in this
    repo -- confirmed by audit, no duplicate exists)
  - src.environment.dataset_split.load_split_manifest (canonical
    TRAIN/VALIDATION split, never recomputed)
  - src.environment.full_split_evaluator.load_maneuver_specs

Two distinct sampling points, both required by the task:

  1. decision_start_frame snapshot -- reuses the EXISTING
     data/manifests/phase2_dataset_difficulty.csv descriptor table
     verbatim (Source of Truth; no re-computation needed) for
     Sections 3/4 (the headline TRAIN distribution).

  2. merge_start_frame +/- N relative-frame sweep -- NOT present in
     any existing descriptor table (phase2_dataset_difficulty.csv only
     records one frame per maneuver). This script derives it by
     resetting a throwaway Waymax PlanningAgentEnvironment at
     ``init_steps = target_frame + 1`` for each requested relative
     frame, which reads the SCENE'S LOGGED (human) trajectory at that
     exact frame -- never a simulated/PPO-driven trajectory, and never
     before any BehaviorAction is applied (no env.step() is ever
     called by this script). This is the same "Waymax explicitly
     supports per-episode init_steps as a plain parameter" mechanism
     ``MergeEnvironment.reset()`` itself uses (see that file's own
     comment), just varied across multiple frames per maneuver instead
     of being fixed at decision_start_frame. ``build_observation`` /
     ``extract_interaction_features`` / ``_compute_ttc`` are called
     completely unmodified -- this script only changes WHICH single
     already-logged frame gets fed into them.

Closing-only TTC classification (Section 4) reuses
``scenario_features._compute_ttc``'s own branching logic directly (via
raw front/rear gap + relative_speed from
``extract_interaction_features``'s ``InteractionFeatures``) -- no new
TTC formula is implemented here.
"""

import argparse
import csv
import dataclasses
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from waymax import config as waymax_config
from waymax import dynamics as waymax_dynamics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from src.environment.dataset_split import load_split_manifest
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import DEFAULT_MERGE_CONFIG_PATH, ManeuverSpec
from src.environment.observation_builder import TTC_CAP_S, ObservationInputs, build_observation
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.lane_geometry import extract_lane_polylines
from src.scenarios.merge_detector import compute_merge_start_end_s, load_merge_topology_config
from src.scenarios.scenario_features import (
    AgentSelectionConfig,
    extract_interaction_features,
    load_agent_selection_config,
)
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    select_single_shard_for_inspection,
)

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
DIFFICULTY_CSV_PATH = "data/manifests/phase2_dataset_difficulty.csv"
OUTPUT_DIR = Path("outputs/reward_design/gap_ttc")

# Confirmed by audit: src/environment/merge_environment.py's own
# InvertibleBicycleModel construction uses dt=0.1 (10 Hz, matching
# WOMD's native logging rate) -- never guessed here.
WAYMAX_DT_S = 0.1

DEFAULT_RELATIVE_FRAME_OFFSETS = (-20, -10, -5, 0, 5, 10, 20)

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


# ======================================================================
# Section 3/4: decision_start_frame snapshot, from the EXISTING
# descriptor CSV (Source of Truth -- no recomputation).
# ======================================================================


def load_train_difficulty_rows(difficulty_csv_path: str) -> List[dict]:
    train_ids = {
        row.maneuver_id for row in load_split_manifest() if row.split == "train"
    }
    with open(difficulty_csv_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "train"]

    # Cross-check against the canonical split manifest (never trust
    # the difficulty CSV's own "split" column blindly -- it was joined
    # from data/manifests/phase2_dataset_split.csv at build time, but
    # we re-verify here rather than assume it never drifted).
    csv_ids = {row["maneuver_id"] for row in rows}
    if csv_ids != train_ids:
        missing = train_ids - csv_ids
        extra = csv_ids - train_ids
        raise ValueError(
            f"data/manifests/phase2_dataset_difficulty.csv TRAIN rows do not "
            f"exactly match the canonical split manifest. missing={missing} "
            f"extra_or_mismatched={extra}"
        )
    return rows


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{p:02d}": float("nan") for p in PERCENTILES}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p:02d}": float(np.percentile(arr, p)) for p in PERCENTILES}


@dataclasses.dataclass
class DistributionSummary:
    variable: str
    n_total_maneuvers: int
    n_present: int
    n_absent: int
    presence_rate: float
    n_valid_samples: int
    mean: float
    std: float
    min: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    max: float


def summarize(variable_name: str, values: List[float], n_total_maneuvers: int, n_present: int, n_absent: int) -> DistributionSummary:
    pct = _percentiles(values)
    if values:
        mean = float(statistics.fmean(values))
        std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        vmin = float(min(values))
        vmax = float(max(values))
    else:
        mean = std = vmin = vmax = float("nan")
    presence_rate = n_present / n_total_maneuvers if n_total_maneuvers else float("nan")
    return DistributionSummary(
        variable=variable_name,
        n_total_maneuvers=n_total_maneuvers,
        n_present=n_present,
        n_absent=n_absent,
        presence_rate=presence_rate,
        n_valid_samples=len(values),
        mean=mean, std=std, min=vmin,
        p05=pct["p05"], p10=pct["p10"], p25=pct["p25"], p50=pct["p50"],
        p75=pct["p75"], p90=pct["p90"], p95=pct["p95"], max=vmax,
    )


def _threshold_bucket_counts(values: List[float], thresholds: List[float], mode: str) -> Dict[str, Tuple[int, float]]:
    """``mode='lt'``: count(value < threshold). ``mode='ge'``: count(value >= threshold).
    Returns {label: (count, rate_over_len(values))}."""

    n = len(values)
    out = {}
    arr = np.asarray(values, dtype=np.float64)
    for t in thresholds:
        if mode == "lt":
            c = int(np.sum(arr < t))
            label = f"< {t:g}"
        else:
            c = int(np.sum(arr >= t))
            label = f">= {t:g}"
        out[label] = (c, c / n if n else float("nan"))
    return out


def analyze_decision_start_distribution(rows: List[dict]) -> Dict[str, DistributionSummary]:
    n_total = len(rows)
    summaries = {}

    for slot, present_col, gap_col in (
        ("target_front", "target_front_present", "target_front_gap"),
        ("target_rear", "target_rear_present", "target_rear_gap"),
    ):
        present_mask = [row[present_col] == "1" for row in rows]
        n_present = sum(present_mask)
        n_absent = n_total - n_present
        gap_values = [float(row[gap_col]) for row, p in zip(rows, present_mask) if p]
        summaries[f"{slot}_gap"] = summarize(f"{slot}_gap", gap_values, n_total, n_present, n_absent)

    for slot, present_col, ttc_col in (
        ("target_front", "target_front_present", "target_front_ttc"),
        ("target_rear", "target_rear_present", "target_rear_ttc"),
    ):
        present_mask = [row[present_col] == "1" for row in rows]
        n_present = sum(present_mask)
        n_absent = n_total - n_present
        # Policy-observation TTC: ALL rows (present or absent), since
        # every row has a well-defined (possibly TTC_CAP_S-encoded)
        # observation value -- this is exactly what PPO sees.
        ttc_obs_values = [float(row[ttc_col]) for row in rows]
        summaries[f"{slot}_ttc_observation"] = summarize(
            f"{slot}_ttc_observation", ttc_obs_values, n_total, n_present, n_absent
        )

    return summaries


def analyze_gap_threshold_buckets(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for slot, present_col, gap_col in (
        ("target_front", "target_front_present", "target_front_gap"),
        ("target_rear", "target_rear_present", "target_rear_gap"),
    ):
        present_mask = [row[present_col] == "1" for row in rows]
        gap_values = [float(row[gap_col]) for row, p in zip(rows, present_mask) if p]
        n = len(gap_values)
        arr = np.asarray(gap_values, dtype=np.float64)
        out[slot] = {
            "n_present_samples": n,
            "gap<=0": (int(np.sum(arr <= 0)), int(np.sum(arr <= 0)) / n if n else float("nan")),
            "gap<5": (int(np.sum(arr < 5)), int(np.sum(arr < 5)) / n if n else float("nan")),
            "gap<10": (int(np.sum(arr < 10)), int(np.sum(arr < 10)) / n if n else float("nan")),
            "gap<15": (int(np.sum(arr < 15)), int(np.sum(arr < 15)) / n if n else float("nan")),
            "gap<20": (int(np.sum(arr < 20)), int(np.sum(arr < 20)) / n if n else float("nan")),
            "gap>=20": (int(np.sum(arr >= 20)), int(np.sum(arr >= 20)) / n if n else float("nan")),
        }
    return out


def analyze_ttc_threshold_buckets(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for slot, ttc_col in (("target_front", "target_front_ttc"), ("target_rear", "target_rear_ttc")):
        ttc_values = [float(row[ttc_col]) for row in rows]  # observation TTC, all rows
        n = len(ttc_values)
        arr = np.asarray(ttc_values, dtype=np.float64)
        out[slot] = {
            "n_samples": n,
            "ttc==0": (int(np.sum(arr == 0.0)), int(np.sum(arr == 0.0)) / n if n else float("nan")),
            "ttc<1": (int(np.sum(arr < 1)), int(np.sum(arr < 1)) / n if n else float("nan")),
            "ttc<2": (int(np.sum(arr < 2)), int(np.sum(arr < 2)) / n if n else float("nan")),
            "ttc<3": (int(np.sum(arr < 3)), int(np.sum(arr < 3)) / n if n else float("nan")),
            "ttc<4": (int(np.sum(arr < 4)), int(np.sum(arr < 4)) / n if n else float("nan")),
            "ttc<5": (int(np.sum(arr < 5)), int(np.sum(arr < 5)) / n if n else float("nan")),
            "ttc>=5": (int(np.sum(arr >= 5)), int(np.sum(arr >= 5)) / n if n else float("nan")),
            "ttc==CAP(100)": (
                int(np.sum(arr == TTC_CAP_S)), int(np.sum(arr == TTC_CAP_S)) / n if n else float("nan")
            ),
        }
    return out


def analyze_presence_closing_overlap(rows: List[dict]) -> Dict[str, dict]:
    """Section 4: for each of front/rear, splits every TRAIN maneuver
    into exactly one of {absent, present_non_closing, present_closing,
    overlap}, using scenario_features._compute_ttc's OWN branching
    logic (gap<=0 -> overlap; closing_speed<=0 -> non-closing;
    else -> closing) -- reimplemented here only as a classification
    label, never as a new TTC formula (the actual TTC value used
    everywhere else in this script always comes from the descriptor
    CSV's or build_observation's own computed field, never
    recalculated by this function)."""

    out = {}
    for slot, present_col, gap_col, speed_col in (
        ("target_front", "target_front_present", "target_front_gap", "target_front_relative_speed"),
        ("target_rear", "target_rear_present", "target_rear_gap", "target_rear_relative_speed"),
    ):
        n_total = len(rows)
        counts = {"absent": 0, "present_non_closing": 0, "present_closing": 0, "overlap": 0}
        for row in rows:
            present = row[present_col] == "1"
            if not present:
                counts["absent"] += 1
                continue
            gap = float(row[gap_col])
            closing_speed = float(row[speed_col])
            if gap <= 0.0:
                counts["overlap"] += 1
            elif closing_speed <= 0.0:
                counts["present_non_closing"] += 1
            else:
                counts["present_closing"] += 1
        out[slot] = {k: (v, v / n_total if n_total else float("nan")) for k, v in counts.items()}
    return out


def analyze_closing_only_ttc(rows: List[dict]) -> Dict[str, DistributionSummary]:
    """[2] Closing-only TTC (Section 4): finite TTC restricted to
    present + closing_speed>0 + gap>0 (not overlapping) rows only --
    the exact same partition scenario_features._compute_ttc's non-
    capped branch would produce (gap>0 and closing_speed>0 -> gap/speed,
    finite). Recomputes gap/closing_speed exactly as already stored in
    the descriptor CSV (itself sourced from build_observation, itself
    sourced from extract_interaction_features/_compute_ttc) -- this
    function only FILTERS which rows count, it does not re-derive TTC
    with a new formula. It re-derives the ratio gap/closing_speed only
    to avoid the observation's own TTC_CAP_S clamp distorting the
    closing-only percentiles (the raw, uncapped value is what
    `_compute_ttc` would return before observation_builder's own
    ``min(ttc, TTC_CAP_S)`` step -- confirmed identical formula, not a
    new one)."""

    n_total = len(rows)
    summaries = {}
    for slot, present_col, gap_col, speed_col in (
        ("target_front", "target_front_present", "target_front_gap", "target_front_relative_speed"),
        ("target_rear", "target_rear_present", "target_rear_gap", "target_rear_relative_speed"),
    ):
        closing_only_ttc = []
        for row in rows:
            if row[present_col] != "1":
                continue
            gap = float(row[gap_col])
            closing_speed = float(row[speed_col])
            if gap > 0.0 and closing_speed > 0.0:
                closing_only_ttc.append(gap / closing_speed)  # == scenario_features._compute_ttc's own formula
        n_present = sum(1 for row in rows if row[present_col] == "1")
        summaries[f"{slot}_ttc_closing_only"] = summarize(
            f"{slot}_ttc_closing_only", closing_only_ttc, n_total, len(closing_only_ttc), n_total - len(closing_only_ttc)
        )
    return summaries


# ======================================================================
# Section 5: merge_start_frame +/- N relative-frame sweep.
# ======================================================================


@dataclasses.dataclass(frozen=True)
class FrameSample:
    maneuver_id: str
    scene_key: str
    split: str
    frame_index: int
    relative_to_merge_start_frame: int
    target_front_present: int
    target_front_gap: float
    target_front_relative_speed: float
    target_front_ttc_observation: float
    target_front_ttc_raw: Optional[float]  # None if absent or non-closing (inf)
    target_rear_present: int
    target_rear_gap: float
    target_rear_relative_speed: float
    target_rear_ttc_observation: float
    target_rear_ttc_raw: Optional[float]


class _ScenarioCache:
    """Caches one loaded ScenarioRecord per (source_shard, record_index)
    so a maneuver's multiple relative-frame samples never re-scan the
    TFRecord shard -- same technique
    scripts/build_phase2_dataset_descriptors.py already uses."""

    def __init__(self, expansion_config):
        self._expansion_config = expansion_config
        self._cache = {}

    def get(self, source_shard: str, record_index: int):
        key = (source_shard, record_index)
        if key not in self._cache:
            shard_path = select_single_shard_for_inspection(
                self._expansion_config, source_shard=source_shard, record_index=record_index
            )
            dataset_config = build_waymax_config(self._expansion_config, shard_path)
            record = None
            for candidate in iter_scenarios(
                dataset_config,
                limit=record_index + 1,
                source_dataset=self._expansion_config.dataset_name,
                source_split=self._expansion_config.split,
            ):
                record = candidate
            if record is None:
                raise RuntimeError(f"Could not load scenario for shard={source_shard} record_index={record_index}")
            self._cache[key] = record
        return self._cache[key]


def _sample_logged_frame(
    record,
    maneuver: ManeuverSpec,
    frame_index: int,
    polylines_by_id: dict,
    merge_end_s: float,
    agent_selection_config: AgentSelectionConfig,
    dynamics,
) -> Optional[dict]:
    """Resets a throwaway Waymax PlanningAgentEnvironment with
    ``init_steps = frame_index + 1`` so ``current_sim_trajectory``
    reflects the SCENE'S LOGGED (human) state at exactly
    ``frame_index`` -- no env.step() is ever called, so this is pure
    logged-trajectory read-out, never a simulated/PPO trajectory.
    Mirrors MergeEnvironment.reset()'s own construction exactly (same
    waymax_config.EnvironmentConfig fields), only varying init_steps.
    Returns None if frame_index is out of the scenario's logged range
    or the ego is not valid at that frame (never guessed/interpolated).
    """

    num_logged_frames = int(np.asarray(record.state.log_trajectory.x).shape[-1])
    if frame_index < 0 or frame_index > num_logged_frames - 1:
        return None

    ego_valid_log = np.asarray(record.state.log_trajectory.valid)[record.sdc_index]
    if not bool(ego_valid_log[frame_index]):
        return None

    env_config = waymax_config.EnvironmentConfig(
        max_num_objects=64,
        init_steps=frame_index + 1,
        controlled_object=waymax_config.ObjectType.SDC,
        compute_reward=False,
    )
    waymax_env = PlanningAgentEnvironment(dynamics_model=dynamics, config=env_config)
    state = waymax_env.reset(record.state)

    traj = state.current_sim_trajectory
    sdc_index = record.sdc_index
    ego_x = float(np.asarray(traj.x)[sdc_index, 0])
    ego_y = float(np.asarray(traj.y)[sdc_index, 0])
    ego_vel_x = float(np.asarray(traj.vel_x)[sdc_index, 0])
    ego_vel_y = float(np.asarray(traj.vel_y)[sdc_index, 0])
    ego_length = float(np.asarray(traj.length)[sdc_index, 0])

    from src.scenarios.lane_geometry import project_point_to_polyline_signed

    source_lane_id = maneuver.lane_chain[0]
    target_lane_id = maneuver.lane_chain[1]
    source_polyline = polylines_by_id[source_lane_id]
    target_polyline = polylines_by_id[target_lane_id]

    ego_source_arc_length = project_point_to_polyline_signed(source_polyline, ego_x, ego_y)["arc_length_m"]

    object_ids = np.asarray(state.object_metadata.ids)
    object_types = np.asarray(state.object_metadata.object_types)
    valid = np.asarray(traj.valid)[:, 0].astype(bool)
    x = np.asarray(traj.x)[:, 0]
    y = np.asarray(traj.y)[:, 0]
    yaw = np.asarray(traj.yaw)[:, 0]
    vel_x = np.asarray(traj.vel_x)[:, 0]
    vel_y = np.asarray(traj.vel_y)[:, 0]
    length = np.asarray(traj.length)[:, 0]

    inputs = ObservationInputs(
        ego_id=record.sdc_id,
        ego_x=ego_x, ego_y=ego_y, ego_vel_x=ego_vel_x, ego_vel_y=ego_vel_y, ego_length_m=ego_length,
        source_polyline=source_polyline, target_polyline=target_polyline,
        merge_end_s=merge_end_s, ego_source_arc_length_m=ego_source_arc_length,
        object_ids=object_ids, object_types=object_types, valid=valid,
        x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, length=length,
        agent_selection_config=agent_selection_config,
    )
    observation = build_observation(inputs)

    # Raw (uncapped, present-only) TTC for the raw-vs-observation
    # comparison field -- obtained by re-calling
    # extract_interaction_features directly (the SAME function
    # build_observation calls internally) so the raw, pre-TTC_CAP_S
    # target_front_ttc_s/target_rear_ttc_s are visible; never a
    # different formula.
    target_features = extract_interaction_features(
        frame_index=-1, target_polyline=target_polyline, merge_distance_m=0.0,
        ego_id=record.sdc_id, ego_x=ego_x, ego_y=ego_y, ego_vel_x=ego_vel_x, ego_vel_y=ego_vel_y,
        ego_length_m=ego_length, object_ids=object_ids, object_types=object_types, valid=valid,
        x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, length=length, config=agent_selection_config,
    )

    front_raw_ttc = (
        float(target_features.front_ttc_s)
        if target_features.front_vehicle_id is not None and np.isfinite(target_features.front_ttc_s)
        else None
    )
    rear_raw_ttc = (
        float(target_features.rear_ttc_s)
        if target_features.rear_vehicle_id is not None and np.isfinite(target_features.rear_ttc_s)
        else None
    )

    return {
        "target_front_present": int(observation[2]),
        "target_front_gap": float(observation[3]),
        "target_front_relative_speed": float(observation[4]),
        "target_front_ttc_observation": float(observation[5]),
        "target_front_ttc_raw": front_raw_ttc,
        "target_rear_present": int(observation[6]),
        "target_rear_gap": float(observation[7]),
        "target_rear_relative_speed": float(observation[8]),
        "target_rear_ttc_observation": float(observation[9]),
        "target_rear_ttc_raw": rear_raw_ttc,
    }


def sweep_relative_frames(
    train_specs: List[ManeuverSpec],
    split_by_id: Dict[str, str],
    scene_key_by_id: Dict[str, str],
    merge_start_frame_by_id: Dict[str, int],
    offsets: Tuple[int, ...],
    dataset_config_path: str,
    merge_config_path: str,
) -> Tuple[List[FrameSample], Dict[str, int]]:
    expansion_config = load_dataset_config(dataset_config_path)
    lane_assignment_config = load_lane_assignment_config(merge_config_path)  # unused directly here but
    # loaded to fail fast/consistently with MergeEnvironment's own
    # constructor-time validation, kept for parity/documentation only.
    merge_topology_config = load_merge_topology_config(merge_config_path)
    agent_selection_config = load_agent_selection_config(merge_config_path)
    dynamics = waymax_dynamics.InvertibleBicycleModel(
        dt=WAYMAX_DT_S, max_accel=6.0, max_steering=0.3, normalize_actions=False
    )

    cache = _ScenarioCache(expansion_config)
    samples: List[FrameSample] = []
    out_of_range_counts: Dict[str, int] = {str(o): 0 for o in offsets}

    for i, spec in enumerate(train_specs):
        record = cache.get(spec.source_shard, spec.record_index)
        polylines_by_id = {p.lane_id: p for p in extract_lane_polylines(record.state.roadgraph_points)}
        source_polyline = polylines_by_id[spec.lane_chain[0]]
        target_polyline = polylines_by_id[spec.lane_chain[1]]
        _, merge_end_s = compute_merge_start_end_s(source_polyline, target_polyline, merge_topology_config)

        merge_start_frame = merge_start_frame_by_id[spec.maneuver_id]

        for offset in offsets:
            frame_index = merge_start_frame + offset
            result = _sample_logged_frame(
                record, spec, frame_index, polylines_by_id, merge_end_s, agent_selection_config, dynamics
            )
            if result is None:
                out_of_range_counts[str(offset)] += 1
                continue
            samples.append(
                FrameSample(
                    maneuver_id=spec.maneuver_id,
                    scene_key=scene_key_by_id[spec.maneuver_id],
                    split=split_by_id[spec.maneuver_id],
                    frame_index=frame_index,
                    relative_to_merge_start_frame=offset,
                    **result,
                )
            )
        print(f"[{i+1}/{len(train_specs)}] {spec.maneuver_id} sampled", flush=True)

    return samples, out_of_range_counts


def summarize_by_offset(samples: List[FrameSample], offsets: Tuple[int, ...]) -> List[dict]:
    rows = []
    for offset in offsets:
        offset_samples = [s for s in samples if s.relative_to_merge_start_frame == offset]
        n = len(offset_samples)
        front_present = [s for s in offset_samples if s.target_front_present == 1]
        rear_present = [s for s in offset_samples if s.target_rear_present == 1]

        front_gap_pct = _percentiles([s.target_front_gap for s in front_present])
        rear_gap_pct = _percentiles([s.target_rear_gap for s in rear_present])

        front_closing_ttc = [
            s.target_front_gap / s.target_front_relative_speed
            for s in front_present
            if s.target_front_gap > 0.0 and s.target_front_relative_speed > 0.0
        ]
        rear_closing_ttc = [
            s.target_rear_gap / s.target_rear_relative_speed
            for s in rear_present
            if s.target_rear_gap > 0.0 and s.target_rear_relative_speed > 0.0
        ]
        front_ttc_pct = _percentiles(front_closing_ttc)
        rear_ttc_pct = _percentiles(rear_closing_ttc)

        rows.append({
            "relative_frame": offset,
            "relative_time_s": offset * WAYMAX_DT_S,
            "n_samples": n,
            "front_presence_rate": len(front_present) / n if n else float("nan"),
            "rear_presence_rate": len(rear_present) / n if n else float("nan"),
            "front_gap_p10": front_gap_pct["p10"], "front_gap_p25": front_gap_pct["p25"],
            "front_gap_p50": front_gap_pct["p50"], "front_gap_p75": front_gap_pct["p75"],
            "front_gap_p90": front_gap_pct["p90"],
            "rear_gap_p10": rear_gap_pct["p10"], "rear_gap_p25": rear_gap_pct["p25"],
            "rear_gap_p50": rear_gap_pct["p50"], "rear_gap_p75": rear_gap_pct["p75"],
            "rear_gap_p90": rear_gap_pct["p90"],
            "front_ttc_closing_p10": front_ttc_pct["p10"], "front_ttc_closing_p25": front_ttc_pct["p25"],
            "front_ttc_closing_p50": front_ttc_pct["p50"], "front_ttc_closing_p75": front_ttc_pct["p75"],
            "front_ttc_closing_p90": front_ttc_pct["p90"],
            "rear_ttc_closing_p10": rear_ttc_pct["p10"], "rear_ttc_closing_p25": rear_ttc_pct["p25"],
            "rear_ttc_closing_p50": rear_ttc_pct["p50"], "rear_ttc_closing_p75": rear_ttc_pct["p75"],
            "rear_ttc_closing_p90": rear_ttc_pct["p90"],
            "n_front_closing_ttc_samples": len(front_closing_ttc),
            "n_rear_closing_ttc_samples": len(rear_closing_ttc),
        })
    return rows


def summarize_window_aggregate(samples: List[FrameSample], window_name: str, offset_predicate) -> dict:
    """PRE/EVENT/POST window aggregates. Computed at the FRAME level
    (documented explicitly, per task Section 5's instruction to
    describe the aggregation method) -- a maneuver contributing more
    in-range offsets to a window contributes more frame-level samples
    to that window's percentiles. A maneuver-level summary (one value
    per maneuver via per-maneuver median across in-window offsets) is
    ALSO provided in the report/markdown for comparison, so a reader
    can see whether frame-level aggregation is being dominated by a
    few maneuvers with more in-range offsets."""

    window_samples = [s for s in samples if offset_predicate(s.relative_to_merge_start_frame)]
    n = len(window_samples)
    front_present = [s for s in window_samples if s.target_front_present == 1]
    rear_present = [s for s in window_samples if s.target_rear_present == 1]
    front_gap_pct = _percentiles([s.target_front_gap for s in front_present])
    rear_gap_pct = _percentiles([s.target_rear_gap for s in rear_present])
    return {
        "window": window_name,
        "n_frame_samples": n,
        "n_unique_maneuvers": len({s.maneuver_id for s in window_samples}),
        "front_presence_rate": len(front_present) / n if n else float("nan"),
        "rear_presence_rate": len(rear_present) / n if n else float("nan"),
        "front_gap_median": front_gap_pct["p50"],
        "front_gap_p25": front_gap_pct["p25"],
        "front_gap_p75": front_gap_pct["p75"],
        "rear_gap_median": rear_gap_pct["p50"],
        "rear_gap_p25": rear_gap_pct["p25"],
        "rear_gap_p75": rear_gap_pct["p75"],
    }


def maneuver_level_window_summary(samples: List[FrameSample], window_name: str, offset_predicate) -> dict:
    """Maneuver-level counterpart to summarize_window_aggregate: ONE
    value per maneuver (median gap across its in-window offsets, if
    any), so a maneuver with more in-range offsets in this window does
    not get more weight than one with fewer."""

    by_maneuver: Dict[str, List[FrameSample]] = {}
    for s in samples:
        if offset_predicate(s.relative_to_merge_start_frame):
            by_maneuver.setdefault(s.maneuver_id, []).append(s)

    front_medians = []
    rear_medians = []
    for _, man_samples in by_maneuver.items():
        front_gaps = [s.target_front_gap for s in man_samples if s.target_front_present == 1]
        rear_gaps = [s.target_rear_gap for s in man_samples if s.target_rear_present == 1]
        if front_gaps:
            front_medians.append(statistics.median(front_gaps))
        if rear_gaps:
            rear_medians.append(statistics.median(rear_gaps))

    front_pct = _percentiles(front_medians)
    rear_pct = _percentiles(rear_medians)
    return {
        "window": window_name,
        "n_maneuvers_with_window_data": len(by_maneuver),
        "n_maneuvers_with_front_present": len(front_medians),
        "n_maneuvers_with_rear_present": len(rear_medians),
        "front_gap_median_of_maneuver_medians": front_pct["p50"],
        "rear_gap_median_of_maneuver_medians": rear_pct["p50"],
    }


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config-path", default=DATASET_CONFIG_PATH)
    parser.add_argument("--merge-config-path", default=DEFAULT_MERGE_CONFIG_PATH)
    parser.add_argument("--difficulty-csv-path", default=DIFFICULTY_CSV_PATH)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--relative-frame-offsets", default=",".join(str(o) for o in DEFAULT_RELATIVE_FRAME_OFFSETS),
        help="Comma-separated relative frame offsets around merge_start_frame.",
    )
    parser.add_argument("--skip-event-sweep", action="store_true",
                         help="Skip Section 5 (merge-event frame sweep) -- much faster, Sections 3/4 only.")
    return parser.parse_args()


def _fmt(x) -> str:
    if isinstance(x, float):
        if x != x:  # NaN
            return "nan"
        return f"{x:.3f}"
    return str(x)


def print_table(headers: List[str], rows: List[List]) -> None:
    widths = [max(len(str(h)), *(len(_fmt(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(_fmt(v).ljust(w) for v, w in zip(r, widths)))


def main() -> None:
    args = parse_args()
    offsets = tuple(int(o.strip()) for o in args.relative_frame_offsets.split(",") if o.strip())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAIN Gap/TTC Distribution Audit (Reward V1 design input)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Section 0: TRAIN split audit
    # ------------------------------------------------------------------
    split_rows = load_split_manifest()
    train_rows = [r for r in split_rows if r.split == "train"]
    validation_rows = [r for r in split_rows if r.split == "validation"]
    print(f"\nCanonical split manifest: data/manifests/phase2_dataset_split.csv")
    print(f"TRAIN maneuvers: {len(train_rows)} | VALIDATION maneuvers: {len(validation_rows)}")
    print(f"TRAIN unique scenes: {len({r.scene_key for r in train_rows})}")
    print(f"VALIDATION unique scenes: {len({r.scene_key for r in validation_rows})}")
    overlap = {r.scene_key for r in train_rows} & {r.scene_key for r in validation_rows}
    print(f"Scene overlap between TRAIN/VALIDATION: {len(overlap)} (must be 0)")
    if overlap:
        raise RuntimeError(f"Scene leakage detected between TRAIN/VALIDATION: {overlap}")

    split_by_id = {r.maneuver_id: r.split for r in split_rows}
    scene_key_by_id = {r.maneuver_id: r.scene_key for r in split_rows}

    # ------------------------------------------------------------------
    # Section 3/4: decision_start_frame snapshot from the EXISTING
    # descriptor CSV.
    # ------------------------------------------------------------------
    print(f"\nLoading existing descriptor table: {args.difficulty_csv_path}")
    difficulty_rows = load_train_difficulty_rows(args.difficulty_csv_path)
    print(f"TRAIN rows in descriptor table: {len(difficulty_rows)} (cross-checked against canonical split)")

    # NaN/inf sanity check on the source CSV before computing anything.
    numeric_cols = [
        "target_front_gap", "target_front_relative_speed", "target_front_ttc",
        "target_rear_gap", "target_rear_relative_speed", "target_rear_ttc",
    ]
    n_nonfinite = 0
    for row in difficulty_rows:
        for col in numeric_cols:
            v = float(row[col])
            if not np.isfinite(v):
                n_nonfinite += 1
    print(f"Non-finite (NaN/inf) values found in descriptor TRAIN numeric columns: {n_nonfinite} (must be 0)")
    if n_nonfinite:
        raise RuntimeError("Descriptor CSV contains non-finite values in a supposedly-finite observation field.")

    decision_start_summaries = analyze_decision_start_distribution(difficulty_rows)
    closing_only_summaries = analyze_closing_only_ttc(difficulty_rows)
    gap_buckets = analyze_gap_threshold_buckets(difficulty_rows)
    ttc_buckets = analyze_ttc_threshold_buckets(difficulty_rows)
    presence_closing_overlap = analyze_presence_closing_overlap(difficulty_rows)

    print("\n--- Section 3: decision_start_frame distribution (policy-observation values) ---")
    headers = ["Metric", "N", "Mean", "Std", "Min", "P05", "P10", "P25", "P50", "P75", "P90", "P95", "Max"]
    table_rows = []
    for key in ("target_front_gap", "target_rear_gap"):
        s = decision_start_summaries[key]
        table_rows.append([s.variable, s.n_valid_samples, s.mean, s.std, s.min, s.p05, s.p10, s.p25, s.p50, s.p75, s.p90, s.p95, s.max])
    for key in ("target_front_ttc_observation", "target_rear_ttc_observation"):
        s = decision_start_summaries[key]
        table_rows.append([s.variable, s.n_valid_samples, s.mean, s.std, s.min, s.p05, s.p10, s.p25, s.p50, s.p75, s.p90, s.p95, s.max])
    print_table(headers, table_rows)

    print("\n--- Section 4[2]: Closing-only TTC (present + closing_speed>0 + gap>0) ---")
    table_rows2 = []
    for key in ("target_front_ttc_closing_only", "target_rear_ttc_closing_only"):
        s = closing_only_summaries[key]
        table_rows2.append([s.variable, s.n_valid_samples, s.mean, s.std, s.min, s.p05, s.p10, s.p25, s.p50, s.p75, s.p90, s.p95, s.max])
    print_table(headers, table_rows2)

    print("\n--- Section 4: Presence / Closing / Overlap classification ---")
    for slot, counts in presence_closing_overlap.items():
        print(f"\n{slot}:")
        for label, (c, rate) in counts.items():
            print(f"  {label:24s} {c:4d}  ({rate:.1%})")

    print("\n--- Gap threshold buckets (present-only) ---")
    for slot, buckets in gap_buckets.items():
        print(f"\n{slot} (n_present={buckets['n_present_samples']}):")
        for label, value in buckets.items():
            if label == "n_present_samples":
                continue
            c, rate = value
            print(f"  {label:10s} {c:4d}  ({rate:.1%})")

    print("\n--- TTC threshold buckets (policy-observation, all rows) ---")
    for slot, buckets in ttc_buckets.items():
        print(f"\n{slot} (n={buckets['n_samples']}):")
        for label, value in buckets.items():
            if label == "n_samples":
                continue
            c, rate = value
            print(f"  {label:16s} {c:4d}  ({rate:.1%})")

    # ------------------------------------------------------------------
    # Section 5: merge_start_frame sweep.
    # ------------------------------------------------------------------
    event_offset_rows: List[dict] = []
    samples: List[FrameSample] = []
    window_summaries: List[dict] = []
    maneuver_window_summaries: List[dict] = []

    if not args.skip_event_sweep:
        print(f"\n--- Section 5: merge_start_frame +/- N sweep (dt={WAYMAX_DT_S}s) ---")
        print(f"Offsets (frames): {offsets}")
        print(f"Offsets (seconds): {[o * WAYMAX_DT_S for o in offsets]}")

        train_specs = sorted(load_maneuver_specs("train"), key=lambda s: s.maneuver_id)
        merge_start_frame_by_id = {s.maneuver_id: s.merge_start_frame for s in train_specs}

        samples, out_of_range = sweep_relative_frames(
            train_specs=train_specs,
            split_by_id=split_by_id,
            scene_key_by_id=scene_key_by_id,
            merge_start_frame_by_id=merge_start_frame_by_id,
            offsets=offsets,
            dataset_config_path=args.dataset_config_path,
            merge_config_path=args.merge_config_path,
        )
        print(f"\nOut-of-range/invalid-ego samples excluded, per offset: {out_of_range}")

        event_offset_rows = summarize_by_offset(samples, offsets)
        print("\nPer-offset summary:")
        headers5 = ["rel_frame", "rel_time_s", "N", "front_pres", "rear_pres",
                    "front_gap_p50", "rear_gap_p50", "front_ttc_c_p50", "rear_ttc_c_p50"]
        rows5 = [[r["relative_frame"], r["relative_time_s"], r["n_samples"],
                  r["front_presence_rate"], r["rear_presence_rate"],
                  r["front_gap_p50"], r["rear_gap_p50"],
                  r["front_ttc_closing_p50"], r["rear_ttc_closing_p50"]] for r in event_offset_rows]
        print_table(headers5, rows5)

        pre = summarize_window_aggregate(samples, "PRE(-20..-1)", lambda o: -20 <= o <= -1)
        event = summarize_window_aggregate(samples, "EVENT(0)", lambda o: o == 0)
        post = summarize_window_aggregate(samples, "POST(+1..+20)", lambda o: 1 <= o <= 20)
        window_summaries = [pre, event, post]

        print("\nWindow aggregates (FRAME-level):")
        headers_w = ["window", "n_frame_samples", "n_unique_maneuvers", "front_presence_rate",
                     "rear_presence_rate", "front_gap_median", "rear_gap_median"]
        rows_w = [[w[h] for h in headers_w] for w in window_summaries]
        print_table(headers_w, rows_w)

        pre_m = maneuver_level_window_summary(samples, "PRE(-20..-1)", lambda o: -20 <= o <= -1)
        event_m = maneuver_level_window_summary(samples, "EVENT(0)", lambda o: o == 0)
        post_m = maneuver_level_window_summary(samples, "POST(+1..+20)", lambda o: 1 <= o <= 20)
        maneuver_window_summaries = [pre_m, event_m, post_m]

        print("\nWindow aggregates (MANEUVER-level, one median per maneuver):")
        headers_wm = ["window", "n_maneuvers_with_window_data", "n_maneuvers_with_front_present",
                      "front_gap_median_of_maneuver_medians", "rear_gap_median_of_maneuver_medians"]
        rows_wm = [[w[h] for h in headers_wm] for w in maneuver_window_summaries]
        print_table(headers_wm, rows_wm)
    else:
        print("\n--skip-event-sweep set; Section 5 skipped.")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    _write_outputs(
        output_dir, decision_start_summaries, closing_only_summaries,
        gap_buckets, ttc_buckets, presence_closing_overlap,
        event_offset_rows, samples, window_summaries, maneuver_window_summaries,
        offsets, len(train_rows), len(difficulty_rows),
    )
    print(f"\nWrote outputs to {output_dir}/")


def _write_outputs(
    output_dir, decision_start_summaries, closing_only_summaries,
    gap_buckets, ttc_buckets, presence_closing_overlap,
    event_offset_rows, samples, window_summaries, maneuver_window_summaries,
    offsets, n_train_maneuvers, n_difficulty_rows,
):
    # train_gap_ttc_summary.csv
    summary_path = output_dir / "train_gap_ttc_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "variable", "n_total_maneuvers", "n_present", "n_absent", "presence_rate",
            "n_valid_samples", "mean", "std", "min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in list(decision_start_summaries.values()) + list(closing_only_summaries.values()):
            writer.writerow(dataclasses.asdict(s))

    # train_gap_ttc_by_event_offset.csv
    if event_offset_rows:
        offset_path = output_dir / "train_gap_ttc_by_event_offset.csv"
        with open(offset_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_offset_rows[0].keys()))
            writer.writeheader()
            writer.writerows(event_offset_rows)

    # train_gap_ttc_samples.csv (raw per-frame samples from the event sweep)
    if samples:
        samples_path = output_dir / "train_gap_ttc_samples.csv"
        with open(samples_path, "w", newline="") as f:
            fieldnames = [f.name for f in dataclasses.fields(FrameSample)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow(dataclasses.asdict(s))

    # train_gap_ttc_report.md
    report_path = output_dir / "train_gap_ttc_report.md"
    with open(report_path, "w") as f:
        f.write("# TRAIN Gap/TTC Distribution Audit\n\n")
        f.write(f"TRAIN maneuvers: {n_train_maneuvers}. Descriptor rows used (decision_start_frame snapshot): {n_difficulty_rows}.\n\n")
        f.write("## Decision-start distribution\n\n")
        f.write("| Metric | N | Mean | Std | Min | P05 | P10 | P25 | P50 | P75 | P90 | P95 | Max |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in decision_start_summaries.values():
            f.write(f"| {s.variable} | {s.n_valid_samples} | {s.mean:.2f} | {s.std:.2f} | {s.min:.2f} | "
                    f"{s.p05:.2f} | {s.p10:.2f} | {s.p25:.2f} | {s.p50:.2f} | {s.p75:.2f} | {s.p90:.2f} | {s.p95:.2f} | {s.max:.2f} |\n")
        f.write("\n## Closing-only TTC\n\n")
        f.write("| Metric | N | Mean | Std | Min | P05 | P10 | P25 | P50 | P75 | P90 | P95 | Max |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in closing_only_summaries.values():
            f.write(f"| {s.variable} | {s.n_valid_samples} | {s.mean:.2f} | {s.std:.2f} | {s.min:.2f} | "
                    f"{s.p05:.2f} | {s.p10:.2f} | {s.p25:.2f} | {s.p50:.2f} | {s.p75:.2f} | {s.p90:.2f} | {s.p95:.2f} | {s.max:.2f} |\n")
        if event_offset_rows:
            f.write("\n## Merge-event offset sweep\n\n")
            f.write(f"dt = {WAYMAX_DT_S}s. Offsets: {list(offsets)}\n\n")
            f.write("| rel_frame | rel_time_s | N | front_pres | rear_pres | front_gap_p50 | rear_gap_p50 | front_ttc_closing_p50 | rear_ttc_closing_p50 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in event_offset_rows:
                f.write(f"| {r['relative_frame']} | {r['relative_time_s']} | {r['n_samples']} | "
                        f"{r['front_presence_rate']:.1%} | {r['rear_presence_rate']:.1%} | "
                        f"{r['front_gap_p50']:.2f} | {r['rear_gap_p50']:.2f} | "
                        f"{r['front_ttc_closing_p50']:.2f} | {r['rear_ttc_closing_p50']:.2f} |\n")
            f.write("\n### Window aggregates (frame-level)\n\n")
            f.write("| window | n_frame_samples | n_unique_maneuvers | front_presence_rate | rear_presence_rate | front_gap_median | rear_gap_median |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for w in window_summaries:
                f.write(f"| {w['window']} | {w['n_frame_samples']} | {w['n_unique_maneuvers']} | "
                        f"{w['front_presence_rate']:.1%} | {w['rear_presence_rate']:.1%} | "
                        f"{w['front_gap_median']:.2f} | {w['rear_gap_median']:.2f} |\n")
            f.write("\n### Window aggregates (maneuver-level)\n\n")
            f.write("| window | n_maneuvers | n_with_front_present | front_gap_median_of_medians | rear_gap_median_of_medians |\n")
            f.write("|---|---|---|---|---|\n")
            for w in maneuver_window_summaries:
                f.write(f"| {w['window']} | {w['n_maneuvers_with_window_data']} | {w['n_maneuvers_with_front_present']} | "
                        f"{w['front_gap_median_of_maneuver_medians']:.2f} | {w['rear_gap_median_of_maneuver_medians']:.2f} |\n")


if __name__ == "__main__":
    main()
