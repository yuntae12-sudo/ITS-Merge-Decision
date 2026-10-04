#!/usr/bin/env python3
"""Real WOMD top view of an ego merging into a front--rear gap.

The selected maneuver is an ACCEPT row in the repository's v2 evidence.
At the plotted frame, ego is on the converging entry lane (177), Front is
on the downstream main lane (250), and Rear is on its upstream continuation
(245).  Lanes 245 and 250 are one topology-connected target-lane route;
lane 177 is the other entry into lane 250.  Nothing in this figure is
synthetic: map polylines, poses, headings, and vehicle dimensions are read
from the original WOMD Scenario protobuf.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scenarios.scenario_proto_loader import iter_scenario_protobufs
from src.visualization import style


SCENARIO_ID = "4478de443439b5e4"
MANEUVER_ID = "training_tfexample.tfrecord-00006-of-01000#45__t70__177_250"
PROTO_PATH = REPO_ROOT / "data/womd/scenario_proto/training/training.tfrecord-00512-of-01000"
PROTO_RECORD_INDEX = 445
EVIDENCE_PATH = REPO_ROOT / "data/manifests/evidence_training.jsonl"

TIMESTEP = 50
EGO_ID = 343
FRONT_ID = 0
REAR_ID = 1
MERGE_LANE_ID = 177
MAIN_IN_LANE_ID = 245
MAIN_OUT_LANE_ID = 250

OUTPUT_DIR = REPO_ROOT / "outputs/capstone_midterm_figures"
OUTPUT_STEM = OUTPUT_DIR / "fig_merge_topview_front_rear_gap"


def _load_selected_scenario():
    for index, scenario in enumerate(iter_scenario_protobufs([str(PROTO_PATH)])):
        if index == PROTO_RECORD_INDEX:
            if scenario.scenario_id != SCENARIO_ID:
                raise RuntimeError(
                    f"protobuf locator mismatch: expected {SCENARIO_ID}, got {scenario.scenario_id}"
                )
            return scenario
    raise RuntimeError(f"record {PROTO_RECORD_INDEX} is absent from {PROTO_PATH}")


def _load_evidence() -> dict:
    with EVIDENCE_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("maneuver_id") == MANEUVER_ID:
                return row
    raise RuntimeError(f"maneuver is absent from {EVIDENCE_PATH}")


def _lane_xy(map_features: dict[int, object], lane_id: int) -> np.ndarray:
    lane = map_features[lane_id].lane
    return np.asarray([(p.x, p.y) for p in lane.polyline], dtype=np.float64)


def _distance_to_polyline(xy: np.ndarray, point: np.ndarray) -> float:
    starts = xy[:-1]
    vectors = xy[1:] - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    t = np.einsum("ij,ij->i", point - starts, vectors) / np.maximum(lengths_sq, 1e-12)
    projections = starts + np.clip(t, 0.0, 1.0)[:, None] * vectors
    return float(np.min(np.linalg.norm(point - projections, axis=1)))


def _xy(polyline) -> np.ndarray:
    return np.asarray([(p.x, p.y) for p in polyline], dtype=np.float64)


def _verify_selection(scenario, evidence: dict, map_features: dict[int, object], tracks: dict[int, object]):
    topology = evidence["topology_evidence"]
    interaction = evidence["interaction_evidence"]
    assert evidence["automatic_decision"] == "accept"
    assert int(scenario.tracks[scenario.sdc_track_index].id) == EGO_ID
    assert topology["source_lane_id"] == MERGE_LANE_ID
    assert topology["target_lane_id"] == MAIN_OUT_LANE_ID
    assert topology["source_exits_to_target"] is True
    assert set(topology["target_entry_lane_ids"]) == {MERGE_LANE_ID, MAIN_IN_LANE_ID}
    assert interaction["front_vehicle_id"] == FRONT_ID
    assert interaction["rear_vehicle_id"] == REAR_ID
    assert interaction["gap_order_valid_at_entry"] is True
    assert interaction["gap_order_persistence_frames"] >= interaction[
        "required_gap_order_persistence_frames"
    ]

    assert MAIN_OUT_LANE_ID in map_features[MERGE_LANE_ID].lane.exit_lanes
    assert MAIN_OUT_LANE_ID in map_features[MAIN_IN_LANE_ID].lane.exit_lanes
    for object_id in (EGO_ID, FRONT_ID, REAR_ID):
        assert tracks[object_id].states[TIMESTEP].valid

    # Strict geometric cross-check on the original protobuf.  The generous
    # 3.0 m ego threshold accounts for its center being between sparse merge
    # centerline samples; the whole vehicle still occupies the merge branch.
    assignments = (
        (EGO_ID, MERGE_LANE_ID, 3.0),
        (FRONT_ID, MAIN_OUT_LANE_ID, 1.0),
        (REAR_ID, MAIN_IN_LANE_ID, 1.0),
    )
    for object_id, lane_id, maximum_distance_m in assignments:
        state = tracks[object_id].states[TIMESTEP]
        distance_m = _distance_to_polyline(
            _lane_xy(map_features, lane_id),
            np.asarray([state.center_x, state.center_y]),
        )
        if distance_m > maximum_distance_m:
            raise RuntimeError(
                f"object {object_id} is {distance_m:.2f} m from lane {lane_id}; "
                f"expected <= {maximum_distance_m:.2f} m"
            )


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    scenario = _load_selected_scenario()
    evidence = _load_evidence()
    map_features = {int(feature.id): feature for feature in scenario.map_features}
    tracks = {int(track.id): track for track in scenario.tracks}
    _verify_selection(scenario, evidence, map_features, tracks)

    merge_xy = _lane_xy(map_features, MERGE_LANE_ID)
    main_in_xy = _lane_xy(map_features, MAIN_IN_LANE_ID)
    main_out_xy = _lane_xy(map_features, MAIN_OUT_LANE_ID)
    origin = main_out_xy[0]

    # A translation only: geometry and north-up orientation remain unchanged.
    merge_xy = merge_xy - origin
    main_in_xy = main_in_xy - origin
    main_out_xy = main_out_xy - origin
    xlim = (-34.0, 22.0)
    ylim = (-39.0, 54.0)

    style.apply_paper_style()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(4.6, 6.0))

    # Real WOMD map context, kept deliberately quiet.
    for feature in scenario.map_features:
        polyline = None
        if feature.HasField("lane"):
            polyline = feature.lane.polyline
        elif feature.HasField("road_line"):
            polyline = feature.road_line.polyline
        elif feature.HasField("road_edge"):
            polyline = feature.road_edge.polyline
        if polyline is None or len(polyline) < 2:
            continue
        points = _xy(polyline) - origin
        if (
            np.max(points[:, 0]) < xlim[0]
            or np.min(points[:, 0]) > xlim[1]
            or np.max(points[:, 1]) < ylim[0]
            or np.min(points[:, 1]) > ylim[1]
        ):
            continue
        ax.plot(points[:, 0], points[:, 1], color="#D8D8D8", linewidth=0.65, zorder=0)

    ax.plot(
        main_in_xy[:, 0], main_in_xy[:, 1], color="#4C72B0", linewidth=2.4,
        label="Main lane", zorder=2,
    )
    ax.plot(main_out_xy[:, 0], main_out_xy[:, 1], color="#4C72B0", linewidth=2.4, zorder=2)
    ax.plot(
        merge_xy[:, 0], merge_xy[:, 1], color="#DD8452", linewidth=2.4,
        linestyle="--", label="Merge lane", zorder=2,
    )

    # All non-selected real vehicles are gray.
    other_label_used = False
    for object_id, track in tracks.items():
        state = track.states[TIMESTEP]
        if not state.valid or int(track.object_type) != 1 or object_id in {EGO_ID, FRONT_ID, REAR_ID}:
            continue
        x, y = np.asarray([state.center_x, state.center_y]) - origin
        if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
            continue
        style.draw_oriented_box(
            ax, x, y, state.heading, state.length, state.width,
            color="#B8B8B8", alpha=0.82, zorder=3,
            label="Other vehicles" if not other_label_used else None,
        )
        other_label_used = True

    selected = (
        (EGO_ID, style.COLOR_EGO, "Ego", "E"),
        (FRONT_ID, style.COLOR_TARGET_FRONT, "Front", "F"),
        (REAR_ID, style.COLOR_TARGET_REAR, "Rear", "R"),
    )
    for object_id, color, label, short_label in selected:
        state = tracks[object_id].states[TIMESTEP]
        x, y = np.asarray([state.center_x, state.center_y]) - origin
        style.draw_oriented_box(
            ax, x, y, state.heading, state.length, state.width,
            color=color, zorder=5, label=label,
        )
        ax.text(
            x + 2.6, y, short_label, color=color, fontsize=7, fontweight="bold",
            ha="left", va="center", zorder=7,
        )

    ax.scatter([0.0], [0.0], s=13, color="#333333", zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East relative to merge point (m)")
    ax.set_ylabel("North relative to merge point (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style.savefig_paper(fig, str(OUTPUT_STEM))
    plt.close(fig)

    print(f"scenario_id: {SCENARIO_ID}")
    print(f"maneuver_id: {MANEUVER_ID}; timestep: {TIMESTEP}")
    print(f"ego/front/rear: {EGO_ID}/{FRONT_ID}/{REAR_ID}")
    print(f"wrote: {OUTPUT_STEM}.png and {OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
