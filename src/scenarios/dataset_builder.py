"""Builds the flat merge-candidate dataset (Compressed Commit E).

This module turns per-scene ``LaneTransition`` + ``MergeDiagnostic`` +
(for ACCEPT) ``InteractionFeatures`` results (Commits C/D) into one flat,
serializable row per transition -- a ``CandidateRecord`` -- so the whole
scanned dataset can be written to a single CSV manifest
(``scripts/extract_merge_scenes.py``) and later manually reviewed
(``scripts/build_merge_manifest.py``) and re-rendered for validation
(``scripts/render_merge_validation.py`` / ``validation_viz.py``).

This module does NOT change any merge-detection or feature-extraction
logic -- it only calls into ``lane_assignment.py``, ``merge_detector.py``,
and ``scenario_features.py`` using the exact same call pattern already
used by ``scripts/inspect_merge_candidate.py``, and flattens the results.

Serialization convention (CSV)
-------------------------------
- Missing / not-applicable values (e.g. REJECT/REVIEW rows have no
  interaction features; some diagnostic fields are None for early
  -rejected transitions) are written as an empty string.
- ``float('inf')`` (non-closing TTC) is written as the literal string
  ``inf`` -- Python's csv writer stringifies a float via ``str()``,
  and ``str(float('inf')) == 'inf'``, so this is Python's native
  behavior, not a special case coded here.
- No NaN and no the string "null"/"None" is ever intentionally written;
  every optional field is either a real value, `inf` (TTC only), or
  blank.

Determinism
------------
Candidates are always sorted by
``(source_split, source_shard, record_index, transition_frame,
source_lane_id, target_lane_id)`` before being written or summarized,
so re-running the scan over the same data produces byte-identical
output (module the directory/timestamps). ``source_split`` is included
even though one dataset-expansion config only ever has a single split
value, for clarity/robustness if a combined multi-split CSV is ever
assembled by hand.

Manifest materialization (fix commit)
---------------------------------------
``materialize_merge_features`` and ``reconstruct_transition`` are also
called by ``scripts/build_merge_manifest.py`` to fully populate the
ACCEPT-only fields for any manually-CONFIRMED_MERGE candidate whose
original detector decision was REVIEW or REJECT (a human confirming a
non-ACCEPT candidate as a genuine merge must still produce a complete,
usable final-manifest row -- see that script's module docstring).

Three distinct frame concepts (fix commit "materialize merge state at
pre-merge reference frame")
----------------------------------------------------------------------
This module -- and the CandidateRecord schema -- deliberately keeps
THREE separate frame fields, never overwriting one with another:

  - ``transition_frame`` (``LaneTransition.transition_frame``): where
    the stable target-lane run begins. This is the detector's own
    anchor and is used, completely unchanged, for every
    ``merge_detector.detect_merge`` gate exactly as before this fix
    (including the source-lane-arc-length input the detector itself
    computes at this frame).
  - ``merge_start_frame`` (``_derive_merge_frames``): the first valid
    ego frame where ego's source-lane arc length reaches
    ``merge_start_s`` -- a geometry-inferred "merge region begins"
    point, offline-derived exactly as before this fix.
  - ``feature_reference_frame`` (NEW, ``_resolve_feature_reference_frame``):
    the single frame at which ALL decision-state features (ego speed,
    d_m, front/rear gap/relative-speed/TTC, traffic density) are
    JOINTLY sampled. Preferred policy: equal to ``merge_start_frame``
    when valid (see ``_resolve_feature_reference_frame``); before this
    fix, ``materialize_merge_features`` sampled everything at
    ``transition_frame`` instead, which is at/near merge completion
    rather than a pre-merge decision point -- the root cause this
    fix addresses (54/56 real ACCEPT rows previously showed
    merge_distance_m < 1m as a direct result).

Offline/online causality note: using logged future trajectory to
locate ``merge_start_frame``/``feature_reference_frame`` is legitimate
for OFFLINE dataset construction (Phase 1), but a future online
PPO/FSM policy (Phase 3) must compute its own state causally from
information available at-or-before the current simulation time --
this offline reference-frame selection must never be read as a
template for how an online policy would pick "when to look".
"""

import csv
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.scenarios.lane_assignment import (
    LaneAssignmentConfig,
    LaneTransition,
    assign_ego_lane_sequence,
    compute_stable_lane_sequence,
    find_lane_transitions,
)
from src.scenarios.lane_geometry import extract_lane_polylines, project_point_to_polyline
from src.scenarios.merge_detector import (
    MergeDecision,
    MergeDiagnostic,
    MergeTopologyConfig,
    compute_merge_start_end_s,
    compute_remaining_merge_distance,
    detect_merge,
)
from src.scenarios.scenario_features import (
    AgentSelectionConfig,
    extract_interaction_features,
)
from src.scenarios.scenario_loader import ScenarioRecord

# Historical default (pre-multi-shard): every scenario this study has
# scanned so far comes from the Waymo Open Motion Dataset. Multi-shard
# dataset expansion configs now carry their own ``dataset_name`` field
# (default "WOMD" -- see scenario_loader.DatasetExpansionConfig /
# load_dataset_config), and each ScenarioRecord's ``source_dataset`` is
# populated from that at load time. This module constant is kept only
# as the fallback used by ``build_candidate_records`` for records that
# predate the ``source_dataset`` field (defensive default, not a
# second independently-maintained idea of what "WOMD" is -- it must
# equal the loader's own default).
SOURCE_DATASET = "WOMD"

# CSV field order -- the authoritative schema for merge_candidates.csv.
CANDIDATE_FIELDS = [
    "candidate_id",
    "scene_key",
    "source_dataset",
    "source_split",
    "source_shard",
    "record_index",
    "transition_index",
    "transition_frame",
    "source_lane_id",
    "target_lane_id",
    "source_start_frame",
    "source_end_frame",
    "target_start_frame",
    "target_end_frame",
    "decision",
    "reason",
    # MergeDiagnostic fields (all decisions).
    "source_lane_ends",
    "source_remaining_distance_m",
    "endpoint_target_distance_m",
    "endpoint_target_arc_length_m",
    "heading_difference_deg",
    "lanes_converge",
    "parallel_continuation",
    "separation_reduction_m",
    "decreasing_fraction",
    "max_collinear_offset_m",
    "upstream_separation_m",
    "pre_merge_frames",
    "target_lane_persistent",
    # ACCEPT-only fields (blank otherwise).
    "merge_start_s",
    "merge_end_s",
    "merge_start_frame",
    "merge_complete_frame",
    # Feature-reference-frame schema (fix commit "materialize merge
    # state at pre-merge reference frame") -- see module docstring
    # "Three distinct frame concepts". Populated for every ACCEPT row
    # (both valid and invalid references), never blank there.
    "feature_reference_frame",
    "feature_reference_policy",
    "feature_reference_valid",
    "feature_reference_reason",
    "ego_longitudinal_speed_mps",
    "merge_distance_m",
    "front_vehicle_id",
    "front_gap_m",
    "front_relative_speed_mps",
    "front_ttc_s",
    "rear_vehicle_id",
    "rear_gap_m",
    "rear_relative_speed_mps",
    "rear_ttc_s",
    "traffic_density",
]

# The only feature-reference policy currently implemented (recorded on
# every ACCEPT CandidateRecord even when the reference turns out to be
# invalid -- it names which policy was ATTEMPTED, not whether it
# succeeded).
FEATURE_REFERENCE_POLICY_MERGE_START_FRAME = "merge_start_frame"

# feature_reference_reason values (see _resolve_feature_reference_frame).
FEATURE_REFERENCE_REASON_UNAVAILABLE = "merge_start_frame_unavailable"
FEATURE_REFERENCE_REASON_INVALID_EGO_FRAME = "merge_start_frame_invalid_ego_frame"
FEATURE_REFERENCE_REASON_AFTER_TRANSITION_FRAME = (
    "merge_start_frame_after_transition_frame"
)


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
    """One flattened merge-transition candidate row."""

    candidate_id: str
    scene_key: str
    source_dataset: str
    source_split: str
    source_shard: str
    record_index: int
    transition_index: int
    transition_frame: int
    source_lane_id: int
    target_lane_id: int
    source_start_frame: int
    source_end_frame: int
    target_start_frame: int
    target_end_frame: int
    decision: str
    reason: Optional[str]

    source_lane_ends: bool
    source_remaining_distance_m: Optional[float]
    endpoint_target_distance_m: Optional[float]
    endpoint_target_arc_length_m: Optional[float]
    heading_difference_deg: Optional[float]
    lanes_converge: bool
    parallel_continuation: bool
    separation_reduction_m: Optional[float]
    decreasing_fraction: Optional[float]
    max_collinear_offset_m: Optional[float]
    upstream_separation_m: Optional[float]
    pre_merge_frames: int
    target_lane_persistent: bool

    merge_start_s: Optional[float] = None
    merge_end_s: Optional[float] = None
    merge_start_frame: Optional[int] = None
    merge_complete_frame: Optional[int] = None
    feature_reference_frame: Optional[int] = None
    feature_reference_policy: Optional[str] = None
    feature_reference_valid: Optional[bool] = None
    feature_reference_reason: Optional[str] = None
    ego_longitudinal_speed_mps: Optional[float] = None
    merge_distance_m: Optional[float] = None
    front_vehicle_id: Optional[int] = None
    front_gap_m: Optional[float] = None
    front_relative_speed_mps: Optional[float] = None
    front_ttc_s: Optional[float] = None
    rear_vehicle_id: Optional[int] = None
    rear_gap_m: Optional[float] = None
    rear_relative_speed_mps: Optional[float] = None
    rear_ttc_s: Optional[float] = None
    traffic_density: Optional[int] = None


def make_candidate_id(
    scene_key: str,
    transition_frame: int,
    source_lane_id: int,
    target_lane_id: int,
) -> str:
    """Builds the full-reproducibility candidate id.

    Format: ``<scene_key>__t<transition_frame>__<source>_<target>``.
    ``scene_key`` already contains ``#`` (e.g.
    ``validation_tfexample.tfrecord-00000-of-00150#28``) -- that is
    fine for this id (a CSV column value), but NOT fine as a bare
    filename; see ``sanitize_candidate_id_for_filename``.
    """

    return f"{scene_key}__t{transition_frame}__{source_lane_id}_{target_lane_id}"


def sanitize_candidate_id_for_filename(candidate_id: str) -> str:
    """Sanitizes a candidate_id for safe use as a filename component.

    Replaces characters that are meaningful to filesystems/paths
    (``#``, ``/``, ``:``) with ``_``. The CSV ``candidate_id`` column
    itself is never sanitized -- only this derived filename form.
    """

    sanitized = candidate_id
    for char in ("#", "/", ":"):
        sanitized = sanitized.replace(char, "_")
    return sanitized


def _derive_merge_frames(
    transition: LaneTransition,
    source_polyline,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_valid: np.ndarray,
    merge_start_s: float,
    merge_end_s: float,
) -> Tuple[Optional[int], Optional[int]]:
    """Derives merge_start_frame / merge_complete_frame for an ACCEPT
    candidate.

    Definition (documented per task instructions -- keep simple, don't
    guess past what's robust):
        - Walk ego's logged trajectory frames from
          ``transition.source_start_frame`` to
          ``transition.target_end_frame`` (inclusive).
        - At each VALID frame, project ego's (x, y) onto the SOURCE
          polyline to get its source-lane arc length.
        - merge_start_frame := the first such frame where that arc
          length >= merge_start_s.
        - merge_complete_frame := transition.transition_frame itself
          (this is already how Commit C/D define "ego confirmed onto
          the target lane" -- the frame the stable target run begins).

    If no valid frame in the window ever reaches merge_start_s (e.g.
    all projections are missing/invalid), merge_start_frame is left
    None rather than guessing; merge_complete_frame is always
    transition.transition_frame for an ACCEPT candidate (well-defined
    by construction), so only merge_start_frame can come back None.
    """

    merge_complete_frame = transition.transition_frame

    start = transition.source_start_frame
    end = transition.target_end_frame

    merge_start_frame = None
    for frame in range(start, end + 1):
        if frame >= ego_valid.shape[0] or not ego_valid[frame]:
            continue
        projection = project_point_to_polyline(
            source_polyline, float(ego_x[frame]), float(ego_y[frame])
        )
        if projection["arc_length_m"] >= merge_start_s:
            merge_start_frame = frame
            break

    return merge_start_frame, merge_complete_frame


def _resolve_feature_reference_frame(
    transition: LaneTransition,
    merge_start_frame: Optional[int],
    ego_valid: np.ndarray,
) -> Tuple[Optional[int], bool, Optional[str]]:
    """Resolves the single frame at which ALL decision-state features
    (ego speed, d_m, front/rear gap/relative-speed/TTC, traffic
    density) are jointly sampled (see module docstring "Three distinct
    frame concepts").

    Policy: ``feature_reference_frame = merge_start_frame`` when valid.
    Valid iff ALL of:
        - ``merge_start_frame is not None``
        - it is a valid index into ``ego_valid``
          (``0 <= merge_start_frame < len(ego_valid)``) AND
          ``ego_valid[merge_start_frame]`` is True
        - ``merge_start_frame <= transition.transition_frame`` (the
          reference must be at-or-before merge completion, never
          after)

    No silent fallback: when invalid, returns ``(None, False, reason)``
    -- callers must leave the decision-state feature fields blank
    rather than falling back to ``transition_frame``.

    Returns:
        (feature_reference_frame_or_None, valid, reason_or_None)
    """

    if merge_start_frame is None:
        return None, False, FEATURE_REFERENCE_REASON_UNAVAILABLE

    if not (0 <= merge_start_frame < ego_valid.shape[0]) or not ego_valid[
        merge_start_frame
    ]:
        return None, False, FEATURE_REFERENCE_REASON_INVALID_EGO_FRAME

    if merge_start_frame > transition.transition_frame:
        return None, False, FEATURE_REFERENCE_REASON_AFTER_TRANSITION_FRAME

    return merge_start_frame, True, None


def materialize_merge_features(
    transition: LaneTransition,
    source_polyline,
    target_polyline,
    ego_source_arc_length_m: Optional[float],
    record: ScenarioRecord,
    merge_topology_config: MergeTopologyConfig,
    agent_selection_config: AgentSelectionConfig,
) -> dict:
    """Computes the full ACCEPT-only feature set for one transition.

    This is the single source of truth for the ACCEPT-only
    ``CandidateRecord`` fields (``merge_start_s``, ``merge_end_s``,
    ``merge_start_frame``, ``merge_complete_frame``,
    ``ego_longitudinal_speed_mps``, ``merge_distance_m``, front/rear
    vehicle_id/gap/relative_speed/ttc, ``traffic_density``). It was
    extracted from ``_build_candidate_records_unsafe`` (the detector
    -ACCEPT dataset-scan path) so the exact same computation can also
    be reused by manual-CONFIRMED_MERGE final-manifest materialization
    (fix commit "materialize manually confirmed merge features") --
    no duplicated feature-computation logic between the two callers.

    Args:
        transition: the (possibly reconstructed) LaneTransition.
        source_polyline: the source lane's geometry.
        target_polyline: the target lane's geometry.
        ego_source_arc_length_m: ego's source-lane arc-length position
            at the transition frame.
        record: the ScenarioRecord this transition belongs to (used to
            read ego/agent trajectory arrays).
        merge_topology_config: thresholds for merge_start_s/merge_end_s.
        agent_selection_config: thresholds for Front/Rear selection.

    Note (fix commit "materialize merge state at pre-merge reference
    frame"): ``ego_source_arc_length_m`` is the DETECTOR's own
    source-lane arc-length input (computed at ``transition_frame`` by
    the caller, fed into ``detect_merge`` -- unchanged by this
    function). It is used here only to compute ``merge_start_s``/
    ``merge_end_s`` (geometry constants, not frame-dependent) via
    ``compute_merge_start_end_s``, which does not depend on
    ``ego_source_arc_length_m`` at all. ``d_m`` and every other
    decision-state feature below are instead (re)computed at the
    resolved ``feature_reference_frame`` -- see module docstring
    "Three distinct frame concepts".

    Returns:
        A dict with exactly the ACCEPT-only CandidateRecord field names
        as keys, ready to be splatted into a CandidateRecord/manifest
        row via ``**extra``.
    """

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)
    ego_vel_x = np.asarray(log_trajectory.vel_x[sdc_index])
    ego_vel_y = np.asarray(log_trajectory.vel_y[sdc_index])
    ego_length = np.asarray(log_trajectory.length[sdc_index])

    object_ids = np.asarray(record.state.object_metadata.ids)
    object_types = np.asarray(record.state.object_metadata.object_types)

    merge_start_s, merge_end_s = compute_merge_start_end_s(
        source_polyline, target_polyline, merge_topology_config
    )

    merge_start_frame, merge_complete_frame = _derive_merge_frames(
        transition,
        source_polyline,
        ego_x,
        ego_y,
        ego_valid,
        merge_start_s,
        merge_end_s,
    )

    (
        feature_reference_frame,
        feature_reference_valid,
        feature_reference_reason,
    ) = _resolve_feature_reference_frame(transition, merge_start_frame, ego_valid)

    base = dict(
        merge_start_s=merge_start_s,
        merge_end_s=merge_end_s,
        merge_start_frame=merge_start_frame,
        merge_complete_frame=merge_complete_frame,
        feature_reference_frame=feature_reference_frame,
        feature_reference_policy=FEATURE_REFERENCE_POLICY_MERGE_START_FRAME,
        feature_reference_valid=feature_reference_valid,
        feature_reference_reason=feature_reference_reason,
    )

    if not feature_reference_valid:
        # No silent fallback to transition_frame: every decision-state
        # feature is left blank/None, an explicit "could not compute a
        # valid pre-merge snapshot" state (see module docstring and
        # _resolve_feature_reference_frame).
        base.update(
            ego_longitudinal_speed_mps=None,
            merge_distance_m=None,
            front_vehicle_id=None,
            front_gap_m=None,
            front_relative_speed_mps=None,
            front_ttc_s=None,
            rear_vehicle_id=None,
            rear_gap_m=None,
            rear_relative_speed_mps=None,
            rear_ttc_s=None,
            traffic_density=None,
        )
        return base

    frame = feature_reference_frame

    # Fresh source-lane projection AT feature_reference_frame -- a NEW
    # projection, separate from the detector's own arc-length input
    # (computed at transition_frame and never touched by this
    # function). d_m must come from THIS projection, not the
    # detector's.
    reference_source_projection = project_point_to_polyline(
        source_polyline, float(ego_x[frame]), float(ego_y[frame])
    )
    reference_ego_source_arc_length_m = reference_source_projection["arc_length_m"]
    d_m = compute_remaining_merge_distance(
        merge_end_s, reference_ego_source_arc_length_m
    )

    features = extract_interaction_features(
        frame_index=frame,
        target_polyline=target_polyline,
        merge_distance_m=d_m,
        ego_id=record.sdc_id,
        ego_x=float(ego_x[frame]),
        ego_y=float(ego_y[frame]),
        ego_vel_x=float(ego_vel_x[frame]),
        ego_vel_y=float(ego_vel_y[frame]),
        ego_length_m=float(ego_length[frame]),
        object_ids=object_ids,
        object_types=object_types,
        valid=np.asarray(record.state.log_trajectory.valid[:, frame]).astype(bool),
        x=np.asarray(record.state.log_trajectory.x[:, frame]),
        y=np.asarray(record.state.log_trajectory.y[:, frame]),
        yaw=np.asarray(record.state.log_trajectory.yaw[:, frame]),
        vel_x=np.asarray(record.state.log_trajectory.vel_x[:, frame]),
        vel_y=np.asarray(record.state.log_trajectory.vel_y[:, frame]),
        length=np.asarray(record.state.log_trajectory.length[:, frame]),
        config=agent_selection_config,
    )

    base.update(
        ego_longitudinal_speed_mps=features.ego_longitudinal_speed_mps,
        merge_distance_m=features.merge_distance_m,
        front_vehicle_id=features.front_vehicle_id,
        front_gap_m=features.front_gap_m,
        front_relative_speed_mps=features.front_relative_speed_mps,
        front_ttc_s=features.front_ttc_s,
        rear_vehicle_id=features.rear_vehicle_id,
        rear_gap_m=features.rear_gap_m,
        rear_relative_speed_mps=features.rear_relative_speed_mps,
        rear_ttc_s=features.rear_ttc_s,
        traffic_density=features.traffic_density,
    )
    return base


def reconstruct_transition(
    record: ScenarioRecord,
    lane_assignment_config: LaneAssignmentConfig,
    transition_frame: int,
    source_lane_id: int,
    target_lane_id: int,
    candidate_id: str = "",
):
    """Rebuilds the exact ``LaneTransition`` + lane polylines for one
    stored candidate row, by rerunning the same deterministic pipeline
    used at scan time: ``extract_lane_polylines`` ->
    ``assign_ego_lane_sequence`` -> ``compute_stable_lane_sequence`` ->
    ``find_lane_transitions``.

    The match is against ALL FOUR of (the record itself -- i.e. which
    ScenarioRecord was loaded, implicitly fixing record_index --
    transition_frame, source_lane_id, target_lane_id), not
    transition_index alone (transition_index is not stable input to
    this match; it is merely the position the transition happened to
    occupy in one particular scan's output list).

    Returns:
        (transition, source_polyline, target_polyline, ego_source_arc_length_m)

    Raises:
        ValueError: if no reconstructed transition matches all four
            identifying fields.
    """

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)

    polylines = extract_lane_polylines(record.state.roadgraph_points)
    lane_by_id = {polyline.lane_id: polyline for polyline in polylines}

    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, lane_assignment_config
    )
    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=lane_assignment_config.persistence_frames,
        max_ambiguous_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
    )
    transitions = find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
    )

    match = None
    for transition in transitions:
        if (
            transition.transition_frame == transition_frame
            and transition.source_lane_id == source_lane_id
            and transition.target_lane_id == target_lane_id
        ):
            match = transition
            break

    if match is None:
        raise ValueError(
            f"Manifest materialization drift detected for "
            f"candidate_id={candidate_id}: no transition matching "
            f"record_index={record.record_index}, "
            f"transition_frame={transition_frame}, "
            f"source_lane_id={source_lane_id}, "
            f"target_lane_id={target_lane_id}"
        )

    source_polyline = lane_by_id.get(match.source_lane_id)
    target_polyline = lane_by_id.get(match.target_lane_id)

    ego_source_arc_length = None
    if source_polyline is not None:
        frame = match.transition_frame
        projection = project_point_to_polyline(
            source_polyline, float(ego_x[frame]), float(ego_y[frame])
        )
        ego_source_arc_length = projection["arc_length_m"]

    return match, source_polyline, target_polyline, ego_source_arc_length


def build_candidate_records(
    record: ScenarioRecord,
    lane_assignment_config: LaneAssignmentConfig,
    merge_topology_config: MergeTopologyConfig,
    agent_selection_config: AgentSelectionConfig,
) -> Tuple[List[CandidateRecord], Optional[Dict]]:
    """Builds all CandidateRecords for one scenario.

    Reuses the exact call pattern from
    ``scripts/inspect_merge_candidate.py``: extract polylines, assign
    + stabilize ego's lane sequence, find transitions, classify each
    with ``detect_merge``, and for ACCEPT decisions extract
    interaction features.

    ``source_dataset``/``source_split`` are read directly from
    ``record`` (populated by the loader at scan time) rather than
    being passed in separately -- this is now the single source of
    truth for what dataset/split a scenario came from (see
    ``scenario_loader.ScenarioRecord``).

    Error isolation: the entire per-scene body is wrapped in
    try/except so one bad scene does not crash a batch scan. On
    failure, returns ``([], error_info)`` where ``error_info`` is a
    dict with scene_key, record_index, exception type name, and
    message -- never swallowed silently.

    Returns:
        (records, error_info): error_info is None on success.
    """

    try:
        return _build_candidate_records_unsafe(
            record,
            lane_assignment_config,
            merge_topology_config,
            agent_selection_config,
        ), None
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        # for per-scene isolation; caller inspects error_info.
        return [], {
            "scene_key": record.scene_key,
            "record_index": record.record_index,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }


def _build_candidate_records_unsafe(
    record: ScenarioRecord,
    lane_assignment_config: LaneAssignmentConfig,
    merge_topology_config: MergeTopologyConfig,
    agent_selection_config: AgentSelectionConfig,
) -> List[CandidateRecord]:

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)

    polylines = extract_lane_polylines(record.state.roadgraph_points)
    lane_by_id = {polyline.lane_id: polyline for polyline in polylines}

    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, lane_assignment_config
    )

    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=lane_assignment_config.persistence_frames,
        max_ambiguous_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
    )

    transitions = find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
    )

    records: List[CandidateRecord] = []

    for transition_index, transition in enumerate(transitions):

        source_polyline = lane_by_id.get(transition.source_lane_id)
        target_polyline = lane_by_id.get(transition.target_lane_id)

        ego_source_arc_length = None
        if source_polyline is not None:
            frame = transition.transition_frame
            projection = project_point_to_polyline(
                source_polyline, float(ego_x[frame]), float(ego_y[frame])
            )
            ego_source_arc_length = projection["arc_length_m"]

        diagnostic = detect_merge(
            transition,
            source_polyline,
            target_polyline,
            ego_source_arc_length,
            merge_topology_config,
        )

        scene_key = record.scene_key
        candidate_id = make_candidate_id(
            scene_key,
            transition.transition_frame,
            transition.source_lane_id,
            transition.target_lane_id,
        )

        extra = {}

        if diagnostic.decision == MergeDecision.ACCEPT:

            extra = materialize_merge_features(
                transition,
                source_polyline,
                target_polyline,
                ego_source_arc_length,
                record,
                merge_topology_config,
                agent_selection_config,
            )

        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                scene_key=scene_key,
                source_dataset=getattr(record, "source_dataset", SOURCE_DATASET),
                source_split=getattr(record, "source_split", "validation"),
                source_shard=record.source_shard,
                record_index=record.record_index,
                transition_index=transition_index,
                transition_frame=transition.transition_frame,
                source_lane_id=transition.source_lane_id,
                target_lane_id=transition.target_lane_id,
                source_start_frame=transition.source_start_frame,
                source_end_frame=transition.source_end_frame,
                target_start_frame=transition.target_start_frame,
                target_end_frame=transition.target_end_frame,
                decision=diagnostic.decision.value,
                reason=diagnostic.reason,
                source_lane_ends=diagnostic.source_lane_ends,
                source_remaining_distance_m=diagnostic.source_remaining_distance_m,
                endpoint_target_distance_m=diagnostic.endpoint_target_distance_m,
                endpoint_target_arc_length_m=diagnostic.endpoint_target_arc_length_m,
                heading_difference_deg=diagnostic.heading_difference_deg,
                lanes_converge=diagnostic.lanes_converge,
                parallel_continuation=diagnostic.parallel_continuation,
                separation_reduction_m=diagnostic.separation_reduction_m,
                decreasing_fraction=diagnostic.decreasing_fraction,
                max_collinear_offset_m=diagnostic.max_collinear_offset_m,
                upstream_separation_m=diagnostic.upstream_separation_m,
                pre_merge_frames=diagnostic.pre_merge_frames,
                target_lane_persistent=diagnostic.target_lane_persistent,
                **extra,
            )
        )

    return records


def _sort_key(candidate: CandidateRecord):
    return (
        candidate.source_split,
        candidate.source_shard,
        candidate.record_index,
        candidate.transition_frame,
        candidate.source_lane_id,
        candidate.target_lane_id,
    )


def sort_candidates(
    candidates: List[CandidateRecord],
) -> List[CandidateRecord]:
    """Stable, deterministic sort per module docstring."""

    return sorted(candidates, key=_sort_key)


def _check_no_duplicate_ids(candidates: List[CandidateRecord]) -> None:
    seen = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError(
                f"Duplicate candidate_id produced within one run: "
                f"{candidate.candidate_id!r}"
            )
        seen.add(candidate.candidate_id)


def _field_to_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return value


def write_candidates_csv(
    candidates: List[CandidateRecord], output_path: Path
) -> None:
    """Writes CandidateRecords to a CSV file with the exact schema.

    Sorts candidates deterministically first (see module docstring),
    and raises a clear error on any duplicate candidate_id within the
    batch before writing anything.
    """

    ordered = sort_candidates(candidates)
    _check_no_duplicate_ids(ordered)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()

        for candidate in ordered:
            row = dataclasses.asdict(candidate)
            row = {
                field: _field_to_csv_value(row[field])
                for field in CANDIDATE_FIELDS
            }
            writer.writerow(row)


def compute_summary_statistics(candidates: List[CandidateRecord]) -> Dict:
    """Computes scan summary statistics for a batch of CandidateRecords.

    Asserts ACCEPT + REJECT + REVIEW == total_stable_transitions before
    returning -- this must hold by construction (every candidate has
    exactly one decision value), so this assertion is a strict
    correctness guard, not a soft check.
    """

    total = len(candidates)
    scene_keys = {c.scene_key for c in candidates}

    accept_count = sum(
        1 for c in candidates if c.decision == MergeDecision.ACCEPT.value
    )
    reject_count = sum(
        1 for c in candidates if c.decision == MergeDecision.REJECT.value
    )
    review_count = sum(
        1 for c in candidates if c.decision == MergeDecision.REVIEW.value
    )

    assert accept_count + reject_count + review_count == total, (
        "ACCEPT + REJECT + REVIEW must equal total_stable_transitions "
        f"(got {accept_count} + {reject_count} + {review_count} != {total})"
    )

    reason_histogram: Dict[str, int] = {}
    for candidate in candidates:
        key = candidate.reason if candidate.reason is not None else "none"
        reason_histogram[key] = reason_histogram.get(key, 0) + 1

    review_ratio = (review_count / total) if total > 0 else 0.0
    accept_review_total = accept_count + review_count
    review_ratio_among_accept_review = (
        (review_count / accept_review_total)
        if accept_review_total > 0
        else 0.0
    )

    return {
        "scenes_with_transitions": len(scene_keys),
        "total_stable_transitions": total,
        "accept_count": accept_count,
        "reject_count": reject_count,
        "review_count": review_count,
        "review_ratio": review_ratio,
        "review_ratio_among_accept_review": review_ratio_among_accept_review,
        "reason_histogram": reason_histogram,
    }


def compute_multi_shard_summary(
    candidates: List[CandidateRecord],
    physical_shards_scanned: int,
    scenes_scanned: int,
    scenes_failed: int,
    per_shard_scan_counts: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None,
) -> Dict:
    """Computes the shard-aware summary shape written by
    ``scripts/extract_merge_scenes.py`` for a multi-shard scan.

    Reuses ``compute_summary_statistics`` for both the ``global``
    section (over all candidates) and each ``per_shard`` entry (over
    just that shard's candidates) -- no parallel statistics
    implementation.

    Args:
        candidates: every CandidateRecord produced by the scan (across
            all shards).
        physical_shards_scanned: number of physical shard files
            iterated (including any that produced zero candidates).
        scenes_scanned: total scenario count scanned across all shards
            (includes scenes with zero transitions).
        scenes_failed: total scene-level failures across all shards.
        per_shard_scan_counts: optional dict of
            ``(source_split, source_shard) -> {"scenes_scanned": int,
            "scenes_failed": int}``, supplied by the caller since scan
            -level counts (as opposed to candidate-level counts) are
            not recoverable from ``candidates`` alone (e.g. a scene
            with zero transitions, or one that failed and produced no
            candidates, otherwise leaves no trace). Shards absent from
            this dict (or when it is None) default to 0/0.

    Returns:
        A dict with ``global`` (the usual ``compute_summary_statistics``
        keys plus ``physical_shards_scanned``/``scenes_scanned``/
        ``scenes_failed``) and ``per_shard`` (a list of per-
        (source_split, source_shard) summaries, sorted by
        (source_split, source_shard), each with ``scenes_scanned``/
        ``scenes_failed`` plus the core ``compute_summary_statistics``
        counts -- not the ratio/histogram fields, which are most
        meaningful in aggregate).
    """

    global_summary = compute_summary_statistics(candidates)
    global_summary["physical_shards_scanned"] = physical_shards_scanned
    global_summary["scenes_scanned"] = scenes_scanned
    global_summary["scenes_failed"] = scenes_failed

    per_shard_scan_counts = per_shard_scan_counts or {}

    by_shard: Dict[Tuple[str, str], List[CandidateRecord]] = {}
    for candidate in candidates:
        key = (candidate.source_split, candidate.source_shard)
        by_shard.setdefault(key, []).append(candidate)

    all_keys = set(by_shard.keys()) | set(per_shard_scan_counts.keys())

    per_shard = []
    for (source_split, source_shard) in sorted(all_keys):
        shard_candidates = by_shard.get((source_split, source_shard), [])
        shard_summary = compute_summary_statistics(shard_candidates)
        scan_counts = per_shard_scan_counts.get(
            (source_split, source_shard), {"scenes_scanned": 0, "scenes_failed": 0}
        )
        per_shard.append(
            {
                "source_split": source_split,
                "source_shard": source_shard,
                "scenes_scanned": scan_counts.get("scenes_scanned", 0),
                "scenes_failed": scan_counts.get("scenes_failed", 0),
                "scenes_with_transitions": shard_summary["scenes_with_transitions"],
                "total_stable_transitions": shard_summary["total_stable_transitions"],
                "accept_count": shard_summary["accept_count"],
                "reject_count": shard_summary["reject_count"],
                "review_count": shard_summary["review_count"],
            }
        )

    return {"global": global_summary, "per_shard": per_shard}


def write_summary_json(summary: Dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def read_candidates_csv(input_path: Path) -> List[Dict[str, str]]:
    """Reads merge_candidates.csv back into a list of raw string dicts
    (used by downstream CLI scripts that need to filter/join rows
    without re-instantiating CandidateRecord objects).
    """

    input_path = Path(input_path)
    with open(input_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)
