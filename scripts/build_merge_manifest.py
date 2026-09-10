"""Phase 1 CLI: sync manual validation labels and build the final
CONFIRMED_MERGE manifest (Compressed Commit E; fix commit "materialize
manually confirmed merge features").

Two responsibilities, run every time this script executes:

1. Label-template sync (safe to rerun): loads
   ``--labels`` (data/manifests/merge_manual_labels.csv) if present,
   adds UNREVIEWED rows for any candidate_id in ``--candidates`` not
   already labeled, keeps (never deletes) labels for candidate_ids no
   longer in the current scan (warns about them), and never overwrites
   an existing label's ``manual_validation``/``manual_note``. Optionally
   applies ``--set-label CANDIDATE_ID LABEL`` to the merged set -- but
   ONLY if CANDIDATE_ID is present in the CURRENT candidates file (see
   ``sync_labels``); an unknown candidate_id raises ``ValueError``
   rather than silently creating a phantom label row.

2. Final manifest: inner-joins candidates with labels WHERE
   manual_validation == CONFIRMED_MERGE, writing
   ``--manifest-output`` (data/manifests/merge_manifest.csv). Zero rows
   is a VALID, expected result before manual review has happened.

   Bug fixed here: ``merge_candidates.csv`` only computes ACCEPT-only
   feature columns (merge_start_s, ego speed, front/rear gap/ttc,
   traffic_density, ...) for detector-ACCEPT rows -- REVIEW/REJECT rows
   legitimately leave them blank there (see dataset_builder.py). A
   human manually confirming a REVIEW (or REJECT) candidate as a
   genuine merge must NOT propagate those blanks into the final
   manifest: every CONFIRMED_MERGE row is reloaded from the real
   scenario, its transition is reconstructed
   (``dataset_builder.reconstruct_transition``), the detector is
   rerun and checked for drift against the stored CSV row, and its
   full ACCEPT-only feature set is (re)computed via
   ``dataset_builder.materialize_merge_features`` -- regardless of
   whether the original stored decision was accept, review, or reject.
   ``detector_decision``/``detector_reason`` in the output are always
   the ORIGINAL stored decision/reason (never rewritten to "accept"
   just because features were materialized).

Example:
    python scripts/build_merge_manifest.py
    python scripts/build_merge_manifest.py --set-label <candidate_id> CONFIRMED_MERGE
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import (
    materialize_merge_features,
    read_candidates_csv,
    reconstruct_transition,
)
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.merge_detector import detect_merge, load_merge_topology_config
from src.scenarios.scenario_features import load_agent_selection_config
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    resolve_physical_shard,
)

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_PHASE1_CONFIG = "configs/phase1_merge.yaml"
DEFAULT_CANDIDATES = "data/manifests/merge_candidates.csv"
DEFAULT_LABELS = "data/manifests/merge_manual_labels.csv"
DEFAULT_MANIFEST_OUTPUT = "data/manifests/merge_manifest.csv"

ALLOWED_LABELS = frozenset(
    {"UNREVIEWED", "CONFIRMED_MERGE", "CONFIRMED_NON_MERGE", "UNCERTAIN"}
)

LABEL_FIELDS = ["candidate_id", "manual_validation", "manual_note"]

# Required (must be non-blank/valid) for every manifest row -- see
# `validate_manifest_row`. Front/Rear fields are deliberately NOT
# required here (a real "no qualifying vehicle" is a valid outcome),
# but their internal consistency is still checked.
REQUIRED_MANIFEST_FIELDS = [
    "candidate_id",
    "scene_key",
    "record_index",
    "source_lane_id",
    "target_lane_id",
    "transition_frame",
    "merge_start_s",
    "merge_end_s",
    "merge_complete_frame",
    "ego_longitudinal_speed_mps",
    "merge_distance_m",
    "traffic_density",
]

MANIFEST_FIELDS = [
    "candidate_id",
    "scene_key",
    "source_dataset",
    "source_split",
    "source_shard",
    "record_index",
    "source_lane_id",
    "target_lane_id",
    "transition_frame",
    "merge_start_frame",
    "merge_complete_frame",
    "feature_reference_frame",
    "feature_reference_policy",
    "feature_reference_valid",
    "feature_reference_reason",
    "merge_start_s",
    "merge_end_s",
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
    "detector_decision",
    "detector_reason",
    "manual_validation",
    "manual_note",
    "feature_materialization_source",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Sync manual merge-candidate validation labels and build "
            "the final CONFIRMED_MERGE manifest."
        )
    )
    parser.add_argument(
        "--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG
    )
    parser.add_argument(
        "--phase1-config", type=str, default=DEFAULT_PHASE1_CONFIG
    )
    parser.add_argument("--candidates", type=str, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    parser.add_argument(
        "--manifest-output", type=str, default=DEFAULT_MANIFEST_OUTPUT
    )
    parser.add_argument(
        "--set-label", type=str, nargs=2, default=None,
        metavar=("CANDIDATE_ID", "LABEL"),
    )

    return parser.parse_args()


def load_labels(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {row["candidate_id"]: dict(row) for row in reader}


def write_labels(path: Path, labels: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for candidate_id in sorted(labels.keys()):
            row = labels[candidate_id]
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "manual_validation": row["manual_validation"],
                    "manual_note": row.get("manual_note", ""),
                }
            )


def sync_labels(candidate_ids, existing_labels: dict, set_label=None):
    """Applies the label-template sync rules. Returns the merged dict.

    Also returns the list of stale candidate_ids (labeled but no longer
    in the current candidate set) for the caller to warn about.

    ``set_label`` safety fix: a NEW ``--set-label CANDIDATE_ID LABEL``
    write is only accepted when CANDIDATE_ID is present in the CURRENT
    ``candidate_ids`` (i.e. freshly read from the current
    merge_candidates.csv) -- raises ``ValueError`` otherwise, and does
    not mutate `merged` at all in that case. This is intentionally
    different from a pre-existing STALE label already present in
    `existing_labels` from a previous run: those are still preserved
    (warned-about, never deleted) -- only a *new* set-label write for
    an identifier absent from the current candidate set is rejected.
    """

    merged = dict(existing_labels)
    candidate_id_set = set(candidate_ids)

    for candidate_id in candidate_ids:
        if candidate_id not in merged:
            merged[candidate_id] = {
                "candidate_id": candidate_id,
                "manual_validation": "UNREVIEWED",
                "manual_note": "",
            }

    stale = sorted(set(merged.keys()) - candidate_id_set)

    if set_label is not None:
        set_candidate_id, set_value = set_label
        if set_value not in ALLOWED_LABELS:
            raise ValueError(
                f"--set-label value must be one of {sorted(ALLOWED_LABELS)}, "
                f"got {set_value!r}"
            )
        if set_candidate_id not in candidate_id_set:
            raise ValueError(
                f"Unknown candidate_id: {set_candidate_id!r} is not "
                "present in the current candidates file -- refusing to "
                "set a label for it."
            )
        merged[set_candidate_id]["manual_validation"] = set_value

    return merged, stale


def _consistent_front_rear(row_values: dict, side: str, errors: list, candidate_id: str):
    """Validates the front_*/rear_* internal-consistency invariant.

    If vehicle_id is blank: gap/relative_speed must also be blank and
    ttc must be exactly inf. If vehicle_id is present: gap/relative
    -speed must be real numbers and ttc must be a real (non-inf)
    number... actually ttc CAN be a real number of any finite value
    (including 0.0 for already-overlapping) -- the only hard rule is
    "not blank" when vehicle_id is present, and "exactly inf" when it
    is not.
    """

    vehicle_id = row_values[f"{side}_vehicle_id"]
    gap = row_values[f"{side}_gap_m"]
    rel_speed = row_values[f"{side}_relative_speed_mps"]
    ttc = row_values[f"{side}_ttc_s"]

    if vehicle_id in (None, ""):
        if gap not in (None, "") or rel_speed not in (None, ""):
            errors.append(
                f"{side}_vehicle_id is blank but {side}_gap_m/"
                f"{side}_relative_speed_mps is not blank"
            )
        if not (isinstance(ttc, float) and math.isinf(ttc)):
            errors.append(
                f"{side}_vehicle_id is blank but {side}_ttc_s is not inf "
                f"(got {ttc!r})"
            )
    else:
        if gap in (None, "") or rel_speed in (None, ""):
            errors.append(
                f"{side}_vehicle_id={vehicle_id!r} is present but "
                f"{side}_gap_m/{side}_relative_speed_mps is blank"
            )
        if ttc is None or ttc == "":
            errors.append(
                f"{side}_vehicle_id={vehicle_id!r} is present but "
                f"{side}_ttc_s is blank"
            )


def validate_manifest_row(row: dict) -> None:
    """Validates one final-manifest row before it is written.

    Raises:
        ValueError: a clear, structured error naming the candidate_id
            and which field(s) failed, if any required field is
            missing/invalid, the feature-reference-frame is invalid
            (see below), or the front/rear consistency invariant (see
            module docstring) is violated. Never silently drops or
            pads a row.
    """

    candidate_id = row.get("candidate_id", "<unknown>")
    missing = [
        field
        for field in REQUIRED_MANIFEST_FIELDS
        if row.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"Incomplete manifest row for candidate_id={candidate_id}: "
            f"missing/blank required field(s): {missing}"
        )

    # feature_reference_valid required-field-style check (fix commit
    # "materialize merge state at pre-merge reference frame"): a
    # CONFIRMED_MERGE row must fail loudly rather than silently write
    # an incomplete/invalid-reference row into the final manifest.
    feature_reference_valid = row.get("feature_reference_valid")
    if str(feature_reference_valid) != "True":
        reason = row.get("feature_reference_reason") or "<unknown>"
        raise ValueError(
            f"Refusing to write manifest row for candidate_id={candidate_id}: "
            f"feature_reference_valid is not True (reason={reason!r}) -- "
            "a CONFIRMED_MERGE candidate must have a valid pre-merge "
            "feature reference frame before it can enter the final "
            "manifest."
        )

    errors: list = []
    _consistent_front_rear(row, "front", errors, candidate_id)
    _consistent_front_rear(row, "rear", errors, candidate_id)
    if errors:
        raise ValueError(
            f"Inconsistent front/rear fields for candidate_id={candidate_id}: "
            + "; ".join(errors)
        )


def _to_float_or_blank(value):
    if value in (None, ""):
        return ""
    return float(value)


def materialize_confirmed_rows(
    candidate_rows,
    confirmed_ids,
    dataset_config,
    lane_assignment_config,
    merge_topology_config,
    agent_selection_config,
):
    """Materializes the full ACCEPT-only feature set for every
    CONFIRMED_MERGE candidate.

    Safety-critical grouping: candidates are grouped by
    ``(source_split, source_shard, record_index)`` -- NEVER by
    record_index alone -- since two different physical shards can
    legitimately share the same record_index. For each distinct
    (source_split, source_shard) pair referenced by any confirmed
    candidate, the exact physical shard path is resolved via
    ``resolve_physical_shard`` (the single source of truth for
    basename -> physical path resolution -- never a hardcoded
    ``data/womd/<split>/<file>`` convention) against ``dataset_config``
    (a ``DatasetExpansionConfig``, the authority on which physical
    files are actually configured/available), then iterated in
    isolation via ``build_waymax_config``, so a candidate from shard A
    can never be matched against a same-numbered record_index loaded
    from shard B.

    ``dataset_config`` (the CLI's ``--dataset-config``, already loaded
    via ``load_dataset_config``) is REQUIRED here -- it is the sole
    authority for resolving each row's ``source_shard`` to a physical
    path, and its ``split`` is checked against each row's own
    ``source_split`` BEFORE any path/file access is attempted (see
    ``resolve_physical_shard``'s split check, which fires first).

    Returns:
        dict candidate_id -> materialized feature dict (the
        ACCEPT-only field names, plus 'feature_materialization_source').

    Raises:
        ValueError: if a row's ``source_split`` disagrees with
            ``dataset_config.split`` (checked before any path/file
            access), if a row's ``source_shard`` does not match any
            configured physical path (no fallback), if a row's
            ``source_shard`` basename is ambiguous (matches more than
            one configured path), if a transition cannot be
            reconstructed (drift), or the recomputed detector
            decision/reason disagrees with the stored CSV row's
            decision/reason.
    """

    rows_by_id = {row["candidate_id"]: row for row in candidate_rows}

    by_shard_record = defaultdict(list)
    for candidate_id in confirmed_ids:
        row = rows_by_id[candidate_id]
        source_split = row.get("source_split") or dataset_config.split
        source_shard = row.get("source_shard")
        if not source_shard:
            raise ValueError(
                f"CONFIRMED_MERGE candidate_id={candidate_id!r} has no "
                "source_shard recorded -- cannot safely determine which "
                "physical shard file to reload."
            )
        by_shard_record[(source_split, source_shard, int(row["record_index"]))].append(row)

    if not by_shard_record:
        return {}

    # Group by physical shard so each shard file is opened at most once,
    # scanning only up to the max record_index actually needed from it.
    by_shard = defaultdict(dict)
    for (source_split, source_shard, record_index), rows in by_shard_record.items():
        by_shard[(source_split, source_shard)][record_index] = rows

    materialized = {}

    for (source_split, source_shard), rows_by_record_index in by_shard.items():

        # Split-mismatch check fires FIRST, before any path/file access
        # (see resolve_physical_shard's docstring/implementation).
        shard_path = resolve_physical_shard(
            dataset_config, source_shard=source_shard, source_split=source_split
        )
        max_record_index = max(rows_by_record_index.keys())

        shard_waymax_config = build_waymax_config(dataset_config, shard_path)

        for record in iter_scenarios(
            shard_waymax_config,
            limit=max_record_index + 1,
            source_dataset="WOMD",
            source_split=source_split,
        ):

            record_rows = rows_by_record_index.get(record.record_index)
            if not record_rows:
                continue

            # Extra safety check: the record we loaded must actually be
            # from the physical shard file we intended (defends against
            # any future refactor that might silently swap in a
            # different iterator).
            if record.source_shard != source_shard:
                raise ValueError(
                    "Manifest materialization safety check failed: "
                    f"expected source_shard={source_shard!r} but loaded "
                    f"record from source_shard={record.source_shard!r} "
                    f"(record_index={record.record_index})."
                )

            for row in record_rows:

                candidate_id = row["candidate_id"]
                transition_frame = int(row["transition_frame"])
                source_lane_id = int(row["source_lane_id"])
                target_lane_id = int(row["target_lane_id"])

                (
                    transition,
                    source_polyline,
                    target_polyline,
                    ego_source_arc_length,
                ) = reconstruct_transition(
                    record,
                    lane_assignment_config,
                    transition_frame,
                    source_lane_id,
                    target_lane_id,
                    candidate_id=candidate_id,
                )

                diagnostic = detect_merge(
                    transition,
                    source_polyline,
                    target_polyline,
                    ego_source_arc_length,
                    merge_topology_config,
                )

                stored_decision = row["decision"]
                stored_reason = row["reason"] if row["reason"] else None
                new_decision = diagnostic.decision.value
                new_reason = diagnostic.reason

                if (new_decision, new_reason) != (stored_decision, stored_reason):
                    raise ValueError(
                        "Detector drift between merge_candidates.csv and "
                        f"current code/config for candidate_id={candidate_id}: "
                        f"stored=({stored_decision},{stored_reason}) "
                        f"recomputed=({new_decision},{new_reason})"
                    )

                features = materialize_merge_features(
                    transition,
                    source_polyline,
                    target_polyline,
                    ego_source_arc_length,
                    record,
                    merge_topology_config,
                    agent_selection_config,
                )

                source = (
                    "detector_accept"
                    if stored_decision == "accept"
                    else "manual_recompute"
                )
                features["feature_materialization_source"] = source

                materialized[candidate_id] = features

    missing_ids = confirmed_ids - set(materialized.keys())
    if missing_ids:
        raise ValueError(
            "Failed to materialize features for CONFIRMED_MERGE "
            f"candidate_id(s) (record_index not reached in the "
            f"corresponding shard's dataset scan): {sorted(missing_ids)}"
        )

    return materialized


def build_manifest(
    candidate_rows,
    labels: dict,
    dataset_config=None,
    lane_assignment_config=None,
    merge_topology_config=None,
    agent_selection_config=None,
):
    """Builds the final CONFIRMED_MERGE manifest.

    For every CONFIRMED_MERGE candidate, the ACCEPT-only feature
    columns are always taken from a fresh
    ``materialize_confirmed_rows`` call (which reloads the real
    scenario, reconstructs the transition, drift-checks the detector,
    and recomputes the full feature set) -- never blindly copied from
    the (possibly blank, for REVIEW/REJECT) merge_candidates.csv
    columns. ``detector_decision``/``detector_reason`` are always the
    ORIGINAL stored CSV decision/reason.

    The four config args are required (not optional in practice) --
    they are keyword args with a None default only so existing/older
    call sites are easy to spot at review time; a call missing them
    will fail inside ``materialize_confirmed_rows``/``iter_scenarios``
    with a clear error rather than this function silently no-op'ing.
    """

    confirmed_ids = {
        row["candidate_id"]
        for row in candidate_rows
        if labels.get(row["candidate_id"], {}).get("manual_validation")
        == "CONFIRMED_MERGE"
    }

    confirmed_splits = {
        row.get("source_split")
        for row in candidate_rows
        if row["candidate_id"] in confirmed_ids
    }
    if len(confirmed_splits) > 1:
        raise ValueError(
            "Candidates CSV contains more than one distinct source_split "
            f"value among its CONFIRMED_MERGE rows: {sorted(confirmed_splits)}. "
            "One manifest invocation processes exactly one logical split "
            "-- refusing to silently process a mixed-split CSV."
        )

    materialized_by_id = materialize_confirmed_rows(
        candidate_rows,
        confirmed_ids,
        dataset_config,
        lane_assignment_config,
        merge_topology_config,
        agent_selection_config,
    )

    manifest_rows = []
    for row in candidate_rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in confirmed_ids:
            continue

        label = labels[candidate_id]
        features = materialized_by_id[candidate_id]

        manifest_row = {
            "candidate_id": candidate_id,
            "scene_key": row["scene_key"],
            "source_dataset": row["source_dataset"],
            "source_split": row["source_split"],
            "source_shard": row["source_shard"],
            "record_index": row["record_index"],
            "source_lane_id": row["source_lane_id"],
            "target_lane_id": row["target_lane_id"],
            "transition_frame": row["transition_frame"],
            "merge_start_frame": features["merge_start_frame"],
            "merge_complete_frame": features["merge_complete_frame"],
            "feature_reference_frame": features["feature_reference_frame"],
            "feature_reference_policy": features["feature_reference_policy"],
            "feature_reference_valid": features["feature_reference_valid"],
            "feature_reference_reason": features["feature_reference_reason"],
            "merge_start_s": features["merge_start_s"],
            "merge_end_s": features["merge_end_s"],
            "ego_longitudinal_speed_mps": features["ego_longitudinal_speed_mps"],
            "merge_distance_m": features["merge_distance_m"],
            "front_vehicle_id": features["front_vehicle_id"],
            "front_gap_m": features["front_gap_m"],
            "front_relative_speed_mps": features["front_relative_speed_mps"],
            "front_ttc_s": features["front_ttc_s"],
            "rear_vehicle_id": features["rear_vehicle_id"],
            "rear_gap_m": features["rear_gap_m"],
            "rear_relative_speed_mps": features["rear_relative_speed_mps"],
            "rear_ttc_s": features["rear_ttc_s"],
            "traffic_density": features["traffic_density"],
            "detector_decision": row["decision"],
            "detector_reason": row["reason"],
            "manual_validation": label["manual_validation"],
            "manual_note": label.get("manual_note", ""),
            "feature_materialization_source": features[
                "feature_materialization_source"
            ],
        }

        # Normalize None -> "" for CSV writing (materialize_merge_features
        # returns Python None for absent front/rear fields; csv.DictWriter
        # would otherwise write the string "None").
        for field in MANIFEST_FIELDS:
            if manifest_row[field] is None:
                manifest_row[field] = ""

        validate_manifest_row(manifest_row)
        manifest_rows.append(manifest_row)

    return sorted(manifest_rows, key=lambda r: r["candidate_id"])


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Build Merge Manifest (fix: materialize confirmed merge features)")
    print("=" * 70)

    candidate_rows = read_candidates_csv(args.candidates)
    candidate_ids = [row["candidate_id"] for row in candidate_rows]

    labels_path = Path(args.labels)
    existing_labels = load_labels(labels_path)

    merged_labels, stale = sync_labels(
        candidate_ids, existing_labels, set_label=args.set_label
    )

    if stale:
        print(
            "\nWARN: previously labeled candidates no longer in current scan:"
        )
        for candidate_id in stale:
            print(f"  {candidate_id}")

    write_labels(labels_path, merged_labels)
    print(f"\nWrote labels CSV     : {labels_path} ({len(merged_labels)} rows)")

    dataset_config = load_dataset_config(args.dataset_config)
    lane_assignment_config = load_lane_assignment_config(args.phase1_config)
    merge_topology_config = load_merge_topology_config(args.phase1_config)
    agent_selection_config = load_agent_selection_config(args.phase1_config)

    manifest_rows = build_manifest(
        candidate_rows,
        merged_labels,
        dataset_config=dataset_config,
        lane_assignment_config=lane_assignment_config,
        merge_topology_config=merge_topology_config,
        agent_selection_config=agent_selection_config,
    )
    write_manifest(Path(args.manifest_output), manifest_rows)
    print(
        f"Wrote manifest CSV   : {args.manifest_output} "
        f"({len(manifest_rows)} rows)"
    )

    counts = {}
    for label in merged_labels.values():
        counts[label["manual_validation"]] = counts.get(label["manual_validation"], 0) + 1

    reviewed = sum(v for k, v in counts.items() if k != "UNREVIEWED")

    print("\n" + "=" * 70)
    print("Manual Review Stats")
    print("=" * 70)
    print(f"  total labeled candidates   : {len(merged_labels)}")
    print(f"  reviewed (non-UNREVIEWED)  : {reviewed}")
    print(f"  CONFIRMED_MERGE            : {counts.get('CONFIRMED_MERGE', 0)}")
    print(f"  CONFIRMED_NON_MERGE        : {counts.get('CONFIRMED_NON_MERGE', 0)}")
    print(f"  UNCERTAIN                  : {counts.get('UNCERTAIN', 0)}")
    print(f"  UNREVIEWED                 : {counts.get('UNREVIEWED', 0)}")

    decision_by_id = {row["candidate_id"]: row["decision"] for row in candidate_rows}
    breakdown = {}
    for candidate_id, label in merged_labels.items():
        detector_decision = decision_by_id.get(candidate_id)
        if detector_decision is None:
            continue
        key = f"{detector_decision.upper()} & {label['manual_validation']}"
        breakdown[key] = breakdown.get(key, 0) + 1

    print("\n  Detector-decision x manual-validation breakdown:")
    for key, count in sorted(breakdown.items()):
        print(f"    {key}: {count}")

    if len(manifest_rows) == 0:
        print(
            "\n  0 confirmed manifest rows (expected until manual review occurs)."
        )

    print("\n" + "=" * 70)
    print("BUILD MERGE MANIFEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
