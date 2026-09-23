# Interaction-aware MERGE Dataset v2

## Status

The former 168-maneuver geometry-only dataset is **legacy v1** and is not
valid evidence for paper results. `MAN_0013` is the pinned regression case:
lane 204 ends 1.85 m before nearly-collinear lane 203 begins, and the old
finite-polyline projection mistook longitudinal approach for convergence.

The v2 code is fail-closed. Training and visualization default to schema
`merge_interaction_v2`; old checkpoints load only when a caller explicitly
uses the legacy diagnostic override. No automatic checkpoint migration is
supported.

## Required source data

Both representations of the same WOMD release/split are required:

1. `*_tfexample.tfrecord*` for Waymax simulation and trajectories.
2. WOMD Scenario-protobuf TFRecords for `LaneCenter.entry_lanes` and
   `LaneCenter.exit_lanes`.

They are joined by the embedded `scenario/id`, never by shard position.
`tf_example` and Scenario-protobuf TFRecords are shuffled/sharded
**independently** from the same underlying WOMD release -- a tf_example
shard's scenario ids are scattered arbitrarily across the ~1000-shard
Scenario protobuf split, never at the matching shard index. There is no
cheaper join than scanning the Scenario protobuf split for the wanted ids
(`index_scenario_topologies_for_ids`, used by
`scripts/extract_merge_v2_evidence.py`, streams and discards unwanted
scenarios instead of holding every topology in memory at once).

The matching WOMD release is confirmed to be **v1.3.1**
(`gs://waymo_open_dataset_motion_v_1_3_1/`) by exact byte-size match on shard 0
(`training_tfexample.tfrecord-00000-of-01000` is 1,224,401,591 bytes in both
this workspace and the v1.3.1 bucket).

**Do not bulk-download the Scenario protobuf split.** An earlier attempt to
fetch the full training split (~455 GB, 1000 shards) and validation split
(~41 GB, 150 shards) exhausted local disk and was deleted. There is no
shard-index correspondence between `tf_example` and Scenario-protobuf
TFRecords -- they are shuffled/sharded independently from the same
underlying WOMD release (verified: of a `tf_example` shard's ~450-500
scenario ids, only ~1 lands in the Scenario-protobuf shard of the same
index). The only way to locate a wanted `scenario_id` is to scan the
Scenario-protobuf split for it -- streaming via `tf.data.TFRecordDataset`
avoids local storage but not network cost (~100 s/shard measured; a full
1000-shard training scan is ~28 hours and effectively re-downloads the
whole split over the network). Only fetch shards you have already confirmed
(via a smaller pilot, or an actual manifest of needed shards) contain scenario
ids this repo's local `tf_example` shards need.

Currently present locally, as a deliberately minimal pilot (one shard per
split, kept from the earlier bulk attempt): `data/womd/scenario_proto/
training/training.tfrecord-00000-of-01000` (496 scenarios) and
`data/womd/scenario_proto/validation/validation.tfrecord-00000-of-00150`
(286 scenarios). Cross-referencing these against all 16 local `tf_example`
shards found 6 training + 8 validation matching scenarios -- enough for an
end-to-end pilot extraction, not a representative sample. Extending
coverage requires either accepting the full-scan cost above for a specific,
justified batch of additional shards, or another candidate-discovery
strategy; do not resume unbounded downloading/streaming without deciding
this explicitly first.

Rather than depend on the official `waymo_open_dataset` PyPI package (which
pins `jax==0.4.30`/`tensorflow==2.12.0`/`numpy==1.23.0`, conflicting with this
repo's PPO training stack), the four `protoc`-generated modules this loader
actually needs (`scenario_pb2`, `map_pb2`, and their two transitive
dependencies) are vendored verbatim under
`src/scenarios/vendor/waymo_open_dataset/` (Apache License 2.0; see that
directory's `README.md` for provenance and refresh instructions). Only
`google.protobuf` (already a transitive dependency of this repo's own
`tensorflow` install) is required -- the `its-merge` conda environment is
never modified. Do not rename tf_example files to make them pass: the loader
explicitly rejects them as topology sources.

## Pipeline

1. Generate the evidence review queue by joining broad trajectory transitions
   to protobuf topology using `scenario/id`:

   ```bash
   PYTHONPATH=. python scripts/extract_merge_v2_evidence.py \
     --dataset-config <matching-tfexample-dataset.yaml> \
     --scenario-proto <scenario-protobuf-shard...> \
     --transitions-csv <broad-transition-candidates.csv>
   ```

2. This builds `TopologyEvidence` and temporal `InteractionEvidence`; the
   latter stores the complete front/rear gap/TTC time series used for the
   decision. `ScenarioTopology.evidence()` auto-derives
   `source_exit_lane_count`, `target_polyline_point_count`,
   `is_lane_change_target` (a `LaneNeighbor` relation with no
   entry_lanes/exit_lanes topology link), and `is_intersection_transition`
   (a topology-connected, single-entry target with a >=30 degree heading
   change) directly from parsed lane topology.
3. Run `classify_v2_merge`. Explicit reject categories, in gate order:
   `REJECT_SERIAL_CONTINUATION`, `REJECT_MAP_ARTIFACT`, `REJECT_DIVERGE`,
   `REJECT_LANE_CHANGE`, `REJECT_INTERSECTION`,
   `REJECT_CUT_IN_NOT_TOPOLOGY_MERGE` (a lateral cut-in with no map-topology
   convergence is a lane change, never MERGE, even when interactive),
   `REJECT_NO_AUTHORITATIVE_TYPE`, `REJECT_INSUFFICIENT_SOURCE_HISTORY`,
   `REJECT_NO_PHYSICAL_TRANSITION`, `REJECT_TARGET_NOT_STABLE`,
   `REJECT_NO_INTERACTION`, `REJECT_GAP_ORDER_NOT_PERSISTENT`. A
   topology-connected target that is neither multi-entry nor a roundabout
   returns `V2Decision.REVIEW` (`REVIEW_AMBIGUOUS_TOPOLOGY`) instead of
   guessing. Only interactive topological merges and roundabout entries can
   be automatically accepted.
4. Use the candidate distribution and labelled review sample to calibrate
   `configs/merge_v2.yaml`, record the rationale, and only then change
   `calibration_status` from `pending` to `calibrated`. Manifest construction
   fails while calibration is pending.
5. `scripts/extract_merge_v2_evidence.py` renders every candidate --
   ACCEPT, REJECT, and REVIEW alike -- to `<review-dir>/{accept,reject,
   review}/<candidate_id>.png` for human audit; a reject or review call
   needs the same visual evidence checked for correctness as an accept.
   Set `manual_validation=CONFIRMED_MERGE` only after reviewing topology,
   ego lateral motion, and surrounding-vehicle trajectories for the
   `accept/` panels.
6. Store audited records as JSONL and build canonical manifests:

   ```bash
   PYTHONPATH=. python scripts/build_merge_v2_manifest.py \
     --evidence-jsonl data/manifests/v2/audited_evidence.jsonl
   ```

7. Run the validity floor before PPO training:

   ```bash
   PYTHONPATH=. python scripts/evaluate_v2_baselines.py \
     --dataset-config <matching-tfexample-dataset.yaml> \
     --split validation
   ```

Always-Keep producing any Success is a hard failure. Always-Merge performance
must be reported next to FSM; a near-saturated Always-Merge result indicates a
degenerate decision set and requires another dataset audit.

## Canonical contract

- Automatic ACCEPT **and** `CONFIRMED_MERGE` are both mandatory.
- At least one target/conflict vehicle must be present in the decision window.
- A two-sided gap insertion must preserve the same front/rear ordering for at
  least five frames after entry.
- Scenario IDs, not maneuver IDs, own TRAIN/TUNE/VALIDATION assignment.
- Future logged frames are permitted only in offline label construction and
  are never added to the online 14D observation.
- Existing v1 artifacts remain reproducible historical diagnostics but must
  never be aggregated with v2 results.
