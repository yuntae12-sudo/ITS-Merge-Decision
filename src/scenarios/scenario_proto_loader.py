"""WOMD Scenario-protobuf topology loader used by dataset schema v2.

The repository's existing ``*_tfexample.tfrecord`` files do not contain lane
connectivity.  This loader deliberately accepts only WOMD Scenario protobuf
TFRecords and fails closed when the optional official package is unavailable.
"""

import dataclasses
from pathlib import PurePosixPath
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np

from src.scenarios.merge_v2 import TopologyEvidence


@dataclasses.dataclass(frozen=True)
class ProtoLane:
    lane_id: int
    entry_lanes: Tuple[int, ...]
    exit_lanes: Tuple[int, ...]
    polyline_xy: np.ndarray
    left_neighbor_ids: Tuple[int, ...] = ()
    right_neighbor_ids: Tuple[int, ...] = ()
    lane_type: str = "TYPE_UNDEFINED"


@dataclasses.dataclass(frozen=True)
class ScenarioTopology:
    scenario_id: str
    lanes: Dict[int, ProtoLane]

    def evidence(
        self,
        source_lane_id: int,
        target_lane_id: int,
        **geometry,
    ) -> TopologyEvidence:
        source = self.lanes[source_lane_id]
        target = self.lanes[target_lane_id]
        geometry.setdefault("target_is_roundabout", self.lane_is_in_cycle(target_lane_id))
        geometry.setdefault("source_exit_lane_count", len(set(source.exit_lanes)))
        geometry.setdefault(
            "target_polyline_point_count", int(target.polyline_xy.shape[0])
        )
        geometry.setdefault(
            "is_lane_change_target",
            self._is_lane_change_target(source, target, target_lane_id),
        )
        geometry.setdefault(
            "is_intersection_transition",
            self._is_intersection_transition(source, target, target_lane_id, geometry),
        )
        return TopologyEvidence(
            source_lane_id=source_lane_id,
            target_lane_id=target_lane_id,
            source_exit_lane_ids=source.exit_lanes,
            target_entry_lane_ids=target.entry_lanes,
            source_exits_to_target=(target_lane_id in source.exit_lanes),
            **geometry,
        )

    @staticmethod
    def _is_lane_change_target(source: "ProtoLane", target: "ProtoLane", target_lane_id: int) -> bool:
        """True when ``target`` is a same-direction adjacent lane
        (``LaneNeighbor``) of ``source`` and topology does not also
        connect them via entry_lanes/exit_lanes -- a parallel-lane
        change, never a merge."""

        is_neighbor = (
            target_lane_id in source.left_neighbor_ids
            or target_lane_id in source.right_neighbor_ids
        )
        return is_neighbor and target_lane_id not in source.exit_lanes

    @staticmethod
    def _is_intersection_transition(
        source: "ProtoLane", target: "ProtoLane", target_lane_id: int, geometry: dict
    ) -> bool:
        """A topology-connected transition (``target`` in
        ``source.exit_lanes``) with a large heading change and no lane
        convergence (single target entry) is a turning movement through
        an intersection, not a merge -- merges converge near-parallel
        streams, they do not turn sharply."""

        if target_lane_id not in source.exit_lanes:
            return False
        if len(set(target.entry_lanes)) >= 2:
            return False
        heading_diff = geometry.get("terminal_heading_difference_deg", 0.0)
        return heading_diff >= 30.0

    def lane_is_in_cycle(self, lane_id: int, max_hops: int = 24) -> bool:
        """Topology-only roundabout cue; canonical acceptance still needs review."""

        frontier = [(lane_id, 0)]
        visited = set()
        while frontier:
            current, depth = frontier.pop()
            if depth >= max_hops:
                continue
            for nxt in self.lanes.get(current, ProtoLane(current, (), (), np.empty((0, 2)))).exit_lanes:
                if nxt == lane_id and depth >= 1:
                    return True
                edge = (current, nxt)
                if edge not in visited:
                    visited.add(edge)
                    frontier.append((nxt, depth + 1))
        return False


def _neighbor_ids(neighbors) -> Tuple[int, ...]:
    """``LaneCenter.left_neighbors``/``right_neighbors`` are repeated
    ``LaneNeighbor`` messages; only the neighbor lane's ``feature_id`` is
    topology (the overlap-range indices are not used by this repo)."""

    return tuple(int(n.feature_id) for n in neighbors)


def _lane_type_name(lane) -> str:
    """Duck-typed friendly: real proto lanes expose an enum descriptor
    that can resolve ``type`` to its symbolic name; a test double
    lacking ``type``/``DESCRIPTOR`` falls back to ``TYPE_UNDEFINED``."""

    lane_type = getattr(lane, "type", None)
    if lane_type is None:
        return "TYPE_UNDEFINED"
    try:
        return lane.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[
            lane_type
        ].name
    except AttributeError:
        return str(lane_type)


def scenario_to_topology(scenario) -> ScenarioTopology:
    """Convert a duck-typed official ``Scenario`` protobuf to topology."""

    lanes = {}
    for feature in scenario.map_features:
        if not feature.HasField("lane"):
            continue
        lane = feature.lane
        xy = np.asarray([(p.x, p.y) for p in lane.polyline], dtype=np.float64)
        lanes[int(feature.id)] = ProtoLane(
            lane_id=int(feature.id),
            entry_lanes=tuple(int(x) for x in lane.entry_lanes),
            exit_lanes=tuple(int(x) for x in lane.exit_lanes),
            polyline_xy=xy,
            left_neighbor_ids=_neighbor_ids(getattr(lane, "left_neighbors", ())),
            right_neighbor_ids=_neighbor_ids(getattr(lane, "right_neighbors", ())),
            lane_type=_lane_type_name(lane),
        )
    return ScenarioTopology(scenario_id=str(scenario.scenario_id), lanes=lanes)


def iter_scenario_protobufs(paths: Iterable[str]) -> Iterator[object]:
    """Yield official WOMD ``Scenario`` messages from TFRecord shards.

    ``paths`` are passed to ``tf.data.TFRecordDataset`` as plain strings,
    never wrapped in ``pathlib.Path`` -- ``Path("gs://bucket/x")``
    silently collapses the URI's double slash to ``gs:/bucket/x``
    (verified: this previously broke every remote/GCS path this
    function was given, while never surfacing on a local path)."""

    paths = tuple(str(p) for p in paths)
    invalid = [p for p in paths if "tfexample" in PurePosixPath(p).name]
    if invalid:
        raise ValueError(
            "tf_example shards do not contain entry_lanes/exit_lanes and "
            f"cannot be topology sources: {invalid}"
        )

    try:
        import tensorflow as tf
        from src.scenarios.vendor.waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise RuntimeError(
            "WOMD Scenario protobuf support requires the vendored "
            "src.scenarios.vendor.waymo_open_dataset.protos.scenario_pb2 "
            "module (see src/scenarios/vendor/waymo_open_dataset/protos/"
            "README.md); tf_example shards cannot be used for "
            "merge_interaction_v2 topology."
        ) from exc

    for path in paths:
        for raw in tf.data.TFRecordDataset(path):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw.numpy()))
            yield scenario


def iter_scenario_topologies(paths: Iterable[str]) -> Iterator[ScenarioTopology]:
    for scenario in iter_scenario_protobufs(paths):
        yield scenario_to_topology(scenario)


def index_scenario_topologies(paths: Iterable[str]) -> Dict[str, ScenarioTopology]:
    """Build a unique scenario-id index; duplicate IDs fail loudly.

    Loads every parsed topology into memory -- fine for a handful of
    shards, but the full WOMD training scenario split is ~1000 shards
    and does not fit in memory this way. For that case, use
    ``index_scenario_topologies_for_ids`` with the (small) set of
    scenario ids actually needed, so memory scales with the wanted set
    rather than the full shard sweep."""

    index = {}
    for topology in iter_scenario_topologies(paths):
        if topology.scenario_id in index:
            raise ValueError(f"duplicate Scenario protobuf id: {topology.scenario_id}")
        index[topology.scenario_id] = topology
    return index


def index_scenario_topologies_for_ids(
    paths: Iterable[str], wanted_scenario_ids: Iterable[str]
) -> Dict[str, ScenarioTopology]:
    """Like ``index_scenario_topologies``, but discards every parsed
    scenario whose id is not in ``wanted_scenario_ids`` -- memory scales
    with the (small) wanted-id set, not with however many shards are
    swept. Intended for joining a handful of local tf_example shards'
    scenario ids against the full ~1000-shard Scenario protobuf split
    without holding every scenario's topology in RAM at once.

    Still fails loudly on a genuine duplicate id among the *wanted*
    scenarios (a real WOMD data-integrity anomaly); ids outside the
    wanted set are never held long enough to detect duplicates among
    themselves, which is an intentional trade for bounded memory. This
    always scans every path in ``paths`` to completion -- it never
    early-exits once all wanted ids are seen once, because doing so
    would also skip checking a *second* occurrence of the last-needed
    id, silently hiding exactly the duplicate this is meant to catch."""

    wanted = set(wanted_scenario_ids)
    index: Dict[str, ScenarioTopology] = {}
    for scenario in iter_scenario_protobufs(paths):
        scenario_id = str(scenario.scenario_id)
        if scenario_id not in wanted:
            continue
        if scenario_id in index:
            raise ValueError(f"duplicate Scenario protobuf id: {scenario_id}")
        index[scenario_id] = scenario_to_topology(scenario)
    return index


def iter_tfexample_scenario_ids(path: str) -> Iterator[Tuple[int, str]]:
    """Read ``scenario/id`` without asking Waymax to discard that feature."""

    import tensorflow as tf

    for record_index, raw in enumerate(tf.data.TFRecordDataset(path)):
        example = tf.train.Example.FromString(bytes(raw.numpy()))
        values = example.features.feature["scenario/id"].bytes_list.value
        if len(values) != 1:
            raise ValueError(
                f"{path} record {record_index} has {len(values)} scenario/id values"
            )
        yield record_index, values[0].decode("utf-8")


def match_tfexample_to_topology(
    tfexample_path: str, topology_by_scenario_id: Dict[str, ScenarioTopology]
) -> Iterator[Tuple[int, str, ScenarioTopology]]:
    """Join tf_example trajectory records to protobuf maps by scenario ID."""

    for record_index, scenario_id in iter_tfexample_scenario_ids(tfexample_path):
        try:
            topology = topology_by_scenario_id[scenario_id]
        except KeyError as exc:
            raise KeyError(
                f"No Scenario protobuf topology for tf_example scenario/id "
                f"{scenario_id!r} ({tfexample_path} record {record_index})"
            ) from exc
        yield record_index, scenario_id, topology


__all__ = [
    "ProtoLane",
    "ScenarioTopology",
    "scenario_to_topology",
    "iter_scenario_protobufs",
    "iter_scenario_topologies",
    "index_scenario_topologies",
    "iter_tfexample_scenario_ids",
    "match_tfexample_to_topology",
]
