import dataclasses
import math
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from waymax import config
from waymax import dataloader
from waymax import visualization
from waymax.metrics import metric_factory


# ======================================================================
# Object type mapping
# ======================================================================

OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER",
}


# ======================================================================
# Roadgraph map element type mapping
# ======================================================================

MAP_ELEMENT_TYPE_NAMES = {
    -1: "UNKNOWN",
    0: "LANE_UNDEFINED",
    1: "LANE_FREEWAY",
    2: "LANE_SURFACE_STREET",
    3: "LANE_BIKE_LANE",
    5: "ROAD_LINE_UNKNOWN",
    6: "ROAD_LINE_BROKEN_SINGLE_WHITE",
    7: "ROAD_LINE_SOLID_SINGLE_WHITE",
    8: "ROAD_LINE_SOLID_DOUBLE_WHITE",
    9: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
    10: "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
    11: "ROAD_LINE_SOLID_SINGLE_YELLOW",
    12: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
    13: "ROAD_LINE_PASSING_DOUBLE_YELLOW",
    14: "ROAD_EDGE_UNKNOWN",
    15: "ROAD_EDGE_BOUNDARY",
    16: "ROAD_EDGE_MEDIAN",
    17: "STOP_SIGN",
    18: "CROSSWALK",
    19: "SPEED_BUMP",
    20: "DRIVEWAY",
}


def main():

    print("=" * 70)
    print("ITS Merge Decision - Phase 0")
    print("=" * 70)

    # ==================================================================
    # 1. JAX / GPU
    # ==================================================================

    print("\n[1] JAX Devices")
    print(jax.devices())

    # ==================================================================
    # 2. WOMD Dataset
    # ==================================================================

    dataset_config = dataclasses.replace(
        config.WOD_1_3_1_VALIDATION,
        path=(
            "data/womd/validation/"
            "validation_tfexample.tfrecord-00000-of-00150"
        ),
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    print("\n[2] Dataset")
    print("Path:", dataset_config.path)
    print("Max objects:", dataset_config.max_num_objects)

    # ==================================================================
    # 3. Waymax Dataloader
    # ==================================================================

    print("\n[3] Creating Waymax dataloader...")

    scenario_generator = dataloader.simulator_state_generator(
        config=dataset_config
    )

    # ==================================================================
    # 4. First WOMD Scene
    # ==================================================================

    print("[4] Loading first WOMD scenario...")

    scenario = next(scenario_generator)

    print("\n[5] Scene loaded successfully!")
    print("SimulatorState type:", type(scenario))
    print("Current timestep:", int(scenario.timestep))
    print("Number of objects:", scenario.num_objects)
    print("Remaining timesteps:", int(scenario.remaining_timesteps))
    print("Trajectory shape:", scenario.sim_trajectory.x.shape)

    if scenario.roadgraph_points is not None:
        print(
            "Roadgraph shape:",
            scenario.roadgraph_points.xyz.shape,
        )
    else:
        print("Roadgraph shape: None")

    # ==================================================================
    # 5. Current Object State
    # ==================================================================

    current = scenario.current_sim_trajectory
    metadata = scenario.object_metadata

    x = np.asarray(current.x[:, 0])
    y = np.asarray(current.y[:, 0])
    yaw = np.asarray(current.yaw[:, 0])

    vel_x = np.asarray(current.vel_x[:, 0])
    vel_y = np.asarray(current.vel_y[:, 0])
    speed = np.asarray(current.speed[:, 0])

    length = np.asarray(current.length[:, 0])
    width = np.asarray(current.width[:, 0])

    valid = np.asarray(
        current.valid[:, 0]
    ).astype(bool)

    ids = np.asarray(metadata.ids)
    object_types = np.asarray(metadata.object_types)

    is_sdc = np.asarray(
        metadata.is_sdc
    ).astype(bool)

    # ==================================================================
    # 6. Ego / SDC
    # ==================================================================

    sdc_indices = np.flatnonzero(is_sdc)

    if len(sdc_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one SDC, but found {len(sdc_indices)}"
        )

    ego_idx = int(sdc_indices[0])

    if not valid[ego_idx]:
        raise RuntimeError(
            "SDC is invalid at the current timestep."
        )

    ego_x = float(x[ego_idx])
    ego_y = float(y[ego_idx])

    ego_yaw = float(yaw[ego_idx])

    ego_vx = float(vel_x[ego_idx])
    ego_vy = float(vel_y[ego_idx])

    ego_speed = float(speed[ego_idx])

    object_type_name = OBJECT_TYPE_NAMES.get(
        int(object_types[ego_idx]),
        "UNKNOWN",
    )

    print("\n" + "-" * 70)
    print("[6] Ego / SDC State")
    print("-" * 70)

    print(f"Index        : {ego_idx}")
    print(f"Object ID    : {int(ids[ego_idx])}")
    print(f"Object Type  : {object_type_name}")

    print(
        f"Position     : "
        f"x={ego_x:.3f} m, "
        f"y={ego_y:.3f} m"
    )

    print(
        f"Velocity     : "
        f"vx={ego_vx:.3f} m/s, "
        f"vy={ego_vy:.3f} m/s"
    )

    print(f"Speed        : {ego_speed:.3f} m/s")
    print(f"Yaw          : {ego_yaw:.3f} rad")
    print(f"Yaw          : {math.degrees(ego_yaw):.2f} deg")

    print(
        f"Vehicle Size : "
        f"L={float(length[ego_idx]):.3f} m, "
        f"W={float(width[ego_idx]):.3f} m"
    )

    # ==================================================================
    # 7. Surrounding Agents
    # ==================================================================

    surrounding_agents = []

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    for i in range(scenario.num_objects):

        if i == ego_idx:
            continue

        if not valid[i]:
            continue

        dx = float(x[i] - ego_x)
        dy = float(y[i] - ego_y)

        distance = math.hypot(dx, dy)

        # --------------------------------------------------------------
        # Global -> Ego-local
        #
        # rel_x > 0 : Ego 앞
        # rel_x < 0 : Ego 뒤
        #
        # rel_y > 0 : Ego 왼쪽
        # rel_y < 0 : Ego 오른쪽
        # --------------------------------------------------------------

        rel_x = (
            cos_yaw * dx
            + sin_yaw * dy
        )

        rel_y = (
            -sin_yaw * dx
            + cos_yaw * dy
        )

        surrounding_agents.append(
            {
                "index": i,
                "id": int(ids[i]),
                "type": OBJECT_TYPE_NAMES.get(
                    int(object_types[i]),
                    "UNKNOWN",
                ),
                "x": float(x[i]),
                "y": float(y[i]),
                "speed": float(speed[i]),
                "distance": distance,
                "rel_x": rel_x,
                "rel_y": rel_y,
            }
        )

    surrounding_agents.sort(
        key=lambda agent: agent["distance"]
    )

    print("\n" + "-" * 70)
    print("[7] Surrounding Agents")
    print("-" * 70)

    print(
        f"Valid surrounding agents: "
        f"{len(surrounding_agents)}"
    )

    print("\nNearest 10 agents:\n")

    print(
        f"{'IDX':>4} "
        f"{'ID':>6} "
        f"{'TYPE':>10} "
        f"{'DIST[m]':>10} "
        f"{'REL_X[m]':>10} "
        f"{'REL_Y[m]':>10} "
        f"{'SPEED':>10}"
    )

    print("-" * 70)

    for agent in surrounding_agents[:10]:

        print(
            f"{agent['index']:>4} "
            f"{agent['id']:>6} "
            f"{agent['type']:>10} "
            f"{agent['distance']:>10.2f} "
            f"{agent['rel_x']:>10.2f} "
            f"{agent['rel_y']:>10.2f} "
            f"{agent['speed']:>10.2f}"
        )

    print("\n" + "=" * 70)
    print("EGO / SURROUNDING STATE EXTRACTION: PASS")
    print("=" * 70)

    # ==================================================================
    # 8. Roadgraph Inspection
    # ==================================================================

    roadgraph = scenario.roadgraph_points

    if roadgraph is None:
        raise RuntimeError(
            "Roadgraph data is not available."
        )

    rg_x = np.asarray(roadgraph.x)
    rg_y = np.asarray(roadgraph.y)

    rg_types = np.asarray(
        roadgraph.types
    )

    rg_ids = np.asarray(
        roadgraph.ids
    )

    rg_valid = np.asarray(
        roadgraph.valid
    ).astype(bool)

    num_total_points = roadgraph.num_points

    num_valid_points = int(
        np.sum(rg_valid)
    )

    print("\n" + "-" * 70)
    print("[8] Roadgraph")
    print("-" * 70)

    print(
        f"Total roadgraph points : "
        f"{num_total_points}"
    )

    print(
        f"Valid roadgraph points : "
        f"{num_valid_points}"
    )

    print(
        f"XYZ shape              : "
        f"{roadgraph.xyz.shape}"
    )

    print(
        f"Direction XYZ shape    : "
        f"{roadgraph.dir_xyz.shape}"
    )

    # ==================================================================
    # 8-1. Map Element Distribution
    # ==================================================================

    print("\nMap element distribution:")

    valid_types = rg_types[rg_valid]

    unique_types, type_counts = np.unique(
        valid_types,
        return_counts=True,
    )

    for type_id, count in zip(
        unique_types,
        type_counts,
    ):

        type_id = int(type_id)

        type_name = MAP_ELEMENT_TYPE_NAMES.get(
            type_id,
            f"UNKNOWN_TYPE_{type_id}",
        )

        print(
            f"  {type_id:>2} | "
            f"{type_name:<35} "
            f"{int(count):>6} points"
        )

    # ==================================================================
    # 8-2. Ego -> Roadgraph Distance
    # ==================================================================

    dx = rg_x - ego_x
    dy = rg_y - ego_y

    rg_distance = np.hypot(
        dx,
        dy,
    )

    rg_rel_x = (
        cos_yaw * dx
        + sin_yaw * dy
    )

    rg_rel_y = (
        -sin_yaw * dx
        + cos_yaw * dy
    )

    valid_distance = np.where(
        rg_valid,
        rg_distance,
        np.inf,
    )

    # ==================================================================
    # 8-3. Roadgraph within 50 m
    # ==================================================================

    radius = 50.0

    nearby_mask = (
        rg_valid
        & (rg_distance <= radius)
    )

    nearby_count = int(
        np.sum(nearby_mask)
    )

    nearby_feature_count = len(
        np.unique(
            rg_ids[nearby_mask]
        )
    )

    print(
        f"\nRoadgraph within "
        f"{radius:.0f} m of Ego:"
    )

    print(f"  Points   : {nearby_count}")
    print(f"  Features : {nearby_feature_count}")

    # ==================================================================
    # 8-4. Nearest Roadgraph Points
    # ==================================================================

    nearest_indices = np.argsort(
        valid_distance
    )[:10]

    print("\nNearest 10 roadgraph points:\n")

    print(
        f"{'IDX':>6} "
        f"{'ID':>6} "
        f"{'TYPE':>28} "
        f"{'DIST':>8} "
        f"{'REL_X':>8} "
        f"{'REL_Y':>8}"
    )

    print("-" * 70)

    for idx in nearest_indices:

        type_id = int(
            rg_types[idx]
        )

        type_name = MAP_ELEMENT_TYPE_NAMES.get(
            type_id,
            f"TYPE_{type_id}",
        )

        print(
            f"{idx:>6} "
            f"{int(rg_ids[idx]):>6} "
            f"{type_name:>28} "
            f"{rg_distance[idx]:>8.2f} "
            f"{rg_rel_x[idx]:>8.2f} "
            f"{rg_rel_y[idx]:>8.2f}"
        )

    print("\n" + "=" * 70)
    print("ROADGRAPH INSPECTION: PASS")
    print("=" * 70)

    # ==================================================================
    # 9. Top-view Visualization
    # ==================================================================

    print("\n" + "-" * 70)
    print("[9] Top-view Visualization")
    print("-" * 70)

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "scene_000_topview.png"
    )

    print(
        "Rendering Waymax top-view..."
    )

    image = visualization.plot_simulator_state(
        scenario,
        use_log_traj=True,
    )

    print(
        "Image shape:",
        image.shape,
    )

    plt.imsave(
        output_path,
        image,
    )

    if not output_path.exists():
        raise RuntimeError(
            "Visualization file was not created."
        )

    print(
        f"Saved visualization: "
        f"{output_path}"
    )

    print("\n" + "=" * 70)
    print("TOP-VIEW VISUALIZATION: PASS")
    print("=" * 70)

    # ==================================================================
    # 10. Basic Waymax Metrics
    # ==================================================================

    print("\n" + "-" * 70)
    print("[10] Basic Waymax Metrics")
    print("-" * 70)

    metrics_config = config.MetricsConfig(
        metrics_to_run=(
            "log_divergence",
            "overlap",
            "offroad",
        )
    )

    print(
        "Metrics:",
        metrics_config.metrics_to_run,
    )

    print(
        "Computing metrics..."
    )

    metric_results = metric_factory.run_metrics(
        simulator_state=scenario,
        metrics_config=metrics_config,
    )

    print("\nEgo / SDC metrics:\n")

    for metric_name in metrics_config.metrics_to_run:

        result = metric_results[
            metric_name
        ]

        metric_values = np.asarray(
            result.value
        )

        metric_valid = np.asarray(
            result.valid
        ).astype(bool)

        if metric_valid[ego_idx]:

            ego_value = float(
                metric_values[ego_idx]
            )

            print(
                f"  {metric_name:<20}: "
                f"{ego_value:.6f}"
            )

        else:

            print(
                f"  {metric_name:<20}: "
                f"INVALID"
            )

    # ==================================================================
    # 10-1. Metric summary for all valid objects
    # ==================================================================

    print("\nMetric summary for all objects:\n")

    print(
        f"{'METRIC':<22}"
        f"{'VALID':>8}"
        f"{'MEAN':>12}"
        f"{'MAX':>12}"
    )

    print("-" * 54)

    for metric_name in metrics_config.metrics_to_run:

        result = metric_results[
            metric_name
        ]

        metric_values = np.asarray(
            result.value
        )

        metric_valid = np.asarray(
            result.valid
        ).astype(bool)

        valid_values = metric_values[
            metric_valid
        ]

        num_valid = len(
            valid_values
        )

        if num_valid > 0:

            mean_value = float(
                np.mean(valid_values)
            )

            max_value = float(
                np.max(valid_values)
            )

        else:

            mean_value = float("nan")
            max_value = float("nan")

        print(
            f"{metric_name:<22}"
            f"{num_valid:>8}"
            f"{mean_value:>12.6f}"
            f"{max_value:>12.6f}"
        )

    # ==================================================================
    # 10-2. Safety interpretation
    # ==================================================================

    overlap_result = metric_results[
        "overlap"
    ]

    offroad_result = metric_results[
        "offroad"
    ]

    overlap_values = np.asarray(
        overlap_result.value
    )

    overlap_valid = np.asarray(
        overlap_result.valid
    ).astype(bool)

    offroad_values = np.asarray(
        offroad_result.value
    )

    offroad_valid = np.asarray(
        offroad_result.valid
    ).astype(bool)

    ego_overlap = None
    ego_offroad = None

    if overlap_valid[ego_idx]:
        ego_overlap = float(
            overlap_values[ego_idx]
        )

    if offroad_valid[ego_idx]:
        ego_offroad = float(
            offroad_values[ego_idx]
        )

    print("\nEgo safety status:")

    if ego_overlap is not None:

        if ego_overlap >= 0.5:
            print(
                "  Overlap : DETECTED"
            )
        else:
            print(
                "  Overlap : CLEAR"
            )

    else:
        print(
            "  Overlap : INVALID"
        )

    if ego_offroad is not None:

        if ego_offroad >= 0.5:
            print(
                "  Offroad : DETECTED"
            )
        else:
            print(
                "  Offroad : CLEAR"
            )

    else:
        print(
            "  Offroad : INVALID"
        )

    print("\n" + "=" * 70)
    print("BASIC METRICS: PASS")
    print("=" * 70)

    # ==================================================================
    # Phase 0 Current Status
    # ==================================================================

    print("\n" + "=" * 70)
    print("PHASE 0 CURRENT VALIDATION")
    print("=" * 70)

    print("[PASS] JAX / CUDA")
    print("[PASS] WOMD scene loading")
    print("[PASS] Ego state extraction")
    print("[PASS] Surrounding agent extraction")
    print("[PASS] Roadgraph inspection")
    print("[PASS] Top-view visualization")
    print("[PASS] Basic Waymax metrics")

    print("=" * 70)


if __name__ == "__main__":
    main()