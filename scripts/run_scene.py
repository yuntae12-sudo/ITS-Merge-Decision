import dataclasses
import math

import jax
import numpy as np

from waymax import config
from waymax import dataloader


OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER",
}


def main():
    print("=" * 70)
    print("ITS Merge Decision - Phase 0")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. JAX device
    # ------------------------------------------------------------------
    print("\n[1] JAX Devices")
    print(jax.devices())

    # ------------------------------------------------------------------
    # 2. Dataset config
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Load one WOMD scenario
    # ------------------------------------------------------------------
    print("\n[3] Creating Waymax dataloader...")

    scenario_generator = dataloader.simulator_state_generator(
        config=dataset_config
    )

    print("[4] Loading first WOMD scenario...")

    scenario = next(scenario_generator)

    print("\n[5] Scene loaded successfully!")
    print("SimulatorState type:", type(scenario))
    print("Current timestep:", int(scenario.timestep))
    print("Number of objects:", scenario.num_objects)
    print("Remaining timesteps:", int(scenario.remaining_timesteps))
    print("Trajectory shape:", scenario.sim_trajectory.x.shape)
    print("Roadgraph shape:", scenario.roadgraph_points.xyz.shape)

    # ------------------------------------------------------------------
    # 4. Current state
    # ------------------------------------------------------------------
    current = scenario.current_sim_trajectory
    metadata = scenario.object_metadata

    # current trajectory shape:
    # (num_objects, 1)
    x = np.asarray(current.x[:, 0])
    y = np.asarray(current.y[:, 0])
    yaw = np.asarray(current.yaw[:, 0])
    vel_x = np.asarray(current.vel_x[:, 0])
    vel_y = np.asarray(current.vel_y[:, 0])
    speed = np.asarray(current.speed[:, 0])
    length = np.asarray(current.length[:, 0])
    width = np.asarray(current.width[:, 0])
    valid = np.asarray(current.valid[:, 0]).astype(bool)

    ids = np.asarray(metadata.ids)
    object_types = np.asarray(metadata.object_types)
    is_sdc = np.asarray(metadata.is_sdc).astype(bool)

    # ------------------------------------------------------------------
    # 5. Find SDC / Ego
    # ------------------------------------------------------------------
    sdc_indices = np.flatnonzero(is_sdc)

    if len(sdc_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one SDC, but found {len(sdc_indices)}"
        )

    ego_idx = int(sdc_indices[0])

    if not valid[ego_idx]:
        raise RuntimeError("SDC is invalid at current timestep.")

    ego_x = float(x[ego_idx])
    ego_y = float(y[ego_idx])
    ego_yaw = float(yaw[ego_idx])
    ego_vx = float(vel_x[ego_idx])
    ego_vy = float(vel_y[ego_idx])
    ego_speed = float(speed[ego_idx])

    print("\n" + "-" * 70)
    print("[6] Ego / SDC State")
    print("-" * 70)

    print(f"Index        : {ego_idx}")
    print(f"Object ID    : {int(ids[ego_idx])}")
    print(
        f"Object Type  : "
        f"{OBJECT_TYPE_NAMES.get(int(object_types[ego_idx]), 'UNKNOWN')}"
    )

    print(f"Position     : x={ego_x:.3f} m, y={ego_y:.3f} m")
    print(
        f"Velocity     : vx={ego_vx:.3f} m/s, "
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

    # ------------------------------------------------------------------
    # 6. Surrounding agents
    # ------------------------------------------------------------------
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

        # Global coordinates -> Ego-local coordinates
        #
        # rel_x > 0 : in front of Ego
        # rel_x < 0 : behind Ego
        # rel_y > 0 : left side
        # rel_y < 0 : right side
        rel_x = cos_yaw * dx + sin_yaw * dy
        rel_y = -sin_yaw * dx + cos_yaw * dy

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

    # 가까운 순으로 정렬
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

    # ------------------------------------------------------------------
    # 7. Final validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EGO / SURROUNDING STATE EXTRACTION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()