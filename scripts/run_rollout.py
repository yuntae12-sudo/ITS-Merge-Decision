import csv
import dataclasses
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from waymax import config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization
from waymax.metrics import metric_factory


# ======================================================================
# Phase 0 paths
# ======================================================================

DATASET_PATH = (
    "data/womd/validation/"
    "validation_tfexample.tfrecord-00000-of-00150"
)

SCENE_NAME = "scene_000"

OUTPUT_ROOT = Path("outputs/phase0") / SCENE_NAME


def to_uint8(image):
    """Convert visualization image to uint8 for PIL."""

    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def get_ego_state(state, ego_idx):
    """Extract current Ego state."""

    current = state.current_sim_trajectory

    valid = bool(
        np.asarray(
            current.valid[ego_idx, 0]
        )
    )

    if not valid:
        return {
            "valid": False,
            "x": np.nan,
            "y": np.nan,
            "speed": np.nan,
            "yaw": np.nan,
        }

    x = float(
        np.asarray(
            current.x[ego_idx, 0]
        )
    )

    y = float(
        np.asarray(
            current.y[ego_idx, 0]
        )
    )

    speed = float(
        np.asarray(
            current.speed[ego_idx, 0]
        )
    )

    yaw = float(
        np.asarray(
            current.yaw[ego_idx, 0]
        )
    )

    return {
        "valid": True,
        "x": x,
        "y": y,
        "speed": speed,
        "yaw": yaw,
    }


def get_ego_metric(
    metric_results,
    metric_name,
    ego_idx,
):
    """Return Ego metric value and validity."""

    result = metric_results[
        metric_name
    ]

    values = np.asarray(
        result.value
    )

    valid = np.asarray(
        result.valid
    ).astype(bool)

    if not valid[ego_idx]:
        return np.nan, False

    return float(values[ego_idx]), True


def main():

    print("=" * 70)
    print("ITS Merge Decision - Phase 0")
    print("Basic Log Playback Rollout")
    print("=" * 70)

    # ==================================================================
    # 1. JAX / GPU
    # ==================================================================

    print("\n[1] JAX Devices")
    print(jax.devices())

    # ==================================================================
    # 2. Dataset Config
    # ==================================================================

    dataset_config = dataclasses.replace(
        config.WOD_1_3_1_VALIDATION,
        path=DATASET_PATH,
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    print("\n[2] Dataset")
    print("Path:", dataset_config.path)

    # ==================================================================
    # 3. Load Scenario
    # ==================================================================

    print("\n[3] Loading WOMD scenario...")

    scenario_generator = (
        dataloader.simulator_state_generator(
            config=dataset_config
        )
    )

    scenario = next(
        scenario_generator
    )

    print(
        "Scene loaded."
    )

    print(
        "Initial timestep:",
        int(scenario.timestep),
    )

    print(
        "Remaining timesteps:",
        int(scenario.remaining_timesteps),
    )

    # ==================================================================
    # 4. Find Ego / SDC
    # ==================================================================

    is_sdc = np.asarray(
        scenario.object_metadata.is_sdc
    ).astype(bool)

    sdc_indices = np.flatnonzero(
        is_sdc
    )

    if len(sdc_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one SDC, "
            f"but found {len(sdc_indices)}"
        )

    ego_idx = int(
        sdc_indices[0]
    )

    ego_id = int(
        np.asarray(
            scenario.object_metadata.ids[
                ego_idx
            ]
        )
    )

    print("\n[4] Ego / SDC")

    print(
        "Ego index:",
        ego_idx,
    )

    print(
        "Ego object ID:",
        ego_id,
    )

    # ==================================================================
    # 5. Metrics Config
    # ==================================================================

    metrics_config = config.MetricsConfig(
        metrics_to_run=(
            "log_divergence",
            "overlap",
            "offroad",
        )
    )

    print("\n[5] Metrics")

    print(
        metrics_config.metrics_to_run
    )

    # ==================================================================
    # 6. Output Directory
    # ==================================================================

    rollout_dir = OUTPUT_ROOT / "rollout"
    snapshot_dir = OUTPUT_ROOT / "snapshots"
    metrics_dir = OUTPUT_ROOT / "metrics"

    rollout_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gif_path = (
        rollout_dir
        / "rollout.gif"
    )

    csv_path = (
        metrics_dir
        / "rollout_metrics.csv"
    )

    # ==================================================================
    # 7. Rollout Setup
    # ==================================================================

    state = scenario

    num_steps = int(
        scenario.remaining_timesteps
    )

    final_timestep = (
        int(scenario.timestep)
        + num_steps
    )

    frames = []
    rollout_records = []

    print("\n" + "-" * 70)
    print("[6] Starting Log Playback Rollout")
    print("-" * 70)

    print(
        f"Rollout range: "
        f"t={int(state.timestep)} "
        f"-> t={final_timestep}"
    )

    # ==================================================================
    # 8. Rollout Loop
    # ==================================================================

    for step in range(
        num_steps + 1
    ):

        timestep = int(
            state.timestep
        )

        # --------------------------------------------------------------
        # Current Ego State
        # --------------------------------------------------------------

        ego_state = get_ego_state(
            state,
            ego_idx,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metric_results = (
            metric_factory.run_metrics(
                simulator_state=state,
                metrics_config=metrics_config,
            )
        )

        log_divergence, log_valid = (
            get_ego_metric(
                metric_results,
                "log_divergence",
                ego_idx,
            )
        )

        overlap, overlap_valid = (
            get_ego_metric(
                metric_results,
                "overlap",
                ego_idx,
            )
        )

        offroad, offroad_valid = (
            get_ego_metric(
                metric_results,
                "offroad",
                ego_idx,
            )
        )

        # --------------------------------------------------------------
        # Record
        # --------------------------------------------------------------

        rollout_records.append(
            {
                "timestep": timestep,
                "ego_valid": ego_state["valid"],
                "ego_x": ego_state["x"],
                "ego_y": ego_state["y"],
                "ego_speed": ego_state["speed"],
                "ego_yaw": ego_state["yaw"],
                "log_divergence": log_divergence,
                "log_divergence_valid": log_valid,
                "overlap": overlap,
                "overlap_valid": overlap_valid,
                "offroad": offroad,
                "offroad_valid": offroad_valid,
            }
        )

        # --------------------------------------------------------------
        # Visualization Frame
        # --------------------------------------------------------------

        frame = (
            visualization.plot_simulator_state(
                state,
                use_log_traj=True,
            )
        )

        frame_uint8 = to_uint8(
            frame
        )

        frames.append(
            Image.fromarray(
                frame_uint8
            )
        )

        # --------------------------------------------------------------
        # Snapshot
        # --------------------------------------------------------------

        snapshot_path = (
            snapshot_dir
            / f"timestep_{timestep:03d}.png"
        )

        Image.fromarray(
            frame_uint8
        ).save(
            snapshot_path
        )

        # --------------------------------------------------------------
        # Console Log
        # --------------------------------------------------------------

        if (
            timestep % 10 == 0
            or timestep == final_timestep
        ):

            print(
                f"[t={timestep:02d}] "
                f"x={ego_state['x']:.2f} "
                f"y={ego_state['y']:.2f} "
                f"speed={ego_state['speed']:.2f} "
                f"log_div={log_divergence:.4f} "
                f"overlap={overlap:.1f} "
                f"offroad={offroad:.1f}"
            )

        # --------------------------------------------------------------
        # Last timestep -> stop
        # --------------------------------------------------------------

        if step == num_steps:
            break

        # --------------------------------------------------------------
        # Advance one timestep using WOMD logged trajectory
        # --------------------------------------------------------------

        state = (
            datatypes.update_state_by_log(
                state,
                num_steps=1,
            )
        )

    # ==================================================================
    # 9. Validation
    # ==================================================================

    print("\n" + "-" * 70)
    print("[7] Rollout Validation")
    print("-" * 70)

    print(
        "Final timestep:",
        int(state.timestep),
    )

    print(
        "State is done:",
        bool(state.is_done),
    )

    if int(state.timestep) != final_timestep:
        raise RuntimeError(
            "Rollout did not reach "
            "the expected final timestep."
        )

    if not bool(state.is_done):
        raise RuntimeError(
            "SimulatorState did not "
            "reach done state."
        )

    # ==================================================================
    # 10. Save CSV
    # ==================================================================

    print("\n[8] Saving rollout metrics...")

    fieldnames = [
        "timestep",
        "ego_valid",
        "ego_x",
        "ego_y",
        "ego_speed",
        "ego_yaw",
        "log_divergence",
        "log_divergence_valid",
        "overlap",
        "overlap_valid",
        "offroad",
        "offroad_valid",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rollout_records
        )

    print(
        f"Saved CSV: {csv_path}"
    )

    # ==================================================================
    # 11. Save GIF
    # ==================================================================

    print("\n[9] Saving rollout GIF...")

    if not frames:
        raise RuntimeError(
            "No visualization frames were generated."
        )

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    print(
        f"Saved GIF: {gif_path}"
    )

    # ==================================================================
    # 12. Rollout Summary
    # ==================================================================

    log_values = np.asarray(
        [
            record["log_divergence"]
            for record in rollout_records
            if record[
                "log_divergence_valid"
            ]
        ],
        dtype=float,
    )

    overlap_values = np.asarray(
        [
            record["overlap"]
            for record in rollout_records
            if record[
                "overlap_valid"
            ]
        ],
        dtype=float,
    )

    offroad_values = np.asarray(
        [
            record["offroad"]
            for record in rollout_records
            if record[
                "offroad_valid"
            ]
        ],
        dtype=float,
    )

    print("\n" + "-" * 70)
    print("[10] Rollout Summary")
    print("-" * 70)

    if len(log_values) > 0:
        print(
            "Max log divergence :",
            f"{np.max(log_values):.6f}",
        )

    if len(overlap_values) > 0:
        print(
            "Overlap detected   :",
            bool(
                np.any(
                    overlap_values >= 0.5
                )
            ),
        )

    if len(offroad_values) > 0:
        print(
            "Offroad detected   :",
            bool(
                np.any(
                    offroad_values >= 0.5
                )
            ),
        )

    print(
        "Frames generated     :",
        len(frames),
    )

    print(
        "Records saved        :",
        len(rollout_records),
    )

    print("\n" + "=" * 70)
    print("LOG PLAYBACK ROLLOUT: PASS")
    print("=" * 70)

    print("\nOutputs:")
    print(f"  GIF       : {gif_path}")
    print(f"  CSV       : {csv_path}")
    print(f"  Snapshots : {snapshot_dir}")

    print("\n" + "=" * 70)
    print("PHASE 0 CURRENT VALIDATION")
    print("=" * 70)

    print("[PASS] JAX / CUDA")
    print("[PASS] WOMD scene loading")
    print("[PASS] Ego / surrounding states")
    print("[PASS] Roadgraph")
    print("[PASS] Top-view visualization")
    print("[PASS] Basic metrics")
    print("[PASS] Log playback rollout")

    print("=" * 70)


if __name__ == "__main__":
    main()