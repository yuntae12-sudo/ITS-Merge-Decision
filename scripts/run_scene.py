import dataclasses

import jax

from waymax import config
from waymax import dataloader


def main():
    print("=" * 60)
    print("ITS Merge Decision - Phase 0")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. JAX device 확인
    # ---------------------------------------------------------
    print("\n[1] JAX Devices")
    print(jax.devices())

    # ---------------------------------------------------------
    # 2. WOMD v1.3.1 Validation Dataset 설정
    # ---------------------------------------------------------
    dataset_config = dataclasses.replace(
        config.WOD_1_3_1_VALIDATION,
        # Waymax main의 gs:/// 경로를 정상 GCS URI로 override
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

    # ---------------------------------------------------------
    # 3. Waymax dataloader 생성
    # ---------------------------------------------------------
    print("\n[3] Creating Waymax dataloader...")

    scenario_generator = dataloader.simulator_state_generator(
        config=dataset_config
    )

    # ---------------------------------------------------------
    # 4. 실제 WOMD scene 하나 읽기
    # ---------------------------------------------------------
    print("[4] Loading first WOMD scenario...")

    scenario = next(scenario_generator)

    print("\n[5] Scene loaded successfully!")

    print("\nSimulatorState type:")
    print(type(scenario))

    print("\nCurrent timestep:")
    print(scenario.timestep)

    print("\nTrajectory X shape:")
    print(scenario.sim_trajectory.x.shape)

    print("\nTrajectory Y shape:")
    print(scenario.sim_trajectory.y.shape)

    print("\nRoadgraph XYZ shape:")
    print(scenario.roadgraph_points.xyz.shape)

    print("\nObject metadata:")
    print(scenario.object_metadata)

    print("\n" + "=" * 60)
    print("WOMD FIRST SCENE LOAD: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()