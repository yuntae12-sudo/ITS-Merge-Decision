# ITS-Merge-Decision

연구 제목:

- Korean: Waymax 기반 자율주행 합류 상황에서 Rule-Based FSM과 PPO 기반 행동 의사결정의 비교 연구
- English: A Comparative Study of Rule-Based FSM and PPO-Based Behavior Decision-Making for Autonomous Vehicle Merging in Waymax

## Research Overview

최종 연구 방향:

```
Waymax + WOMD 기반 Merge scenario
    ↓
Driving State
    ↓
Rule-Based FSM vs PPO
    ↓
Behavior Decision
    ↓
비교 평가
```

현재 Repository는 Phase 0 (환경 구성 및 기본 파이프라인 검증) 상태이다.

## Phase 0 Goal

- Waymax 설치 및 실행
- WOMD 실제 데이터 loading
- SimulatorState 확인
- Ego / surrounding state 확인
- Roadgraph 확인
- Visualization
- Metric
- Log playback rollout

## Environment

실제 확인된 환경:

- WSL2
- Ubuntu 24.04
- Conda environment: `its-merge`
- Python 3.10
- NVIDIA RTX 4060
- JAX (CUDA 지원, `jax.devices()` -> `[CudaDevice(id=0)]`)
- Waymax (GitHub main 기반 설치)
- WOMD Motion Dataset v1.3.1

## Repository Structure

```
ITS-Merge-Decision/
├── configs/                  # (Phase 1 이후 설정 파일 예정)
├── data/
│   └── womd/
│       └── validation/       # WOMD validation shard (git 미추적)
├── scripts/
│   ├── run_scene.py          # Phase 0 scene 검증 스크립트
│   └── run_rollout.py        # Phase 0 log playback rollout 스크립트
├── src/                      # 공용 소스 코드
├── tests/                    # 테스트 코드
├── outputs/                  # 실행 중 자동 생성되는 raw/debug 산출물 (git 미추적)
│   └── phase0/
│       └── scene_000/
│           ├── figures/      # topview.png
│           ├── rollout/      # rollout.gif
│           ├── snapshots/    # timestep_000.png ... timestep_090.png
│           └── metrics/      # rollout_metrics.csv
├── results/                  # Rule FSM / PPO / 비교 실험 결과 정리 (git 추적)
│   ├── rule_fsm/
│   ├── ppo/
│   └── comparison/
├── paper/                    # 논문용 최종 Figure / Table (git 추적)
│   ├── figures/
│   └── tables/
├── .gitignore
├── requirements.txt
└── README.md
```

`outputs/`는 스크립트 실행마다 다시 생성되는 산출물이며, `results/`와 `paper/`는
선별되어 정리한 실험 결과와 논문용 최종 자료를 위한 공간으로 서로 혼용하지 않는다.

## Dataset

WOMD (Waymo Open Motion Dataset) v1.3.1 사용.

Phase 0에서는 validation shard 하나만 사용한다:

```
data/womd/validation/validation_tfexample.tfrecord-00000-of-00150
```

다운로드 예시:

```bash
mkdir -p data/womd/validation

gcloud storage cp \
  gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150 \
  data/womd/validation/
```

다운로드에는 Waymo Open Dataset 사용 승인과 Google Cloud 인증(gcloud 로그인)이 필요하다.

## Run Phase 0 Scene Validation

```bash
conda activate its-merge
python scripts/run_scene.py
```

검증 내용:

- WOMD loading
- Ego state
- Surrounding agents
- Roadgraph
- Visualization
- Basic metrics

## Run Log Playback Rollout

```bash
python scripts/run_rollout.py
```

검증 내용:

- timestep 0 -> 90 (91 frames)
- Log playback
- GIF
- CSV
- Snapshots

## Outputs

```
outputs/phase0/scene_000/
    figures/
    rollout/
    snapshots/
    metrics/
```

## Phase 0 Validation Status

- [x] JAX / CUDA
- [x] WOMD scene loading
- [x] Ego state extraction
- [x] Surrounding agent extraction
- [x] Roadgraph inspection
- [x] Top-view visualization
- [x] Basic Waymax metrics
- [x] WOMD log playback rollout

현재 검증에 사용된 첫 번째 validation scene은 환경 검증용이며,
Merge scenario로 선정된 scene은 아니다. Merge scenario filtering / extraction은
다음 Phase에서 진행한다.

## Next Phase

**Phase 1: Merge Scenario Dataset Construction**

- WOMD / ScenarioMax 기반 Merge 후보 탐색
- Merge scenario 정의
- 자동 filtering
- Scenario metadata 구축
- 학습/평가 dataset 구성
