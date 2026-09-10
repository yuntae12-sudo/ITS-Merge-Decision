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

---

# Phase 1: Merge Scenario Detection (Complete)

## Objective

WOMD 로그 데이터에서 "차선 합류(merge)" 상황을 자동으로 탐지하고, 확정된 merge
transition에 대해 Phase 3 PPO/FSM 공통 상태(8D interaction feature)의 원형이 되는
raw interaction feature(front/rear gap, relative speed, TTC, merge distance,
traffic density 등)를 추출하여, Phase 2 PPO 개발에 사용할 학습 pool과
초기 held-out 검증 pool을 구성하는 것이 Phase 1의 목표다. 이 문서는 Phase 1의
최종(Stage C) 상태를 기록한다.

## Shards Used

- Training (10 physical shards): `data/womd/training/training_tfexample.tfrecord-00000-of-01000`
  through `-00009-of-01000`.
- Validation (6 physical shards): `data/womd/validation/validation_tfexample.tfrecord-00000-of-00150`
  through `-00005-of-00150`.

Training과 validation은 물리적으로 분리된 WOMD shard 집합이며, 모든 산출물 row는
`source_split` 필드(`training` / `validation`)로 provenance가 명시되어 서로 섞이지
않는다 (Stage C에서 재검증됨).

## Merge Detector: ACCEPT / REVIEW / REJECT

각 stable lane transition(A차선 -> B차선)은 `src/scenarios/merge_detector.py`의
기하학적 판별 로직으로 3가지 중 하나로 분류된다:

- **ACCEPT**: 기하학적으로 확신 있는 진짜 merge(차선 종료형 합류, 수렴 지점 근접
  등). 이 transition만 interaction feature가 추출되어 학습 pool에 포함된다.
- **REJECT**: 확신 있는 non-merge (평행 차선 변경, 교차로 등).
- **REVIEW**: 현재 사용 가능한 기하 정보만으로는 "진짜 merge"와 "지도상 직렬
  segment 연속"을 확실히 구분할 수 없는 애매한 경우. Stage B 시각 감사에서 417개
  REVIEW 후보 중 merge-like로 재분류된 것이 0건 확인됨 (screening 결과, 미채택).

## Transition vs Maneuver vs Scene

세 단위를 절대 하나의 숫자로 합쳐서 보고하지 않는다 (Stage B/C 요구사항):

- **Transition**: 하나의 안정된 A->B 차선 전환 이벤트. `merge_manifest_*.csv`의
  한 row.
- **Maneuver**: 동일 scene 내에서 연속된 transition들을 하나의 실제 합류 동작으로
  묶은 단위 (`training_visual_merge_maneuvers.csv`). 하나의 maneuver가 여러
  transition으로 구성될 수 있음.
- **Scene**: 하나의 물리적 WOMD scenario record (`training_visual_merge_scenes.csv`).
  하나의 scene에 여러 maneuver/transition이 존재할 수 있음.

## Final Training Pool Counts

`outputs/phase1/training_10shard_pilot/`:

- ACCEPT transitions (detector): **176**
- Visual Merge 확인 transitions: **176** (시각 감사 결과 176/176 = 100% 확인)
- Unique Merge maneuvers: **168**
- Unique Merge scenes: **157** (confidence: **151 HIGH / 6 MEDIUM**)

모든 176개 ACCEPT row는 `source_of_detection=ACCEPT`이며 REVIEW/REJECT에서 유입된
row는 0건이다. 10개 training shard가 모두 대표되어 있다 (record_index/shard 조합
고유, cross-shard 혼동 없음).

## Validation Baseline

`outputs/phase1/feature_reference_fix/` (6-shard validation pool):

- Scenes scanned: **1755**
- Total stable transitions: **2006**
- ACCEPT: **56**
- REJECT: **253**
- REVIEW: **1697**

Validation pool은 초기 held-out set으로만 사용 가능하며, 논문 수준의 최종 통계적
결론을 내리기에는 아직 충분하지 않은 것으로 명시적으로 간주한다 (표본 확장은
Phase 2 이후 과제).

## Feature Reference Policy

Interaction feature는 `feature_reference_policy=merge_start_frame` 정책에 따라
**`merge_start_frame`** 시점에서 추출된다 (transition이 최종 확정되는
`transition_frame`이 아님). 세 가지 프레임 개념을 구분한다:

- `merge_start_frame`: merge 기하학적으로 시작하는 것으로 추정된 프레임.
  `feature_reference_frame`은 항상 이 값과 동일하다 (모든 176개 ACCEPT row에서
  검증됨).
- `feature_reference_frame`: 실제로 raw interaction feature가 계산된 프레임
  (== `merge_start_frame`).
- `transition_frame`: lane transition이 최종적으로(사후적으로, 이후 프레임들을
  참고하여) 확정된 프레임. 항상 `feature_reference_frame`보다 뒤에 온다
  (`feature_reference_frame < transition_frame`, 176/176 검증됨). Detector의
  merge 판정 자체는 사후 정보를 사용해도 되지만, 실제로 저장되는 feature는 그보다
  이른 시점(merge_start_frame)의 상태를 사용하여 Phase 3 온라인 정책이 사용할
  수 있는 시점에 가깝게 materialize한다.

### Negative front_gap_m audit (Stage C)

176개 ACCEPT row 중 21개가 음수 `front_gap_m` (범위 약 -0.29m ~ -4.64m, 모두
`front_ttc_s=0.0`)을 가진다. Stage C에서 이 중 4개 대표 사례(mild/strong 음수,
음수 ego longitudinal speed 사례 포함)를 `merge_start_frame`(=
`feature_reference_frame`) 시점의 실제 WOMD 장면 기하로 직접 재검증했다
(`outputs/phase1/stage_c_exit/negative_gap_audit/`).

결론: `front_gap_m`은 target lane 기준 center-to-center 거리(`front_s - ego_s`,
항상 양수 — front 선택 로직상 보장됨)에서 두 차량 length의 절반 합을 뺀
"bumper-to-bumper" 거리다. 감사한 모든 사례에서 center-to-center 거리는 작고
양수였으나(0.25m~4.39m), 두 차량 length 절반 합(약 4.7~5.0m)이 이를 초과하여
음수가 발생했다 — 즉 merge 시작 시점에 ego와 선택된 front 차량이 실제로 거의
맞닿아 있는 밀집 교통 상황이다. 렌더링된 이미지로 front 차량 선택이 항상
기하학적으로 타당함(target lane에 정렬, 올바른 heading, 교차/역방향 차량이
아님)을 확인했다. **결론: NEGATIVE_GAP_EXPECTED_GEOMETRY** — 계산 버그가
아니라 밀집 merge 상황에서 나타나는 정상적인 기하이며, feature 값을 clamp하거나
수정하지 않았다.

## Output Locations

- `outputs/phase1/training_10shard_pilot/`: 최종 training pool 산출물
  (`merge_manifest_training_scratch.csv`, `training_visual_merge_maneuvers.csv`,
  `training_visual_merge_scenes.csv`, 감사/시각화 파일).
- `outputs/phase1/feature_reference_fix/`: 6-shard validation baseline 산출물.
- `outputs/phase1/stage_c_exit/negative_gap_audit/`: Stage C 음수 gap 감사
  산출물 (`audit.csv`, `summary.json`, `figures/`).
- `outputs/`는 `.gitignore`로 전부 미추적 처리되어 있다 (Stage C에서 재확인).

### Scan summary output isolation

`scripts/extract_merge_scenes.py`는 후보 CSV(`--output`)와 별도로 scan summary
JSON(`merge_scan_summary.json`)을 쓴다. Stage B에서는 이 summary가 고정 경로
(`outputs/phase1/summaries/`)에 저장되어, 서로 다른 두 번의 scan 실행(training
10-shard, validation 6-shard)의 summary가 서로 덮어써지는 문제가 있었다. Stage C
에서 `--summary-dir` 옵션을 추가하고, 명시적으로 주어지지 않을 경우
`--output`이 기본값에서 변경되었으면 그 부모 디렉터리를 summary 위치로 자동
사용하도록 수정했다 (완전 기본 호출 시에는 기존 고정 경로를 그대로 유지해 하위
호환). 올바른 호출 예시:

```bash
python scripts/extract_merge_scenes.py \
    --dataset-config outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml \
    --output outputs/phase1/training_10shard_pilot/merge_candidates_training_10shard.csv \
    --summary-dir outputs/phase1/training_10shard_pilot

python scripts/extract_merge_scenes.py \
    --dataset-config outputs/phase1/feature_reference_fix/dataset_validation_6shard.yaml \
    --output outputs/phase1/feature_reference_fix/merge_candidates_6shard_postfix.csv \
    --summary-dir outputs/phase1/feature_reference_fix
```

회귀 테스트: `tests/scenarios/test_extract_merge_scenes_cli.py`.

## Canonical Label Safety

`data/manifests/merge_manual_labels.csv`는 수동 확인된 canonical label 파일이며,
어떤 자동화 스크립트도 이 파일을 덮어쓰거나 재포맷하지 않는다. Phase 1 전 과정에서
이 파일의 체크섬은 변경되지 않았다 (Stage C 최종 확인:
md5=`1f0071fb436396767d8fb79319016fb1`).

## Known Limitations / Deferred Phase 2 Items

다음 항목은 Phase 1에서 의도적으로 구현하지 않고 Phase 2로 이월한다:

1. PPO state 전처리에서 음수 `front_gap_m`을 어떻게 다룰지 결정 필요 (clamp할지,
   그대로 사용할지, 별도 피처로 인코딩할지).
2. 음수 ego longitudinal speed가 관측된 1건(`...00009-of-01000#204`)의 의미를
   먼저 조사한 뒤 clamp 여부를 결정할 것 — 맹목적으로 clamp하지 말 것.
3. 논문 수준 최종 FSM-vs-PPO 평가 전에 validation pool 확장을 고려할 것 (현재
   6-shard/1755-scene pool은 초기 개발용).
4. Training yield(예: ACCEPT/100)와 validation yield(예: ACCEPT/100)를 직접
   비교 가능한 통계적 비율로 해석하지 말 것 — training pool 구성 시 사용된
   분모는 "transition이 존재하는 scene 수" 근사치이며, validation의
   "스캔된 WOMD scene 수"와 동일한 정의가 아니다.

## Next Phase

**Phase 2: PPO Policy Development** (not started)
