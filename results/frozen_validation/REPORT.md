# Frozen VALIDATION 388 Report — PPO Update 12

## 1. Frozen Checkpoint

- Selected update: 12
- Checkpoint path: `outputs/checkpoints/full_seed0_step000012.pkl`
- Checkpoint SHA-256: `f1962aa52d5e50a6588732e13623a5e292d7b9d9be8af0720f099911546e4eb1`
- TRAIN64 success 최고, timeout 최저, mean return 최고를 근거로 선택했다.
- Validation은 checkpoint 선택에 사용하지 않았다.

## 2. Validation Protocol

- Dataset: canonical Frozen VALIDATION (`merge_decision`), 388 maneuvers
- Deterministic argmax, current frozen `frenet_mpc`, Reward V1, 최대 100 physical steps
- Exception count: 0

## 3. Outcome Results

| Metric | Count | Rate |
|---|---:|---:|
| Success | 158 | 40.72% |
| Collision | 100 | 25.77% |
| Off-road | 10 | 2.58% |
| Timeout | 120 | 30.93% |

## 4. Downstream Results

| Metric | Count | Rate |
|---|---:|---:|
| Intervention | 6562 | 43.03% |
| Collision Blocked | 4963 | 35.38% |
| Planner Infeasible | 1599 | 7.65% |

Rate는 기존 trainer와 동일하게 각 episode의 final cumulative count / physical steps를 계산한 뒤 episode 평균한 값이다.

## 5. Reward Results

- Mean / std / median return: 0.1827 / 1.0035 / -0.0831
- Mean components — terminal -0.0309, safety -0.1391, progress 0.3626, decision -0.0098, decision-cost 0.0000

## 6. Action Distribution

| Action | Count | Ratio |
|---|---:|---:|
| KEEP | 7878 | 95.12% |
| FOLLOW | 130 | 1.57% |
| MERGE | 273 | 3.30% |
| STOP | 1 | 0.01% |

## 7. TRAIN64 vs VALIDATION

| Metric | TRAIN64 U12 | VALIDATION | Gap |
|---|---:|---:|---:|
| Success | 31.25% | 40.72% | +9.47%p |
| Collision | 43.75% | 25.77% | -17.98%p |
| Off-road | 3.12% | 2.58% | -0.55%p |
| Timeout | 21.88% | 30.93% | +9.05%p |
| Intervention | 58.89% | 43.03% | -15.86%p |
| Collision Blocked | 51.62% | 35.38% | -16.23%p |
| Planner Infeasible | 7.27% | 7.65% | +0.37%p |
| Mean Return | -0.0914 | 0.1827 | +0.2740 |

## 8. Representative Episodes

- Success: validation_tfexample.tfrecord-00000-of-00150#139__t29__245_146, validation_tfexample.tfrecord-00000-of-00150#151__t60__327_363, validation_tfexample.tfrecord-00000-of-00150#152__t9__76_84, validation_tfexample.tfrecord-00000-of-00150#181__t15__451_604, validation_tfexample.tfrecord-00000-of-00150#194__t31__287_280
- Collision: validation_tfexample.tfrecord-00000-of-00150#148__t55__418_720, validation_tfexample.tfrecord-00000-of-00150#154__t73__83_63, validation_tfexample.tfrecord-00000-of-00150#157__t69__787_736, validation_tfexample.tfrecord-00000-of-00150#171__t61__274_285, validation_tfexample.tfrecord-00000-of-00150#186__t50__88_82
- Off-road: validation_tfexample.tfrecord-00000-of-00150#163__t50__233_262, validation_tfexample.tfrecord-00000-of-00150#22__t15__126_164, validation_tfexample.tfrecord-00000-of-00150#283__t58__554_555, validation_tfexample.tfrecord-00002-of-00150#153__t75__584_621, validation_tfexample.tfrecord-00002-of-00150#210__t16__125_154
- Timeout: validation_tfexample.tfrecord-00000-of-00150#0__t33__206_203, validation_tfexample.tfrecord-00000-of-00150#0__t89__240_249, validation_tfexample.tfrecord-00000-of-00150#124__t51__308_309, validation_tfexample.tfrecord-00000-of-00150#127__t33__104_109, validation_tfexample.tfrecord-00000-of-00150#131__t42__339_156

## 9. Key Observations

- Success gap은 +9.47%p이다.
- Dominant failure mode는 timeout (120 episodes)이다.
- Intervention rate는 43.03%이다.
- Return/action distribution은 descriptive diagnostic이며 checkpoint 재선택에 사용하지 않는다.

## 10. Gate Decision

**READY_FOR_FSM_BASELINE_COMPARISON** — runtime anomaly와 큰 success 붕괴가 없고 safety threshold 미만이다.

## 11. Next Recommended Action

동일 Frozen VALIDATION에서 FSM baseline comparison으로 진행하거나, 위 gate가 요구한 generalization/safety/pipeline 분석을 먼저 수행한다. Checkpoint 재선택은 하지 않는다.
