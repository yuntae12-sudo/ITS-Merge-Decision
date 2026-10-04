# Fixed TRAIN64 Checkpoint Progression Report

## 1. Audit

- 기존 `restore_ppo_checkpoint`, `run_ppo_episode`, canonical TRAIN loader, Reward V1, environment counter를 재사용했다.
- 평가용 `scripts/evaluate_checkpoint_progression.py`, 전용 테스트, 결과 artifact만 추가했다.
- Checkpoint 위치: `outputs/checkpoints/full_seed0_step000001.pkl`부터 update 36까지.
- 기존 Fixed TRAIN64 `outputs/diagnostics/train_diag64_seed20260928.txt`를 재사용했다(SHA-256 `ae461f83a3a163e56f26da3c98faa8bec1c91549c3202cc9989da9aea1a87562`).

## 2. Evaluation Protocol

- Dataset: canonical TRAIN(`merge_decision` schema), 동일한 64개 maneuver.
- Checkpoint: 1(가장 가까운 initial), 6, 12, 18, 24, 30, 36. update-0 checkpoint는 존재하지 않는다.
- Deterministic argmax policy, 고정 순서, `frenet_mpc`, Reward V1, 최대 100 physical steps를 사용했다.
- Frozen VALIDATION은 실행하지 않았다.

## 3. Checkpoint Results

| Checkpoint | Success | Collision | Off-road | Timeout | Intervention | Collision Blocked | Planner Infeasible | Mean Return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.88% | 45.31% | 4.69% | 28.12% | 57.77% | 51.99% | 5.78% | -0.2611 |
| 6 | 17.19% | 46.88% | 3.12% | 32.81% | 60.09% | 50.71% | 9.38% | -0.3288 |
| 12 | 31.25% | 43.75% | 3.12% | 21.88% | 58.89% | 51.62% | 7.27% | -0.0914 |
| 18 | 26.56% | 43.75% | 1.56% | 28.12% | 60.85% | 49.77% | 11.09% | -0.1679 |
| 24 | 26.56% | 42.19% | 1.56% | 29.69% | 61.63% | 49.13% | 12.50% | -0.1675 |
| 30 | 26.56% | 43.75% | 3.12% | 26.56% | 59.42% | 50.81% | 8.61% | -0.1752 |
| 36 | 25.00% | 43.75% | 4.69% | 26.56% | 58.11% | 50.65% | 7.46% | -0.1995 |

## 4. Action Distribution

| Checkpoint | KEEP | FOLLOW | MERGE | STOP |
|---|---:|---:|---:|---:|
| 1 | 98.20% | 1.17% | 0.62% | 0.00% |
| 6 | 97.70% | 0.62% | 1.68% | 0.00% |
| 12 | 95.43% | 0.24% | 4.32% | 0.00% |
| 18 | 82.42% | 0.00% | 7.23% | 10.35% |
| 24 | 79.20% | 1.37% | 6.74% | 12.69% |
| 30 | 92.76% | 1.27% | 5.25% | 0.72% |
| 36 | 96.96% | 0.00% | 3.04% | 0.00% |

## 5. Key Observations

- Success는 21.88%에서 25.00%로 증가했지만 단조 증가하지 않았다.
- Collision은 45.31%에서 43.75%로 소폭 감소했다.
- Intervention은 57.77%에서 58.11%로 감소하지 않았다.
- Mean return은 -0.2611에서 -0.1995로 개선됐지만 update 12 이후 악화됐다.
- TRAIN64 최고 success는 update 12의 31.25%이며 final update 36이 best가 아니다.
- Action distribution은 trend 진단용으로만 사용했고 성능 판정 기준에는 포함하지 않았다.
- 기존 rollout schema에는 merge completion time이 없어 만들지 않았으며, 사용 가능한 merge commit step만 보고했다.

## 6. Scenario Transition Analysis

- Nearest-initial update 1 → final update 36: success로 개선 3개, success에서 악화 1개, failure mode 변경 2개.
- Maneuver별 paired outcome은 `transitions/early_vs_final.csv`에 저장했다.
- 지배적인 final failure mode는 collision 28/64와 timeout 17/64이다.

## 7. Gate Decision

**CHECKPOINT_SELECTION_REQUIRED**

Final checkpoint가 TRAIN64 success 기준 best checkpoint가 아니며 progression도 noisy하다.

## 8. Next Recommended Action

TRAIN64만을 근거로 update 12를 포함한 checkpoint 선택을 먼저 확정해야 한다. 그 전에는 Frozen VALIDATION을 실행하지 않는 것을 권고하며, 이번 작업에서는 실제 VALIDATION을 실행하지 않았다.
