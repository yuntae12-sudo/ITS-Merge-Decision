#!/usr/bin/env python3
"""Fixed TRAIN64 same-maneuver outcome transition analysis (read-only).

Parses the per-maneuver "maneuver_id: outcome=X steps=N" lines already
produced by scripts/diagnostics/analyze_planner_infeasible.py's console
log for each evaluated checkpoint step, and computes:
  - per-step outcome distribution
  - same-maneuver Step A -> Step B outcome transition matrices

Never re-runs the environment -- pure text parsing + counting.
"""

import csv
import re
import sys
from collections import Counter, defaultdict

OUTCOME_LINE_RE = re.compile(r"^\s*(\S+): outcome=(\w+) steps=(\d+)\s*$")


def parse_step_section(log_path, step_marker, next_markers):
    """Extracts {maneuver_id: outcome} for one "=== STEP N ===" section."""

    results = {}
    with open(log_path) as f:
        in_section = False
        for line in f:
            if step_marker in line:
                in_section = True
                continue
            if in_section and any(m in line for m in next_markers):
                break
            if in_section:
                m = OUTCOME_LINE_RE.match(line)
                if m:
                    results[m.group(1)] = m.group(2)
    return results


def transition_matrix(outcomes_a, outcomes_b):
    matrix = Counter()
    common_ids = set(outcomes_a.keys()) & set(outcomes_b.keys())
    for mid in common_ids:
        matrix[(outcomes_a[mid], outcomes_b[mid])] += 1
    return matrix, common_ids


def main():
    log_path = "/tmp/retrain_progression.log"
    steps = [1, 9, 18, 27, 36]
    markers = {s: f"=== STEP {s} ===" for s in steps}

    outcomes_by_step = {}
    for i, step in enumerate(steps):
        next_markers = [markers[s] for s in steps[i + 1:]]
        outcomes_by_step[step] = parse_step_section(log_path, markers[step], next_markers)
        print(f"Step {step}: {len(outcomes_by_step[step])} maneuvers parsed, "
              f"outcomes={dict(Counter(outcomes_by_step[step].values()))}")

    output_dir = "outputs/diagnostics/retrain_freeze3213b1b"

    # Outcome progression table.
    with open(f"{output_dir}/fixed_train64_progression.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "success", "collision", "offroad", "timeout", "total"])
        for step in steps:
            c = Counter(outcomes_by_step[step].values())
            writer.writerow([step, c.get("success", 0), c.get("collision", 0),
                              c.get("offroad", 0), c.get("timeout", 0), len(outcomes_by_step[step])])

    for (step_a, step_b) in [(1, 36), (27, 36)]:
        matrix, common_ids = transition_matrix(outcomes_by_step[step_a], outcomes_by_step[step_b])
        print(f"\n=== Transition Step {step_a} -> Step {step_b} ({len(common_ids)} common maneuvers) ===")
        for (from_o, to_o), count in sorted(matrix.items(), key=lambda kv: -kv[1]):
            print(f"  {from_o:<10} -> {to_o:<10}: {count}")
        changed = sum(c for (a, b), c in matrix.items() if a != b)
        unchanged = sum(c for (a, b), c in matrix.items() if a == b)
        print(f"  changed: {changed}/{len(common_ids)}, unchanged: {unchanged}/{len(common_ids)}")

        with open(f"{output_dir}/outcome_transitions_step{step_a}_to_{step_b}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["from_outcome", "to_outcome", "count"])
            for (from_o, to_o), count in sorted(matrix.items(), key=lambda kv: -kv[1]):
                writer.writerow([from_o, to_o, count])

    print(f"\nWrote {output_dir}/fixed_train64_progression.csv")
    print(f"Wrote {output_dir}/outcome_transitions_step1_to_36.csv")
    print(f"Wrote {output_dir}/outcome_transitions_step27_to_36.csv")


if __name__ == "__main__":
    main()
