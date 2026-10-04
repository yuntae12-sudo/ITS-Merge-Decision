#!/usr/bin/env python3
"""Runs all 4 capstone-midterm figure scripts in sequence.

Each script is independent and real-data-only (see each script's own
module docstring + README.md in this directory for exact sources).
Figures 1/2/4 run a real MergeEnvironment rollout and take a few
minutes each; Figure 3 only re-plots an already-generated CSV and is
fast.
"""

import runpy
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURE_SCRIPTS = [
    "fig01_dataset_scenario.py",
    "fig02_rule_based_baseline.py",
    "fig03_ppo_training.py",
    "fig04_evaluation_framework.py",
]


def main():
    for script_name in FIGURE_SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        print(f"=== Running {script_name} ===")
        runpy.run_path(script_path, run_name="__main__")
        print()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))
    main()
