"""Regression test for scripts/extract_merge_scenes.py's scan-summary
output-path isolation (Stage C / Phase 1 exit fix).

Stage B found that two separate scan invocations (a training-shard
scan and a validation-shard scan) both wrote their
``merge_scan_summary.json`` to the same fixed
``outputs/phase1/summaries`` directory, so the later run silently
overwrote the earlier run's summary even though their candidates CSVs
(``--output``) were written to distinct paths. This test exercises
``resolve_summary_dir`` directly (no real WOMD data / no full scan
needed) to lock in the fix: distinct ``--output`` paths now yield
distinct default summary directories, an explicit ``--summary-dir``
always wins, and the fully-default invocation keeps its original
backward-compatible location.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from extract_merge_scenes import (
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY_DIR,
    resolve_summary_dir,
)


def test_default_output_and_no_summary_dir_uses_default_summary_dir():
    # Fully-default invocation (no --output, no --summary-dir)
    # preserves the original backward-compatible fixed location.
    output_path = Path(DEFAULT_OUTPUT)
    resolved = resolve_summary_dir(None, DEFAULT_OUTPUT, output_path)
    assert resolved == Path(DEFAULT_SUMMARY_DIR)


def test_overridden_output_without_summary_dir_colocates_with_output():
    # This is the Stage B collision case, fixed: an overridden
    # --output with no explicit --summary-dir now derives the summary
    # directory from --output's parent instead of the shared default.
    output_arg = "outputs/phase1/training_10shard_pilot/merge_candidates_training_10shard.csv"
    output_path = Path(output_arg)
    resolved = resolve_summary_dir(None, output_arg, output_path)
    assert resolved == output_path.parent


def test_two_distinct_output_paths_never_collide_on_summary_dir():
    training_output = Path(
        "outputs/phase1/training_10shard_pilot/merge_candidates_training_10shard.csv"
    )
    validation_output = Path(
        "outputs/phase1/feature_reference_fix/merge_candidates_6shard_postfix.csv"
    )

    training_summary_dir = resolve_summary_dir(
        None, str(training_output), training_output
    )
    validation_summary_dir = resolve_summary_dir(
        None, str(validation_output), validation_output
    )

    assert training_summary_dir != validation_summary_dir
    assert training_summary_dir == training_output.parent
    assert validation_summary_dir == validation_output.parent


def test_explicit_summary_dir_always_wins():
    # Even with a default --output, an explicit --summary-dir takes
    # precedence over both the derived-from-output and the fixed
    # default behavior.
    output_path = Path(DEFAULT_OUTPUT)
    resolved = resolve_summary_dir(
        "outputs/phase1/custom_run", DEFAULT_OUTPUT, output_path
    )
    assert resolved == Path("outputs/phase1/custom_run")

    overridden_output = Path("outputs/phase1/some_run/candidates.csv")
    resolved_overridden = resolve_summary_dir(
        "outputs/phase1/custom_run", str(overridden_output), overridden_output
    )
    assert resolved_overridden == Path("outputs/phase1/custom_run")
