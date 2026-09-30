"""Tests for src/training/progress_reporter.py.

Display-only terminal UX: pure formatting/arithmetic, no PPO/GAE/RNG/
sampler/checkpoint/W&B/reward/environment dependency. Verifies the
one-line progress format, the "calculating..." pre-first-update state,
resume disambiguation, and that a formatting failure inside the
combined on_update callback never raises (must not abort training).
"""

import time

import pytest

from src.training.progress_reporter import (
    ProgressReporter,
    compute_snapshot,
    format_avg_per_update,
    format_hms,
    format_progress_line,
)


def test_format_hms_basic():
    assert format_hms(0) == "00:00:00"
    assert format_hms(59) == "00:00:59"
    assert format_hms(60) == "00:01:00"
    assert format_hms(3661) == "01:01:01"


def test_format_hms_exceeds_24_hours_does_not_wrap():
    # Long training runs can exceed 24h -- must not wrap like
    # time.strftime("%H:%M:%S") would.
    seconds = 27 * 3600 + 23 * 60 + 27  # 27:23:27
    assert format_hms(seconds) == "27:23:27"


def test_format_avg_per_update():
    assert format_avg_per_update(0) == "00:00"
    assert format_avg_per_update(69) == "01:09"
    assert format_avg_per_update(20 * 60 + 9) == "20:09"


def test_progress_line_before_first_update_shows_calculating():
    snapshot = compute_snapshot(
        completed_this_invocation=0, total_updates=36, start_time=time.time(),
    )
    line = format_progress_line(snapshot)
    assert "[0/36]" in line
    assert "0.0%" in line
    assert "calculating..." in line
    # ETA/avg/finish must all be the placeholder, not a bogus 0-based value.
    assert line.count("calculating...") == 3


def test_progress_line_matches_documented_format_after_updates():
    start_time = 1000.0
    now = start_time + 14 * (20 * 60 + 9)  # 14 updates at exactly 20:09 avg
    snapshot = compute_snapshot(
        completed_this_invocation=14, total_updates=36, start_time=start_time, now=now,
    )
    line = format_progress_line(snapshot, now=now)
    assert line.startswith("[14/36] 38.9%")
    assert "elapsed 04:42:06" in line  # 14 * 1209s = 16926s = 04:42:06
    assert "avg 20:09/update" in line
    assert "ETA " in line
    assert "finish " in line


def test_progress_line_includes_env_steps_per_sec_when_provided():
    snapshot = compute_snapshot(
        completed_this_invocation=5, total_updates=36, start_time=time.time() - 100,
        env_steps_per_sec=42.5,
    )
    line = format_progress_line(snapshot)
    assert "env_steps/sec 42.50" in line


def test_progress_line_omits_env_steps_per_sec_when_none():
    snapshot = compute_snapshot(
        completed_this_invocation=5, total_updates=36, start_time=time.time() - 100,
        env_steps_per_sec=None,
    )
    line = format_progress_line(snapshot)
    assert "env_steps/sec" not in line


def test_progress_line_includes_checkpoint_path_when_provided():
    snapshot = compute_snapshot(
        completed_this_invocation=1, total_updates=36, start_time=time.time() - 10,
        checkpoint_path="outputs/checkpoints/full_seed0_step000001.pkl",
    )
    line = format_progress_line(snapshot)
    assert line.endswith("| checkpoint outputs/checkpoints/full_seed0_step000001.pkl")


def test_progress_line_omits_checkpoint_path_when_none():
    snapshot = compute_snapshot(
        completed_this_invocation=1, total_updates=36, start_time=time.time() - 10,
        checkpoint_path=None,
    )
    line = format_progress_line(snapshot)
    assert "checkpoint" not in line


def test_resume_disambiguates_global_step_from_invocation_progress():
    """A --resume run starting at global ppo_update_step=10, running 5
    more updates this invocation, must show progress out of the 5
    requested THIS invocation (never silently restarting the percentage
    at global step 10/5), while still surfacing the true global step."""

    snapshot = compute_snapshot(
        completed_this_invocation=2, total_updates=5, start_time=time.time() - 40,
        global_ppo_update_step=12, resumed_from_step=10,
    )
    line = format_progress_line(snapshot)
    assert line.startswith("[2/5] 40.0%")
    assert "(global update 12)" in line


def test_no_resume_disambiguation_shown_for_fresh_run():
    snapshot = compute_snapshot(
        completed_this_invocation=2, total_updates=5, start_time=time.time() - 40,
        global_ppo_update_step=2, resumed_from_step=0,
    )
    line = format_progress_line(snapshot)
    assert "global update" not in line


def test_progress_reporter_report_prints_and_returns_line(capsys):
    reporter = ProgressReporter(total_updates=3, start_time=time.time() - 5)
    line = reporter.report(update_index=0, global_env_step=100, ppo_update_step=1)
    captured = capsys.readouterr()
    assert line in captured.out
    assert line.startswith("[1/3]")


def test_progress_reporter_computes_env_steps_per_sec_between_calls(capsys):
    reporter = ProgressReporter(total_updates=3, start_time=time.time())
    reporter._last_global_env_step = 0
    reporter._last_call_time = time.time() - 2.0  # simulate 2s since the previous update
    line = reporter.report(update_index=0, global_env_step=200, ppo_update_step=1)
    assert "env_steps/sec" in line


def test_progress_reporter_first_call_has_no_env_steps_per_sec(capsys):
    reporter = ProgressReporter(total_updates=3, start_time=time.time())
    line = reporter.report(update_index=0, global_env_step=100, ppo_update_step=1)
    assert "env_steps/sec" not in line


def test_progress_reporter_never_raises_on_internal_error(capsys):
    """A malformed input (e.g. total_updates=0) must degrade gracefully
    (no crash), matching the "display bug must never abort training"
    requirement -- verified directly on the pure formatting path here;
    the combined on_update wrapper's own try/except is exercised in
    test_train_ppo_progress_wiring.py."""

    reporter = ProgressReporter(total_updates=0, start_time=time.time())
    line = reporter.report(update_index=0, global_env_step=10, ppo_update_step=1)
    assert "[1/0]" in line
