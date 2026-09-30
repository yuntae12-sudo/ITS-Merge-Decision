"""Tests for scripts/train_ppo.py's display-only progress-reporting
wiring (make_periodic_checkpoint_callback / make_combined_update_callback).

Never imports/exercises real PPO training, environment, or checkpoint
I/O -- these are pure unit tests of the callback composition logic
added for terminal UX, verifying:
  - the periodic-checkpoint save/no-save decision and its returned
    path are unaffected by the progress-reporter wiring;
  - a progress-reporter failure never propagates out of the combined
    callback (a display bug must not abort a real training run).
"""

from unittest.mock import MagicMock

import scripts.train_ppo as train_ppo


def test_combined_callback_calls_periodic_checkpoint_and_reports_progress():
    saved_path = "outputs/checkpoints/full_seed0_step000005.pkl"
    periodic_fn = MagicMock(return_value=saved_path)
    reporter = MagicMock()

    on_update = train_ppo.make_combined_update_callback(periodic_fn, reporter)
    on_update(update_index=4, training_state="ts", rng_key="rk", global_env_step=500, ppo_update_step=5)

    periodic_fn.assert_called_once_with("ts", "rk", 500, 5)
    reporter.report.assert_called_once_with(
        update_index=4, global_env_step=500, ppo_update_step=5, checkpoint_path=saved_path,
    )


def test_combined_callback_reports_none_checkpoint_path_when_not_due():
    periodic_fn = MagicMock(return_value=None)
    reporter = MagicMock()

    on_update = train_ppo.make_combined_update_callback(periodic_fn, reporter)
    on_update(update_index=1, training_state="ts", rng_key="rk", global_env_step=100, ppo_update_step=2)

    reporter.report.assert_called_once_with(
        update_index=1, global_env_step=100, ppo_update_step=2, checkpoint_path=None,
    )


def test_combined_callback_works_with_no_periodic_checkpoint_configured():
    reporter = MagicMock()
    on_update = train_ppo.make_combined_update_callback(None, reporter)
    on_update(update_index=0, training_state="ts", rng_key="rk", global_env_step=10, ppo_update_step=1)
    reporter.report.assert_called_once_with(
        update_index=0, global_env_step=10, ppo_update_step=1, checkpoint_path=None,
    )


def test_combined_callback_never_raises_when_progress_reporter_fails(capsys):
    """A progress-display bug must never abort a real training run --
    the periodic checkpoint (the actually-important side effect) must
    still have been performed before the reporter failure is caught."""

    periodic_fn = MagicMock(return_value=None)
    reporter = MagicMock()
    reporter.report.side_effect = RuntimeError("display formatting bug")

    on_update = train_ppo.make_combined_update_callback(periodic_fn, reporter)
    on_update(update_index=0, training_state="ts", rng_key="rk", global_env_step=10, ppo_update_step=1)  # must not raise

    periodic_fn.assert_called_once()
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_make_periodic_checkpoint_callback_returns_none_when_interval_zero():
    assert train_ppo.make_periodic_checkpoint_callback(
        interval=0, checkpoint_dir="/tmp", filename_prefix="x", seed=0,
        reward_version="v1", config_snapshot={}, numpy_rng=None,
    ) is None


def test_make_periodic_checkpoint_callback_skips_off_interval_updates(tmp_path, monkeypatch):
    import numpy as np

    saved_paths = []
    monkeypatch.setattr(train_ppo, "save_checkpoint", lambda payload, path: saved_paths.append(path))

    fn = train_ppo.make_periodic_checkpoint_callback(
        interval=5, checkpoint_dir=tmp_path, filename_prefix="test", seed=0,
        reward_version="v1", config_snapshot={}, numpy_rng=np.random.RandomState(0),
    )

    class _FakeState:
        policy_state = MagicMock(params="pp", opt_state="po")
        value_state = MagicMock(params="vp", opt_state="vo")

    result_off_interval = fn(_FakeState(), "rng_key", 100, 3)
    assert result_off_interval is None
    assert saved_paths == []

    result_on_interval = fn(_FakeState(), "rng_key", 100, 5)
    assert result_on_interval is not None
    assert result_on_interval.endswith("test_step000005.pkl")
    assert saved_paths == [result_on_interval]
