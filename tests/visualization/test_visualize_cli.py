"""Tests for scripts/visualization/visualize_ppo_run.py's CLI argument
wiring (PPO Visualization Default-5 Extension, task spec Section 9):
--scan-only/--render-all mutual exclusivity, and that an explicit
--maneuver-ids scope is never truncated by the default per-outcome
render cap. These test argparse/selection wiring directly -- no
checkpoint/environment/GPU resources needed.
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "visualization", "visualize_ppo_run.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("visualize_ppo_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, REPO_ROOT)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_module()


# ======================================================================
# E. --scan-only / --render-all mutual exclusivity
# ======================================================================


def test_scan_only_and_render_all_are_mutually_exclusive(cli_module):
    with pytest.raises(SystemExit):
        cli_module.parse_args_from(["--checkpoint", "x.pkl", "--scan-only", "--render-all"])


def test_scan_only_alone_parses_fine(cli_module):
    args = cli_module.parse_args_from(["--checkpoint", "x.pkl", "--scan-only"])
    assert args.scan_only is True
    assert args.render_all is False


def test_render_all_alone_parses_fine(cli_module):
    args = cli_module.parse_args_from(["--checkpoint", "x.pkl", "--render-all"])
    assert args.render_all is True
    assert args.scan_only is False


def test_neither_flag_defaults_to_representative_mode(cli_module):
    args = cli_module.parse_args_from(["--checkpoint", "x.pkl"])
    assert args.scan_only is False
    assert args.render_all is False
    assert args.max_per_outcome == cli_module.DEFAULT_MAX_PER_OUTCOME


# ======================================================================
# F. Explicit maneuver selection is never dropped by the default cap
# ======================================================================


def test_explicit_scope_render_policy_has_no_cap(cli_module):
    """scope=explicit must resolve to an uncapped (mode='all') render
    policy so a user's named --maneuver-ids are never dropped by the
    default per-outcome cap, even if e.g. 6 of the user's explicit
    maneuvers happen to share the same outcome (task spec Section 6).
    Exercises the REAL resolve_render_policy() main() itself calls."""

    max_per_outcome, render_policy = cli_module.resolve_render_policy(
        render_all=False, scan_only=False, scope="explicit", max_per_outcome=5,
    )
    assert max_per_outcome is None
    assert render_policy["mode"] == "all"


def test_checkpoint_scope_render_policy_uses_default_cap(cli_module):
    max_per_outcome, render_policy = cli_module.resolve_render_policy(
        render_all=False, scan_only=False, scope="checkpoint", max_per_outcome=5,
    )
    assert max_per_outcome == 5
    assert render_policy == {"mode": "representative_per_outcome", "max_per_outcome": 5}


def test_render_all_render_policy(cli_module):
    max_per_outcome, render_policy = cli_module.resolve_render_policy(
        render_all=True, scan_only=False, scope="checkpoint", max_per_outcome=5,
    )
    assert max_per_outcome is None
    assert render_policy == {"mode": "all"}


def test_scan_only_render_policy_keeps_default_cap_but_wont_render(cli_module):
    """--scan-only still resolves a numeric max_per_outcome (used only
    if rendering were to happen) but its render_policy mode records
    that nothing gets rendered."""

    max_per_outcome, render_policy = cli_module.resolve_render_policy(
        render_all=False, scan_only=True, scope="checkpoint", max_per_outcome=5,
    )
    assert max_per_outcome == 5
    assert render_policy == {"mode": "scan_only"}
