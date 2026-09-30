"""Display-only PPO training progress reporting (terminal UX).

Pure formatting/arithmetic, no dependency on JAX/environment/training
state -- this module NEVER reads or influences PPO/GAE/RNG/sampler/
checkpoint/W&B/reward/environment behavior. It only turns
(update counts, wall-clock timestamps) into the one-line progress
string printed to stdout after each completed PPO update.

Plain stdlib ``print(..., flush=True)`` (no ANSI cursor control, no
tqdm) so the output stays correct and readable when piped through
``... | tee run.log`` -- every line is a complete, independent record.
"""

import dataclasses
import datetime
import time
from typing import Optional


def format_hms(seconds: float) -> str:
    """Formats a duration as HH:MM:SS (HH may exceed 99 for long runs,
    unlike ``time.strftime``, which wraps at 24 hours)."""

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_avg_per_update(seconds: float) -> str:
    """Formats an average per-update duration as MM:SS (updates are
    expected to take minutes, not hours, so this stays compact)."""

    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


@dataclasses.dataclass(frozen=True)
class ProgressSnapshot:
    """One immutable snapshot of training progress -- everything
    ``format_progress_line`` needs, computed once per completed update
    so the formatting function itself stays a pure, easily-testable
    string transform."""

    completed_this_invocation: int
    total_updates: int
    elapsed_s: float
    env_steps_per_sec: Optional[float]
    checkpoint_path: Optional[str]
    global_ppo_update_step: int
    resumed_from_step: int


def compute_snapshot(
    completed_this_invocation: int,
    total_updates: int,
    start_time: float,
    now: Optional[float] = None,
    env_steps_per_sec: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    global_ppo_update_step: int = 0,
    resumed_from_step: int = 0,
) -> ProgressSnapshot:
    now = now if now is not None else time.time()
    return ProgressSnapshot(
        completed_this_invocation=completed_this_invocation,
        total_updates=total_updates,
        elapsed_s=max(0.0, now - start_time),
        env_steps_per_sec=env_steps_per_sec,
        checkpoint_path=checkpoint_path,
        global_ppo_update_step=global_ppo_update_step,
        resumed_from_step=resumed_from_step,
    )


def format_progress_line(snapshot: ProgressSnapshot, now: Optional[float] = None) -> str:
    """Renders one progress line:

    [14/36] 38.9% | elapsed 04:42:13 | avg 20:09/update | ETA 07:23:27 | finish 2026-10-01 02:43:31

    Before the first completed update (completed_this_invocation == 0),
    average/ETA/finish are not yet computable -- rendered as
    "calculating..." rather than a divide-by-zero or a misleading 0.
    """

    now = now if now is not None else time.time()
    completed = snapshot.completed_this_invocation
    total = snapshot.total_updates
    pct = (100.0 * completed / total) if total > 0 else 0.0

    prefix = f"[{completed}/{total}] {pct:.1f}%"
    if snapshot.resumed_from_step > 0:
        prefix += f" (global update {snapshot.global_ppo_update_step})"

    elapsed_str = format_hms(snapshot.elapsed_s)

    if completed <= 0:
        avg_str = "calculating..."
        eta_str = "calculating..."
        finish_str = "calculating..."
    else:
        avg_per_update_s = snapshot.elapsed_s / completed
        remaining_updates = max(0, total - completed)
        eta_s = avg_per_update_s * remaining_updates
        avg_str = f"{format_avg_per_update(avg_per_update_s)}/update"
        eta_str = format_hms(eta_s)
        finish_dt = datetime.datetime.fromtimestamp(now + eta_s)
        finish_str = finish_dt.strftime("%Y-%m-%d %H:%M:%S")

    parts = [
        prefix,
        f"elapsed {elapsed_str}",
        f"avg {avg_str}",
        f"ETA {eta_str}",
        f"finish {finish_str}",
    ]
    if snapshot.env_steps_per_sec is not None:
        parts.append(f"env_steps/sec {snapshot.env_steps_per_sec:.2f}")
    line = " | ".join(parts)
    if snapshot.checkpoint_path is not None:
        line += f" | checkpoint {snapshot.checkpoint_path}"
    return line


class ProgressReporter:
    """Stateful wrapper around ``compute_snapshot``/``format_progress_line``
    for use as a ``trainer.run_training(on_update=...)`` callback.
    Tracks ONLY display state (start time, this-invocation's completed
    count) -- never touches training_state/rng_key/optimizer state, and
    never raises (a display bug must never abort a training run)."""

    def __init__(self, total_updates: int, resumed_from_step: int = 0, start_time: Optional[float] = None):
        self.total_updates = total_updates
        self.resumed_from_step = resumed_from_step
        self.start_time = start_time if start_time is not None else time.time()
        self._last_global_env_step: Optional[int] = None
        self._last_call_time: Optional[float] = None

    def report(
        self,
        update_index: int,
        global_env_step: int,
        ppo_update_step: int,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Called once per completed PPO update. Returns the formatted
        line (also printed here via plain ``print(..., flush=True)``).
        ``update_index`` is 0-based within THIS invocation, matching
        ``trainer.run_training``'s own callback convention."""

        now = time.time()
        env_steps_per_sec = None
        if self._last_global_env_step is not None and self._last_call_time is not None:
            step_delta = global_env_step - self._last_global_env_step
            time_delta = max(1e-9, now - self._last_call_time)
            env_steps_per_sec = step_delta / time_delta
        self._last_global_env_step = global_env_step
        self._last_call_time = now

        snapshot = compute_snapshot(
            completed_this_invocation=update_index + 1,
            total_updates=self.total_updates,
            start_time=self.start_time,
            now=now,
            env_steps_per_sec=env_steps_per_sec,
            checkpoint_path=checkpoint_path,
            global_ppo_update_step=ppo_update_step,
            resumed_from_step=self.resumed_from_step,
        )
        line = format_progress_line(snapshot, now=now)
        print(line, flush=True)
        return line
