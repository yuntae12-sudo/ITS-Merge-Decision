"""Figure 4 -- 168-maneuver Phase 3 robustness result.

EXPERIMENTAL figure, built PROGRAMMATICALLY from the committed
Stage 3-G audit output (``outputs/phase3/robustness/audit_168_maneuvers.json``)
-- never hardcoded. Verifies the well-known headline numbers
(total=168, success=106, collision=37, truncation=20, offroad=5) as an
internal consistency CHECK, computed fresh from the file every run; if
the recomputed numbers ever disagree with this check, the script
STOPS instead of silently plotting a mismatched chart.

Caption (must accompany this figure wherever used): "Phase 3
engineering robustness/regression result -- MERGE-every-step under
frenet_mpc across all 168 canonical maneuvers, a survivability
stimulus (crash/NaN/OOB-free execution), NOT the final FSM-vs-PPO
research comparison."
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.figures import _style

AUDIT_JSON = "outputs/phase3/robustness/audit_168_maneuvers.json"
OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"

EXPECTED = {"total": 168, "success": 106, "collision": 37, "truncation": 20, "offroad": 5}


def load_and_summarize():
    with open(AUDIT_JSON) as f:
        rows = json.load(f)

    total = len(rows)
    n_success = sum(1 for r in rows if r["termination_reason"] == "success")
    n_collision = sum(1 for r in rows if r["termination_reason"] == "failure_collision")
    n_truncation = sum(1 for r in rows if r["termination_reason"] == "truncation_horizon")
    n_offroad = sum(1 for r in rows if r["termination_reason"] == "failure_offroad")

    # Also cross-check against the rows' own boolean flags (success/collision/offroad),
    # which the audit script computed independently from termination_reason.
    n_success_flag = sum(1 for r in rows if r["success"])
    n_collision_flag = sum(1 for r in rows if r["collision"])
    n_offroad_flag = sum(1 for r in rows if r["offroad"])

    assert n_success == n_success_flag, f"success mismatch: {n_success} vs flag {n_success_flag}"
    assert n_collision == n_collision_flag, f"collision mismatch: {n_collision} vs flag {n_collision_flag}"
    assert n_offroad == n_offroad_flag, f"offroad mismatch: {n_offroad} vs flag {n_offroad_flag}"

    other = total - n_success - n_collision - n_truncation - n_offroad

    summary = {
        "total": total, "success": n_success, "collision": n_collision,
        "truncation": n_truncation, "offroad": n_offroad, "other": other,
    }
    return summary, rows


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    summary, rows = load_and_summarize()
    print(f"[fig4] recomputed summary from {AUDIT_JSON}: {summary}")

    mismatches = {k: (summary[k], v) for k, v in EXPECTED.items() if summary.get(k) != v}
    if mismatches:
        print(f"[fig4] STOP: recomputed numbers disagree with the documented Stage 3-G headline "
              f"figures. Mismatches (recomputed, expected): {mismatches}")
        print("[fig4] Refusing to force a fabricated/incorrect chart. Figure 4 NOT generated.")
        return False

    print(f"[fig4] Consistency check PASSED: total={summary['total']} success={summary['success']} "
          f"collision={summary['collision']} truncation={summary['truncation']} offroad={summary['offroad']} "
          f"(other={summary['other']})")

    # Save exact plotted numbers.
    data_path = os.path.join(OUT_DATA, "fig4_robustness_summary.csv")
    with open(data_path, "w") as f:
        f.write("category,count\n")
        for k in ("success", "collision", "truncation", "offroad"):
            f.write(f"{k},{summary[k]}\n")
        if summary["other"] != 0:
            f.write(f"other,{summary['other']}\n")
    print(f"[fig4] wrote {data_path}")

    categories = ["success", "collision", "truncation", "offroad"]
    colors = [_style.COLOR_SUCCESS, _style.COLOR_COLLISION, _style.COLOR_TRUNCATION, _style.COLOR_OFFROAD]
    counts = [summary[c] for c in categories]

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, ax = plt.subplots(figsize=(5.5, 4.2) if not ppt else (9, 6.5))
        bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.6)
        for bar, count in zip(bars, counts):
            ax.annotate(f"{count}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=(14 if ppt else 9))
        ax.set_ylabel(f"Maneuvers (of {summary['total']} total)")
        if not ppt:
            ax.set_title("Phase 3 robustness result (168 maneuvers, frenet_mpc, MERGE-every-step)\n"
                          "Engineering survivability stimulus -- NOT the final FSM-vs-PPO comparison", fontsize=8)
        ax.set_ylim(0, max(counts) * 1.2)
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, "fig4_robustness_audit_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, "fig4_robustness_audit_paper"))
        plt.close(fig)

    print("[fig4] done.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
