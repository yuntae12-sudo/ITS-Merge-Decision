"""Figure 5 -- downstream intervention analysis.

EXPERIMENTAL figure, computed programmatically from the same committed
Stage 3-G audit data
(``outputs/phase3/robustness/audit_168_maneuvers.json``). For each of
the 168 maneuvers, computes

    intervention_rate = (steps with downstream_status != OK)
                         / (steps_executed)

directly from that maneuver's own ``downstream_status_counts`` dict
(``OK`` count subtracted from ``steps_executed``), matching Stage
3-H Section 3's own definition
(``downstream_failure_count / steps_elapsed``) exactly. This is a
recomputation from the SAME per-maneuver per-status counts already in
the committed audit file -- not a re-run of any rollout.

Recomputes the Stage 3-H headline finding ("23/106 success episodes
had intervention_rate > 0.30") from this source data. If the
recomputed number disagrees with 23/106, this script prints the
discrepancy explicitly (it does NOT force the reported figure to
match) and proceeds to plot the actual recomputed numbers.

Produces:
  - a histogram of intervention_rate across all 168 maneuvers, and
  - a strip/scatter plot (maneuver index vs. intervention_rate),
    success vs. non-success distinguished by color.

Caption: "Success alone may conceal substantial downstream
intervention -- a MERGE decision can culminate in success=True while
most/all of its steps were executed via the bounded-brake fallback,
not the planner's own intended trajectory (Stage 3-H Section 3)."
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style

AUDIT_JSON = "outputs/phase3/robustness/audit_168_maneuvers.json"
OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"

DOCUMENTED_SUCCESS_HIGH_INTERVENTION_COUNT = 23
DOCUMENTED_SUCCESS_TOTAL = 106
DOCUMENTED_THRESHOLD = 0.30


def compute_intervention_rates():
    with open(AUDIT_JSON) as f:
        rows = json.load(f)

    records = []
    for r in rows:
        steps = r["steps_executed"]
        counts = r["downstream_status_counts"]
        ok_count = counts.get("OK", 0)
        non_ok = steps - ok_count
        rate = (non_ok / steps) if steps > 0 else 0.0
        records.append({
            "maneuver_id": r["maneuver_id"],
            "split": r["split"],
            "steps_executed": steps,
            "intervention_rate": rate,
            "success": bool(r["success"]),
        })
    return records


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    records = compute_intervention_rates()
    n_total = len(records)
    successes = [r for r in records if r["success"]]
    non_successes = [r for r in records if not r["success"]]
    n_success = len(successes)

    n_success_high_intervention = sum(1 for r in successes if r["intervention_rate"] > DOCUMENTED_THRESHOLD)

    mean_rate_success = float(np.mean([r["intervention_rate"] for r in successes])) if successes else float("nan")
    mean_rate_nonsuccess = float(np.mean([r["intervention_rate"] for r in non_successes])) if non_successes else float("nan")

    print(f"[fig5] Recomputed: {n_success_high_intervention}/{n_success} success maneuvers have "
          f"intervention_rate > {DOCUMENTED_THRESHOLD}")
    print(f"[fig5] mean intervention_rate | success=True:  {mean_rate_success:.3f} (n={n_success})")
    print(f"[fig5] mean intervention_rate | success=False: {mean_rate_nonsuccess:.3f} (n={len(non_successes)})")

    if (n_success_high_intervention, n_success) != (DOCUMENTED_SUCCESS_HIGH_INTERVENTION_COUNT, DOCUMENTED_SUCCESS_TOTAL):
        print(f"[fig5] DISCREPANCY: documented Stage 3-H finding was "
              f"{DOCUMENTED_SUCCESS_HIGH_INTERVENTION_COUNT}/{DOCUMENTED_SUCCESS_TOTAL}; "
              f"recomputed from source data is {n_success_high_intervention}/{n_success}. "
              "Reporting the RECOMPUTED number as ground truth (not forcing a match).")
    else:
        print("[fig5] Recomputed number matches the documented Stage 3-H finding exactly.")

    # Save exact plotted data.
    data_path = os.path.join(OUT_DATA, "fig5_intervention_rates.csv")
    with open(data_path, "w") as f:
        f.write("maneuver_id,split,steps_executed,intervention_rate,success\n")
        for r in records:
            f.write(f"{r['maneuver_id']},{r['split']},{r['steps_executed']},{r['intervention_rate']},{r['success']}\n")
    print(f"[fig5] wrote {data_path}")

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, (ax_hist, ax_strip) = plt.subplots(1, 2, figsize=(10, 4) if not ppt else (16, 6.5))

        bins = np.linspace(0.0, 1.0, 21)
        ax_hist.hist([r["intervention_rate"] for r in successes], bins=bins, alpha=0.7,
                     color=_style.COLOR_SUCCESS_EPISODE, label=f"success (n={len(successes)})")
        ax_hist.hist([r["intervention_rate"] for r in non_successes], bins=bins, alpha=0.7,
                     color=_style.COLOR_NONSUCCESS_EPISODE, label=f"non-success (n={len(non_successes)})")
        ax_hist.axvline(DOCUMENTED_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
                         label=f"threshold={DOCUMENTED_THRESHOLD}")
        ax_hist.set_xlabel("intervention_rate")
        ax_hist.set_ylabel("Maneuver count")
        ax_hist.legend(fontsize=(11 if ppt else 7))

        for idx, r in enumerate(records):
            color = _style.COLOR_SUCCESS_EPISODE if r["success"] else _style.COLOR_NONSUCCESS_EPISODE
            ax_strip.scatter(idx, r["intervention_rate"], color=color, s=(20 if ppt else 10), alpha=0.75)
        ax_strip.axhline(DOCUMENTED_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
        ax_strip.set_xlabel("Maneuver index (audit order)")
        ax_strip.set_ylabel("intervention_rate")

        if not ppt:
            fig.suptitle(f"Downstream intervention analysis (168 maneuvers): "
                          f"{n_success_high_intervention}/{n_success} success episodes have intervention_rate > {DOCUMENTED_THRESHOLD}\n"
                          "Success alone may conceal substantial downstream intervention", fontsize=9)
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, "fig5_intervention_analysis_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, "fig5_intervention_analysis_paper"))
        plt.close(fig)

    print("[fig5] done.")


if __name__ == "__main__":
    main()
