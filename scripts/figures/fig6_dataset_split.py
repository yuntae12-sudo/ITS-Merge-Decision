"""Figure 6 -- dataset / experiment setup summary.

EXPERIMENTAL figure, computed programmatically from the real committed
split manifest (``data/manifests/phase2_dataset_split.csv``, one row
per maneuver_id: maneuver_id, scene_key, split). Verifies
total=168, TRAIN=110, VALIDATION=58, and checks (directly from the
manifest's own ``scene_key`` column) that no ``scene_key`` appears in
both splits -- if any of these checks fail, the script stops and
reports the discrepancy rather than forcing a chart.

Simple horizontal bar chart: TRAIN vs VALIDATION maneuver counts.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.figures import _style

MANIFEST_PATH = "data/manifests/phase2_dataset_split.csv"
OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"

EXPECTED_TOTAL = 168
EXPECTED_TRAIN = 110
EXPECTED_VALIDATION = 58


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    counts = {}
    scenes_by_split = {}
    for r in rows:
        split = r["split"]
        counts[split] = counts.get(split, 0) + 1
        scenes_by_split.setdefault(split, set()).add(r["scene_key"])

    n_train = counts.get("train", 0)
    n_val = counts.get("validation", 0)
    overlap = scenes_by_split.get("train", set()) & scenes_by_split.get("validation", set())

    print(f"[fig6] Recomputed from {MANIFEST_PATH}: total={total} train={n_train} validation={n_val} "
          f"scene_key overlap between splits={len(overlap)}")

    problems = []
    if total != EXPECTED_TOTAL:
        problems.append(f"total={total} != expected {EXPECTED_TOTAL}")
    if n_train != EXPECTED_TRAIN:
        problems.append(f"train={n_train} != expected {EXPECTED_TRAIN}")
    if n_val != EXPECTED_VALIDATION:
        problems.append(f"validation={n_val} != expected {EXPECTED_VALIDATION}")
    if overlap:
        problems.append(f"{len(overlap)} scene_key(s) appear in BOTH splits: {sorted(overlap)[:5]}...")

    if problems:
        print("[fig6] STOP: dataset-split consistency check FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("[fig6] Figure 6 NOT generated.")
        return False

    print("[fig6] Consistency check PASSED (counts match, zero scene-level overlap).")

    data_path = os.path.join(OUT_DATA, "fig6_dataset_split_summary.csv")
    with open(data_path, "w") as f:
        f.write("split,count\n")
        f.write(f"train,{n_train}\n")
        f.write(f"validation,{n_val}\n")
    print(f"[fig6] wrote {data_path}")

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, ax = plt.subplots(figsize=(5.5, 2.6) if not ppt else (9, 4.5))
        labels = ["VALIDATION", "TRAIN"]
        values = [n_val, n_train]
        colors = [_style.COLOR_TARGET_LANE, _style.COLOR_SOURCE_LANE]
        bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(6, 0), va="center",
                        fontsize=(14 if ppt else 9))
        ax.set_xlabel(f"Maneuvers (of {total} total)")
        if not ppt:
            ax.set_title("Phase 2/3 canonical dataset split (real scene-level split, 0 overlap)", fontsize=9)
        ax.set_xlim(0, max(values) * 1.25)
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, "fig6_dataset_split_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, "fig6_dataset_split_paper"))
        plt.close(fig)

    print("[fig6] done.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
