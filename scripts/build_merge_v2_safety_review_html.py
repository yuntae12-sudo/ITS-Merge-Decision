#!/usr/bin/env python3
"""Builds a local (no server) HTML index for the 20 SAFETY_CRITICAL
targeted-audit candidates, so a human can visually cross-check the
geometry-based diagnostic_verdict against the actual WOMD rollout GIF.

diagnostic_verdict shown here is NOT a human_label -- it is this audit's
own geometry-based diagnostic (see audit_merge_v2_safety_geometry.py).
"""

import csv
from pathlib import Path

AUDIT_CSV = "outputs/merge_v2_decision_audit_v2/safety_critical_targeted_audit.csv"
OUT_PATH = Path("outputs/merge_v2_decision_audit_v2/safety_review/index.html")

VERDICT_COLOR = {
    "PHYSICAL_CONFLICT_LIKELY": "#ffcccc",
    "PROJECTION_ARTIFACT_LIKELY": "#ccffcc",
    "AMBIGUOUS": "#fff3cc",
}


def main():
    with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    from collections import Counter
    verdict_counts = Counter(r["diagnostic_verdict"] for r in rows)

    lines = [
        "<html><head><meta charset='utf-8'>",
        "<title>MERGE v2 SAFETY_CRITICAL Targeted Geometry Audit</title></head><body>",
        "<h1>SAFETY_CRITICAL Targeted Geometry Audit (20 candidates)</h1>",
        "<p>Resolves whether the sustained target-lane negative-gap signal "
        "(front/rear_gap_m &lt; -0.5m for &gt;=5 consecutive frames) reflects a "
        "genuine near-conflict or a target-polyline projection artifact. "
        "Geometry recomputed from raw tf_example poses using Waymax's own "
        "SAT/OBB overlap primitive (waymax.utils.geometry.has_overlap) -- "
        "the same primitive OverlapMetric uses for live collision "
        "termination -- not a reimplementation.</p>",
        f"<p><b>Verdict distribution:</b> "
        + ", ".join(f"{k}={v}" for k, v in verdict_counts.items())
        + f" (out of {len(rows)})</p>",
        "<p><b>diagnostic_verdict is NOT a human_label</b> -- it does not change "
        "automatic_decision, calibration_status, or any threshold.</p>",
        "<table border=1 cellpadding=6 cellspacing=0>",
        "<tr><th>candidate_id</th><th>side</th><th>neg_gap_min_m</th>"
        "<th>persistence_frames</th><th>min_center_dist_m</th>"
        "<th>physical_overlap_any</th><th>min_clearance_m</th>"
        "<th>verdict</th><th>GIF</th></tr>",
    ]
    for row in rows:
        color = VERDICT_COLOR.get(row["diagnostic_verdict"], "#ffffff")
        gif_rel = "gifs/" + Path(row["review_gif_path"]).name if row["review_gif_path"] else ""
        gif_cell = f"<img src='{gif_rel}' width='420'>" if gif_rel else "(no gif)"
        lines.append(
            f"<tr style='background-color:{color}'>"
            f"<td>{row['candidate_id']}</td>"
            f"<td>{row['front_or_rear']}</td>"
            f"<td>{float(row['negative_gap_min_m']):.2f}</td>"
            f"<td>{row['negative_gap_persistence_frames']}</td>"
            f"<td>{float(row['min_center_distance_m']):.2f}</td>"
            f"<td>{row['physical_overlap_any']}</td>"
            f"<td>{float(row['min_physical_clearance_m']):.2f}</td>"
            f"<td><b>{row['diagnostic_verdict']}</b></td>"
            f"<td>{gif_cell}</td>"
            "</tr>"
        )
    lines.append("</table></body></html>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
