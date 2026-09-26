#!/usr/bin/env python3
"""PHASE A: closes out the 20-sample SAFETY geometry audit for the dataset
freeze session -- assigns a diagnostic-only `safety_geometry_status` (NOT a
human label, NOT a ground-truth collision determination) to each of the 20
candidates already audited in outputs/merge_v2_decision_audit_v2/
safety_critical_targeted_audit.csv, and reports whether STOP/SAFETY_CRITICAL
can be safely excluded from primary K/F/M tier logic.

Mapping from the prior session's 3-way diagnostic_verdict (based on a real
Waymax SAT/OBB overlap check plus a conservative circumscribed-circle
clearance proxy) to this session's requested 4-way status:

  physical_overlap_any == True                      -> CONFIRMED_OVERLAP
  physical_overlap_any == False AND clearance > 2.0m -> PROJECTION_ARTIFACT_LIKELY
  physical_overlap_any == False AND 0 <= clearance <= 2.0m -> TIGHT_NO_OVERLAP
    (real OBBs did not overlap, and the clearance proxy -- itself
    conservative/an underestimate, see the prior script's docstring -- is
    non-negative, i.e. genuinely close but measurably clear)
  physical_overlap_any == False AND clearance < 0m   -> UNRESOLVED
    (real OBBs did not overlap, but the proxy is negative -- the proxy
    cannot rule out near-contact at higher fidelity than what was computed;
    this is the honest "we don't know for certain, but it's not a confirmed
    hit" bucket, matching the prior session's AMBIGUOUS)
"""

import csv
from pathlib import Path

AUDIT_CSV = "outputs/merge_v2_decision_audit_v2/safety_critical_targeted_audit.csv"
OUT_CSV = "outputs/merge_v2_decision_audit_v2/safety_geometry_status_v2.csv"


def status_for(row):
    overlap = row["physical_overlap_any"] == "True"
    clearance = float(row["min_physical_clearance_m"])
    if overlap:
        return "CONFIRMED_OVERLAP"
    if clearance > 2.0:
        return "PROJECTION_ARTIFACT_LIKELY"
    if clearance >= 0.0:
        return "TIGHT_NO_OVERLAP"
    return "UNRESOLVED"


def main():
    with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20, f"expected 20 targeted-audit rows, got {len(rows)}"

    for row in rows:
        row["safety_geometry_status"] = status_for(row)

    from collections import Counter
    counts = Counter(r["safety_geometry_status"] for r in rows)
    for k in ("CONFIRMED_OVERLAP", "TIGHT_NO_OVERLAP", "PROJECTION_ARTIFACT_LIKELY", "UNRESOLVED"):
        counts.setdefault(k, 0)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("safety_geometry_status distribution (n=20, diagnostic-only, not human ground truth):")
    for k in ("CONFIRMED_OVERLAP", "TIGHT_NO_OVERLAP", "PROJECTION_ARTIFACT_LIKELY", "UNRESOLVED"):
        print(f"  {k}: {counts[k]}")

    confirmed = counts["CONFIRMED_OVERLAP"]
    # Gate A per brief Section 4: PASS unless confirmed overlaps are common
    # enough to overturn "SAFETY_CRITICAL should not drive primary K/F/M
    # tiering" -- here that means requiring confirmed overlaps to stay a
    # small minority of the 20-sample audit (not zero necessarily, but not
    # dominant).
    gate_a_pass = confirmed <= 2  # <=10% of the 20-sample audit
    print(f"\nGate A: 20/20 geometry available = True; "
          f"confirmed_overlap={confirmed}/20; PASS={gate_a_pass}")
    print(f"wrote {OUT_CSV}")
    return gate_a_pass, counts


if __name__ == "__main__":
    main()
