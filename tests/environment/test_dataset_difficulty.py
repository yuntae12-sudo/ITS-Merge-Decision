"""Stage C-1: dataset difficulty descriptor + interaction-tag tests.

Covers Section 16's leakage/regression checklist for the new
data/manifests/phase2_dataset_difficulty.csv +
configs/phase2_difficulty.yaml artifacts:
  - scene-group disjointness / exactly-once assignment (reuses the
    canonical split, joined not recomputed)
  - difficulty features contain no policy outcome / no future ego
    state (structural: the descriptor generator only calls
    env.reset(), never env.step() or any policy)
  - presence flags correctly gate missing slots
  - difficulty tagging is deterministic (rerunning produces identical
    output)
  - canonical split unchanged
  - Phase 1 canonical labels unchanged
"""

import ast
import csv

import yaml

from src.environment.dataset_split import load_split_manifest
from src.environment.observation_builder import TTC_CAP_S

DESCRIPTOR_PATH = "data/manifests/phase2_dataset_difficulty.csv"
DIFFICULTY_CONFIG_PATH = "configs/phase2_difficulty.yaml"
DESCRIPTOR_SCRIPT_PATH = "scripts/build_phase2_dataset_descriptors.py"
MANEUVER_TABLE = "outputs/phase1/training_10shard_pilot/training_visual_merge_maneuvers.csv"

TAG_NAMES = (
    "OPEN_GAP",
    "FRONT_CONSTRAINED",
    "REAR_CONSTRAINED",
    "SOURCE_FOLLOW_RELEVANT",
    "LATE_MERGE",
    "DENSE_INTERACTION",
    "MULTI_CONSTRAINT",
)


def _load_descriptor_rows():
    with open(DESCRIPTOR_PATH, newline="") as f:
        return list(csv.DictReader(f))


# ======================================================================
# Coverage / identity
# ======================================================================


def test_descriptor_covers_all_168_canonical_maneuvers_exactly_once():
    rows = _load_descriptor_rows()
    assert len(rows) == 168

    maneuver_ids = [r["maneuver_id"] for r in rows]
    assert len(maneuver_ids) == len(set(maneuver_ids))


def test_descriptor_split_is_joined_not_recomputed():
    """The descriptor CSV's split column must exactly match the
    canonical split manifest -- never an independently recomputed
    value (Section 14's explicit requirement)."""

    rows = _load_descriptor_rows()
    canonical_split = {r.maneuver_id: r.split for r in load_split_manifest()}

    for row in rows:
        assert row["split"] == canonical_split[row["maneuver_id"]], (
            f"{row['maneuver_id']}: descriptor split={row['split']} != "
            f"canonical split={canonical_split[row['maneuver_id']]}"
        )


def test_descriptor_train_validation_counts_match_canonical_split():
    rows = _load_descriptor_rows()
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]

    assert len(train) == 110
    assert len(validation) == 58


# ======================================================================
# Presence-flag gating
# ======================================================================


def test_presence_flags_gate_slot_values_correctly():
    """When a slot is absent (`*_present == 0`), its gap/relative-speed
    must be the fixed 0.0 sentinel and its TTC must be exactly
    TTC_CAP_S -- never interpreted as a real physical measurement."""

    rows = _load_descriptor_rows()

    for slot in ("target_front", "target_rear", "source_front"):
        for row in rows:
            present = row[f"{slot}_present"] == "1"
            gap = float(row[f"{slot}_gap"])
            rel_speed = float(row[f"{slot}_relative_speed"])
            ttc = float(row[f"{slot}_ttc"])

            if not present:
                assert gap == 0.0, f"{row['maneuver_id']}/{slot}: absent but gap={gap}"
                assert rel_speed == 0.0, f"{row['maneuver_id']}/{slot}: absent but rel_speed={rel_speed}"
                assert ttc == TTC_CAP_S, f"{row['maneuver_id']}/{slot}: absent but ttc={ttc}"


def test_closing_speed_descriptors_are_present_gated():
    """front_closing_speed/rear_closing_speed must be blank (None) when
    the corresponding target slot is absent -- never a fabricated
    zero standing in for "no vehicle"."""

    rows = _load_descriptor_rows()

    for row in rows:
        front_present = row["target_front_present"] == "1"
        rear_present = row["target_rear_present"] == "1"

        if not front_present:
            assert row["front_closing_speed"] == ""
        else:
            assert row["front_closing_speed"] != ""

        if not rear_present:
            assert row["rear_closing_speed"] == ""
        else:
            assert row["rear_closing_speed"] != ""


def test_presence_counts_reported_are_internally_consistent():
    """Sanity check: presence count + absence count == split size, for
    every tracked slot (Section 7's explicit reporting requirement)."""

    rows = _load_descriptor_rows()
    for split_name in ("train", "validation"):
        split_rows = [r for r in rows if r["split"] == split_name]
        for slot in ("target_front", "target_rear", "source_front"):
            present = sum(1 for r in split_rows if r[f"{slot}_present"] == "1")
            absent = sum(1 for r in split_rows if r[f"{slot}_present"] == "0")
            assert present + absent == len(split_rows)


# ======================================================================
# No policy outcome / no future ego state (structural)
# ======================================================================


def test_descriptor_columns_contain_no_policy_outcome_fields():
    """The descriptor table must never contain a policy-outcome column
    (success/outcome/commit_frame/action/reward) -- difficulty must be
    derivable independent of any policy's behavior."""

    rows = _load_descriptor_rows()
    forbidden_words = {"outcome", "success", "reward", "action", "commit", "policy"}

    fieldnames = set(rows[0].keys())
    for name in fieldnames:
        words = set(name.lower().split("_"))
        overlap = words & forbidden_words
        assert not overlap, (
            f"descriptor column {name!r} looks policy-outcome-derived "
            f"(matched word(s) {overlap})"
        )


def test_descriptor_generator_never_calls_env_step():
    """Structural leakage guard: the descriptor generator script must
    only call MergeEnvironment.reset() (a single-frame, causal
    snapshot at decision_start_frame) and must NEVER call .step() --
    stepping the environment would require choosing an action, which
    would make the resulting descriptors policy-conditioned rather
    than policy-independent physical properties of the scenario at its
    start frame."""

    with open(DESCRIPTOR_SCRIPT_PATH) as f:
        source = f.read()

    tree = ast.parse(source)
    step_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "step"
    ]
    assert not step_calls, (
        "build_phase2_dataset_descriptors.py must not call .step() on "
        "the environment -- descriptors must be reset()-only snapshots"
    )


# ======================================================================
# Determinism
# ======================================================================


def test_difficulty_tagging_is_deterministic(tmp_path):
    """Rerunning the tag-assignment logic on the same descriptor row
    must produce byte-identical tags -- no hidden randomness."""

    import sys

    sys.path.insert(0, "scripts")
    from apply_phase2_difficulty_tags import compute_tags

    with open(DIFFICULTY_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    rows = _load_descriptor_rows()
    for row in rows[:20]:
        tags_a = compute_tags(row, config)
        tags_b = compute_tags(row, config)
        assert tags_a == tags_b


def test_difficulty_tags_present_in_descriptor_table():
    rows = _load_descriptor_rows()
    for row in rows:
        for tag in TAG_NAMES:
            assert tag in row
            assert row[tag] in ("0", "1")


def test_multi_constraint_tag_matches_its_own_definition():
    """MULTI_CONSTRAINT must equal (count of the 5 component tags >= 2)
    for every row -- catches a stale/inconsistent recomputation."""

    rows = _load_descriptor_rows()
    components = (
        "FRONT_CONSTRAINED",
        "REAR_CONSTRAINED",
        "SOURCE_FOLLOW_RELEVANT",
        "LATE_MERGE",
        "DENSE_INTERACTION",
    )
    for row in rows:
        count = sum(int(row[c]) for c in components)
        expected_multi = int(count >= 2)
        assert int(row["MULTI_CONSTRAINT"]) == expected_multi, row["maneuver_id"]


def test_open_gap_tag_matches_absence_of_all_three_slots():
    rows = _load_descriptor_rows()
    for row in rows:
        all_absent = (
            row["target_front_present"] == "0"
            and row["target_rear_present"] == "0"
            and row["source_front_present"] == "0"
        )
        assert int(row["OPEN_GAP"]) == int(all_absent), row["maneuver_id"]


# ======================================================================
# Canonical preservation
# ======================================================================


def test_canonical_split_manifest_row_counts_unchanged():
    rows = load_split_manifest()
    train = [r for r in rows if r.split == "train"]
    validation = [r for r in rows if r.split == "validation"]
    assert len(train) == 110
    assert len(validation) == 58


def test_phase1_canonical_maneuver_table_row_count_unchanged():
    with open(MANEUVER_TABLE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 168


def test_scalar_difficulty_tier_explicitly_not_used():
    """Section 11: a scalar EASY/MEDIUM/HARD tier must not be silently
    invented -- the frozen config must explicitly record the decision
    not to use one (or DEFERRED), never leave it undocumented."""

    with open(DIFFICULTY_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    assert config["scalar_difficulty_tier"] in ("NOT_USED", "DEFERRED")
