"""The known-answer recovery benchmark must score what it claims and refuse what it cannot score.

Four properties, each of which a plausible implementation gets wrong:

* the ENDPOINTS anchor — a candidate identical to the answer scores 1.0, the untouched base scores
  0.0, and a partial recovery lands strictly between them;
* the ATTRIBUTION CLOSES — the per-step credits sum to the headline exactly, so a per-step number is
  a decomposition of the score rather than a second opinion about it;
* the GATE is arithmetic, not prose — a step that only clears a weaker correctness contract leaves
  the denominator, so the headline is not reachable by relaxing correctness;
* the FAIRNESS RULE FIRES — a candidate that reached its number by reading a withheld surface is
  REFUSED rather than scored, and so is a candidate nobody audited. A fairness rule that is only
  written down is not one, so the mutation is exercised end to end: a transcript that reads the key,
  through the real audit, into the real scorer.
"""

from __future__ import annotations

import json
import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf import recovery as R

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
_DESCRIPTOR = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"

CLEAN_AUDIT = {"clean": True, "hits": []}


# --------------------------------------------------------------------------------- a synthetic key
# Deliberately synthetic, and deliberately shaped like the real one: one step that removes most of
# the operations, one that removes NONE of them, and one that only clears a weaker gate. Every
# pathology this benchmark exists to survive is present in three steps, with no answer key in a
# tracked file.

BASE_FAMILIES = {"integer_arithmetic": 1000, "conversion": 400, "branch": 200, "floating_arithmetic": 100}
STEPS = [
    # name,              family delta (positive = removed),                       gate
    ("strength_reduce", {"integer_arithmetic": 600}, "bit_exact"),
    ("fsm_gate", {}, "bit_exact"),
    ("device_epilogue", {"conversion": 300, "floating_arithmetic": -50}, "chain_d_exact"),
]


def _key_document() -> dict:
    families = dict(BASE_FAMILIES)
    total = sum(families.values())
    levers = []
    for order, (name, delta, gate) in enumerate(STEPS, start=1):
        for family, moved in delta.items():
            families[family] = families.get(family, 0) - moved
        total -= sum(delta.values())
        levers.append(
            {
                "name": name,
                "order": order,
                "total_after": total,
                "marginal": sum(delta.values()),
                "family_delta": dict(delta),
                "gate": gate,
            }
        )
    return {
        "schema": R.KEY_SCHEMA,
        "benchmark": "synthetic_host_lane_recovery",
        "target": "synthetic",
        "metric": "host_dynamic_operations",
        "base": {"total": sum(BASE_FAMILIES.values()), "families": dict(BASE_FAMILIES)},
        "destination": {"total": total, "families": dict(families)},
        "levers": levers,
        "provenance": {"answer_surfaces": []},
    }


@pytest.fixture()
def key(tmp_path):
    path = tmp_path / R.KEY_FILENAME
    path.write_text(yaml.safe_dump(_key_document(), sort_keys=False))
    return R.load_key(path)


def _after(*names: str) -> dict[str, int]:
    """The family census a candidate that re-authored exactly ``names`` would emit."""
    families = dict(BASE_FAMILIES)
    for name, delta, _gate in STEPS:
        if name in names:
            for family, moved in delta.items():
                families[family] = families.get(family, 0) - moved
    return families


# ------------------------------------------------------------------------------------- endpoints


def test_a_candidate_identical_to_the_answer_scores_one(key):
    record = R.score_recovery(dict(key.destination_families), key, audit=CLEAN_AUDIT)
    assert record["recovery"] == pytest.approx(1.0)


def test_the_untouched_base_scores_zero(key):
    record = R.score_recovery(dict(key.base_families), key, audit=CLEAN_AUDIT)
    assert record["recovery"] == pytest.approx(0.0)


def test_a_partial_recovery_lands_strictly_between(key):
    record = R.score_recovery(_after("strength_reduce"), key, audit=CLEAN_AUDIT)
    assert 0.0 < record["recovery"] < 1.0


def test_the_per_step_attribution_sums_to_the_headline(key):
    record = R.score_recovery(_after("strength_reduce"), key, audit=CLEAN_AUDIT)
    attributed = sum(row["attributed"] for row in record["levers"])
    assert attributed == pytest.approx(record["recovery"] * record["denominator"], abs=1e-3)


def test_credit_lands_on_the_step_that_moved_that_family(key):
    """Attribution is by operation family, so re-authoring one step credits THAT step."""
    record = R.score_recovery(_after("strength_reduce"), key, audit=CLEAN_AUDIT)
    rows = {row["lever"]: row for row in record["levers"]}
    assert rows["strength_reduce"]["share"] == pytest.approx(1.0)
    assert rows["device_epilogue"]["attributed"] == pytest.approx(0.0)


def test_a_step_that_moves_no_operations_is_named_with_zero_weight(key):
    """The metric cannot see this step. Saying so is the honest form; averaging it away is not."""
    record = R.score_recovery(dict(key.destination_families), key, audit=CLEAN_AUDIT)
    row = next(r for r in record["levers"] if r["lever"] == "fsm_gate")
    assert row["weight"] == 0.0
    assert row["share"] is None
    assert "ZERO operations" in row["note"]


def test_over_achieving_one_family_is_not_extra_credit(key):
    """Removing MORE of a family than the reference did is a different program, not a better score."""
    families = _after("strength_reduce")
    families["integer_arithmetic"] = 0  # remove far more than the key's step did
    record = R.score_recovery(families, key, audit=CLEAN_AUDIT)
    assert record["recovery"] <= 1.0
    assert record["families"]["integer_arithmetic"]["recovered"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------- the gate


def test_a_step_behind_a_weaker_gate_leaves_the_denominator(key):
    strict = R.score_recovery(_after("strength_reduce"), key, required_gate="bit_exact", audit=CLEAN_AUDIT)
    loose = R.score_recovery(_after("strength_reduce"), key, audit=CLEAN_AUDIT)
    excluded = [row["lever"] for row in strict["gate"]["excluded_levers"]]
    assert "device_epilogue" in excluded
    assert strict["denominator"] < loose["denominator"]
    assert strict["recovery"] > loose["recovery"]  # same work, smaller honest denominator


def test_an_unestablished_gate_is_never_admitted(tmp_path):
    document = _key_document()
    for lever in document["levers"]:
        lever["gate"] = R.GATE_UNKNOWN
    path = tmp_path / R.KEY_FILENAME
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    key = R.load_key(path)
    record = R.score_recovery(dict(key.destination_families), key, audit=CLEAN_AUDIT)
    assert record["recovery"] is None
    assert "excluded" in record["consequence"] or "denominator" in record["consequence"]


# --------------------------------------------------------------------------------- key integrity


def test_a_key_whose_steps_do_not_close_is_refused(tmp_path):
    document = _key_document()
    document["destination"]["total"] += 7
    path = tmp_path / R.KEY_FILENAME
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(R.RecoveryKeyError, match="do not close"):
        R.load_key(path)


def test_a_missing_key_says_so_rather_than_inventing_one(tmp_path):
    with pytest.raises(R.RecoveryKeyError, match="ANSWER SURFACE"):
        R.load_key(tmp_path / "absent.yaml")


# -------------------------------------------------------------------------------------- fairness


def test_an_unaudited_candidate_is_refused_not_scored(key):
    record = R.score_recovery(dict(key.destination_families), key, audit=None)
    assert record["recovery"] is None
    assert record["admission"]["status"] == "refused_unaudited"


def test_an_audit_hit_that_read_withheld_content_refuses_the_score(key):
    audit = {"clean": False, "hits": [{"tool": "Bash", "kind": "path_read", "token": R.KEY_FILENAME, "input": "cat"}]}
    record = R.score_recovery(dict(key.destination_families), key, audit=audit)
    assert record["recovery"] is None
    assert record["admission"]["status"] == "refused_answer_access"


def test_an_unrecognised_hit_kind_refuses_too(key):
    """The vocabulary fails closed upstream; this asserts the refusal reaches the score."""
    audit = {"clean": True, "hits": [{"tool": "Bash", "kind": "a_kind_nobody_declared"}]}
    record = R.score_recovery(dict(key.destination_families), key, audit=audit)
    assert record["recovery"] is None


def test_advisory_hits_do_not_block_a_score(key):
    """Guard the guard: a blocked probe means the mask WORKED, and must not refuse a clean run."""
    audit = {"clean": True, "hits": [{"tool": "Bash", "kind": "blocked_probe", "token": "golden.yaml"}]}
    record = R.score_recovery(dict(key.destination_families), key, audit=audit)
    assert record["recovery"] == pytest.approx(1.0)


# ------------------------------------------------------------- the mutation, through the real audit


@pytest.fixture()
def audit_transcript(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(_DESCRIPTOR))
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    from merlin_experiments.phase1.audit import AnswerAudit

    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = load_target_experiment(_DESCRIPTOR)
    L = AnswerAudit.for_descriptor(descriptor, _DESCRIPTOR.parent / "input_bundles" / "raw_baseline_hwbringup_v0")

    return L.audit_transcript


def _transcript(tmp_path, command, stdout):
    lines = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": command}}]},
        },
        {
            "type": "user",
            "tool_use_result": {"stdout": stdout},
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": stdout or "(Bash completed with no output)"}
                ]
            },
        },
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


def test_reading_the_key_is_a_violation_and_the_score_is_refused(key, audit_transcript, tmp_path):
    """THE MUTATION. A candidate that reaches the number by reading the key must not be scored.

    End to end and with nothing stubbed: a real transcript that reads a real key path, through the
    real transcript audit, into the real scorer. If the audit does not classify this as a violation,
    or the scorer scores it anyway, this test fails -- which is the only form in which a fairness
    rule exists.
    """
    verdict = audit_transcript(
        _transcript(tmp_path, f"cat {key.path}", "levers:\n  - name: strength_reduce\n    marginal: 600"),
        arm="merlin_assisted",
    )
    assert verdict["clean"] is False, "the audit did not see a read of the recovery key"
    assert any(hit["kind"] == "path_read" for hit in verdict["hits"])
    record = R.score_recovery(dict(key.destination_families), key, audit=verdict)
    assert record["recovery"] is None
    assert record["admission"]["status"] == "refused_answer_access"


def test_a_blocked_probe_of_the_key_still_scores(key, audit_transcript, tmp_path):
    """Guard the guard: the same command with the mask WORKING returns nothing and is not a leak."""
    verdict = audit_transcript(_transcript(tmp_path, f"cat {key.path} 2>/dev/null", ""), arm="merlin_assisted")
    assert verdict["clean"] is True
    assert R.score_recovery(dict(key.destination_families), key, audit=verdict)["recovery"] is not None


def test_the_key_and_its_declared_sources_are_masked_surfaces(tmp_path, monkeypatch):
    """The key is in the DERIVED answer-surface set, and so is the material it names."""
    import importlib

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    from merlin.targetgen.target_experiment import load_target_experiment

    source = tmp_path / "answer_material"
    source.mkdir()
    (source / "lever.patch").write_text("--- a\n+++ b\n")
    document = _key_document()
    document["provenance"]["answer_surfaces"] = [str(source)]
    key_file = tmp_path / R.KEY_FILENAME
    key_file.write_text(yaml.safe_dump(document, sort_keys=False))
    monkeypatch.setenv(R.KEY_ENV, str(key_file))

    assert key_file.resolve() in surfaces_module.recovery_key_files()
    assert source in surfaces_module.recovery_source_dirs()
    te = load_target_experiment(_DESCRIPTOR)
    paths = {surface.path for surface in surfaces_module.answer_surfaces(te) if surface.origin == "recovery_key"}
    assert key_file.resolve() in paths
    assert source in paths
    tokens = surfaces_module.audit_tokens(te)["answer"]
    assert R.KEY_FILENAME in tokens


def test_the_backend_package_is_a_masked_surface_but_its_contract_grant_is_not_a_token():
    """The derivations are withheld; the `contracts/` grant an arm must read is not accused of it."""
    import importlib

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(_DESCRIPTOR)
    backend = surfaces_module.backend_package_dir(te)
    assert backend is not None
    assert backend in {s.path for s in surfaces_module.answer_surfaces(te) if s.origin == "backend"}
    granted = f"{te.rtl_facts_pin}gemmini_rtl_facts.json"
    assert not [t for t in surfaces_module.audit_tokens(te)["answer"] if t in granted]


# ------------------------------------------------------------------------------------- ablations


def test_no_instrument_subset_ever_names_a_step(key):
    """The ablation surface may hand over a destination. It may never hand over the plan."""
    names = {name for name, _delta, _gate in STEPS}
    for size in range(len(R.INSTRUMENTS) + 1):
        for start in range(len(R.INSTRUMENTS) - size + 1):
            subset = R.INSTRUMENTS[start : start + size]
            blob = json.dumps(R.feedback(key, instruments=subset))
            leaked = sorted(name for name in names if name in blob)
            assert not leaked, f"instruments {subset} leaked step name(s) {leaked}"


def test_withholding_an_instrument_removes_its_channel(key):
    full = R.feedback(key, instruments=("destination_total", "destination_families"))
    ablated = R.feedback(key, instruments=("destination_total",))
    assert "destination_families" in full and "destination_families" not in ablated
    assert ablated["instruments_withheld"].count("destination_families") == 1


def test_an_undeclared_instrument_is_refused(key):
    with pytest.raises(R.RecoveryKeyError, match="unknown instrument"):
        R.feedback(key, instruments=("read_the_patches",))


def test_the_structural_instrument_reports_rather_than_raises_without_a_reference(key):
    record = R.feedback(key, instruments=("structural_gap",), command_buffer={"commands": []})
    assert record["structural"]["status"] == "key_declares_no_reference"


# ----------------------------------------------------------------------------------- minting a key


def test_a_minted_key_derives_every_marginal_from_adjacent_measurements():
    ledger = [
        {"label": "00_base", "ops": {"total": 100, "families": {"integer_arithmetic": 60, "branch": 40}}},
        {"label": "step_one", "ops": {"total": 70, "families": {"integer_arithmetic": 30, "branch": 40}}},
        {"label": "step_two", "ops": {"total": 55, "families": {"integer_arithmetic": 30, "branch": 25}}},
    ]
    document = R.mint_key(ledger, benchmark="b", target="t", metric="host_dynamic_operations")
    assert [lever["marginal"] for lever in document["levers"]] == [30, 15]
    assert document["levers"][0]["family_delta"] == {"integer_arithmetic": 30}
    assert all(lever["gate"] == R.GATE_UNKNOWN for lever in document["levers"])


def test_a_minted_key_round_trips_through_the_loader(tmp_path):
    ledger = [
        {"label": "00_base", "ops": {"total": 100, "families": {"integer_arithmetic": 60, "branch": 40}}},
        {"label": "step_one", "ops": {"total": 70, "families": {"integer_arithmetic": 30, "branch": 40}}},
    ]
    document = R.mint_key(ledger, benchmark="b", target="t", metric="m", gates={"step_one": "bit_exact"})
    path = tmp_path / R.KEY_FILENAME
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    key = R.load_key(path)
    assert key.destination_total == 70
    assert R.score_recovery({"integer_arithmetic": 30, "branch": 40}, key, audit=CLEAN_AUDIT)["recovery"] == 1.0


def test_the_repo_is_location_independent():
    """The suite resolves its own checkout; this file must not have pinned a path."""
    assert repo_root().is_dir()
