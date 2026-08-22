"""The whole-model capstone must not pass a submission that ran the model on the CPU.

Two independent defects made it do exactly that, and either one alone was sufficient:

1. Every model capsule shipped ``expected.instruction_classes: []`` and ``semantic.must_accelerate:
   false``. No coverage requirement and no acceleration requirement means a correct-numbers CPU run is a
   pass — the vacuity that was removed from the op capsules years earlier and never from the one capsule
   the whole suite builds toward, which is also the only one anybody quotes.
2. Even with must_accelerate on, the capstone could not raise a violation, because a violation requires
   an ELIGIBLE region and the eligibility oracle read the model's ENTRY tensor as its operand: a token-id
   vector, ``i64``, against a contract declaring ``int8``. Every capstone was "ineligible", and an
   ineligible region legitimately falls back.

Both are fixed by DERIVATION, not by authoring: the demand comes from the model's own captured linalg
crossed with the target's declared capabilities and its own role census.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import coverage_report as cr
from merlin.targetgen import eligibility as el
from merlin.targetgen.capsule_source import model_accelerator_demand


def _capstones():
    """(target, capsule dict, linalg text) for every locally generated model capsule.

    Model capsules are generated artifacts and untracked (they ship weights and a golden), so a fresh
    clone has none and these tests skip rather than fail.
    """
    root = repo_root() / "merlin" / "contract" / "capsules"
    out = []
    for prof in sorted((root / "profiles").glob("*.yaml")):
        t = prof.stem
        if t.endswith(".hidden"):
            continue
        d = (root / t / "model") if (root / t / "model").is_dir() else (root / "model")
        for cf in sorted(d.glob("*/capsule.yaml")) if d.is_dir() else []:
            lin = cf.parent / "capsule.interface.mlir"
            if lin.is_file():
                out.append((t, yaml.safe_load(cf.read_text()), lin.read_text()))
    return out


def _binding(target: str):
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.target_experiment import load_target_experiment
    root = repo_root() / "merlin" / "contract" / "capsules"
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets" / target / "target_experiment.yaml")
    prof = yaml.safe_load((root / "profiles" / f"{target}.yaml").read_text()) or {}
    return CS.derive_binding(te, prof.get("datapath") or {})


def test_a_model_region_reads_the_compile_dtype_not_the_token_ids():
    """``inputs[]`` on a whole model is its entry tensor. For a language model that is i64 token ids —
    the dtype of nothing the accelerator computes on."""
    caps = _capstones()
    if not caps:
        pytest.skip("no model capsules generated in this checkout")
    checked = 0
    for _t, cap, _lin in caps:
        declared = ((cap.get("operation") or {}).get("attributes") or {}).get("dtype")
        entry = next((i.get("dtype") for i in cap.get("inputs") or [] if i.get("role") == "input"), None)
        if not declared or entry == declared:
            continue                      # only the mismatching case is interesting
        r = cr._capsule_region(cap)
        assert r.in_dtype == declared and r.weight_dtype == declared, (r.in_dtype, entry, declared)
        checked += 1
    if not checked:
        pytest.skip("no capstone whose entry dtype differs from its compile dtype")


def _grounded_pairs():
    """(target, capsule) pairs where THIS target's own derivation grounds the demand.

    Not simply "the capsule says must_accelerate": a target with no model directory of its own falls back
    to the shared corpus root and so reads another target's capstone, whose demand was grounded against a
    different device's capabilities. Asserting it there tests the wrong claim.
    """
    out = []
    for t, cap, lin in _capstones():
        if not (cap.get("semantic") or {}).get("must_accelerate"):
            continue
        try:
            fam, classes = model_accelerator_demand(lin, _binding(t))
        except Exception:                                        # noqa: BLE001 — unresolvable target
            continue
        if fam and classes:
            out.append((t, cap))
    return out


def test_a_grounded_capstone_fails_a_cpu_only_submission():
    """The whole point. An accelerated run passes the same capsule, so this is not a blanket fail."""
    grounded = _grounded_pairs()
    if not grounded:
        pytest.skip("no capstone with a grounded accelerator demand in this checkout")
    for t, cap in grounded:
        name = cap["name"]
        cpu = cr._acceleratable_coverage([{"capsule": name, "tiers": {}}], {name: cap}, t)
        assert name in cpu["must_accelerate_violations"], \
            f"{t}/{name}: a CPU-only whole-model run must not pass the capstone"
        assert not cpu["must_accelerate_pass"]
        hit = cr._acceleratable_coverage(
            [{"capsule": name, "tiers": {"L2": {"status": "pass"}}}], {name: cap}, t)
        assert hit["must_accelerate_pass"], f"{t}/{name}: an accelerated run must still pass"


def test_a_grounded_capstone_demands_real_instruction_classes():
    grounded = _grounded_pairs()
    if not grounded:
        pytest.skip("no capstone with a grounded accelerator demand in this checkout")
    for t, cap in grounded:
        assert (cap.get("expected") or {}).get("instruction_classes"), \
            f"{t}/{cap['name']}: must_accelerate asserted with no coverage requirement behind it"


def test_the_capstone_demand_matches_the_op_capsules_of_the_same_family():
    """One deriver, so the capstone and an op capsule cannot disagree about this target's sequence.

    Reaching past ``binding.classes_for`` to the ISA taxonomy directly is what produced an EMPTY class
    list on a target whose own op capsules carry the full eight-class systolic sequence — the taxonomy is
    only one of the three regimes that callable resolves.
    """
    caps = _capstones()
    if not caps:
        pytest.skip("no model capsules generated in this checkout")
    checked = 0
    for t, cap, lin in caps:
        try:
            b = _binding(t)
        except Exception:                                        # noqa: BLE001 — unresolvable target
            continue
        fam, classes = model_accelerator_demand(lin, b)
        if not classes:
            continue
        assert fam == "contraction", fam
        assert classes == list(b.classes_for(op="matmul", output_dtype=b.cap_dtype(b.operand_dtype))), \
            f"{t}: the capstone's classes drifted from the target's own contraction sequence"
        checked += 1
    if not checked:
        pytest.skip("no target in this checkout derives a class sequence")


def test_the_demand_fails_closed_when_it_cannot_be_grounded():
    """An ungrounded demand fails a CONFORMANT submission — the one direction running it cannot catch —
    so nothing is asserted unless both halves derive."""
    class _NoTarget:
        target = None
        operand_dtype = "int8"

        def cap_dtype(self, d):
            return d

        def classes_for(self, **_):
            return []

    assert model_accelerator_demand("module {}", _NoTarget()) == (None, [])
    # and a real binding with no module to read
    caps = _capstones()
    if caps:
        try:
            b = _binding(caps[0][0])
        except Exception:                                        # noqa: BLE001
            return
        assert model_accelerator_demand("", b) == (None, [])


def test_an_ungrounded_capstone_records_why():
    """Silence would be indistinguishable from "this target has no accelerator demand"."""
    for t, cap, _lin in _capstones():
        sem = cap.get("semantic") or {}
        if sem.get("must_accelerate"):
            continue
        assert sem.get("not_asserted_reason"), \
            f"{t}/{cap['name']}: must_accelerate withheld with no reason recorded"


# --- the tier verdict must follow the tiles, and a failing tier must not pass --------------------

def _grade_with(mesh_exec: dict, declared=("L0", "L1", "L2", "L3")):
    """Drive the tier-derivation block with a synthetic mesh_execution record."""
    from merlin import compile_cli as CCLI
    from merlin.targetgen import capsule_runner as CR

    capsule = {"name": "M_probe", "kind": "model",
               "operation": {"op": "model", "attributes": {"model": "probe", "compile_dtype": "int8",
                                                           "dtype": "i8"}},
               "required_oracle_tiers": list(declared),
               "semantic": {"semantic_family": "contraction", "must_accelerate": True}}
    out = {"status": "verified", "verify": {"gate_ok": True}, "mesh_execution": mesh_exec}
    # `_grade_model_capsule` imports compile_model INSIDE the function, so the module attribute is what
    # has to move.
    real = CCLI.compile_model
    CCLI.compile_model = lambda *a, **k: out
    try:
        return CR._grade_model_capsule(capsule, target="gemmini", timeout=1)
    finally:
        CCLI.compile_model = real


def test_a_tier_passes_when_every_tile_passed():
    r = _grade_with({"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                     "n_unavailable": 0, "n_unsynthesizable": 0})
    assert r["tiers"] == {"L3": "pass"}, r["tiers"]
    assert r["status"] == "pass", r


def test_a_tier_that_ran_and_failed_is_not_a_pass():
    """The contradiction this guards: `status: pass` printed beside `tiers: {L3: fail}`, with the
    flattering half being the one a reader takes away."""
    r = _grade_with({"n_tiles": 15, "n_passed": 14, "n_failed": 1,
                     "n_unavailable": 0, "n_unsynthesizable": 0})
    assert r["tiers"] == {"L3": "fail"}, r["tiers"]
    assert r["status"] == "fail", r
    assert r["failure"]["category"] == "FUNCTIONAL_MISMATCH"


@pytest.mark.parametrize("key", ["n_unavailable", "n_unsynthesizable"])
def test_an_unrun_tile_is_not_counted_as_a_pass(key):
    """NOT-RUN-IS-NOT-PASS at tile granularity: a layer the oracle could not run leaves the model's
    accelerator claim unproven, so the tier cannot pass on the strength of the tiles that did run."""
    r = _grade_with({"n_tiles": 15, "n_passed": 14, "n_failed": 0,
                     "n_unavailable": 0, "n_unsynthesizable": 0, key: 1})
    assert r["tiers"] == {"L3": "fail"}, r["tiers"]
    assert r["status"] == "fail", r


def test_no_tiles_at_all_is_reported_unknown_not_failed():
    """Distinct from a failing tier: nothing ran, so there is no verdict to report either way."""
    r = _grade_with({"n_tiles": 0, "n_passed": 0, "n_failed": 0})
    assert r["tiers"] == {}
    assert r["status"] == "incomplete"
    assert r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
