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


@pytest.fixture(autouse=True)
def _descriptor_pinned_host_lane(monkeypatch):
    """These tests isolate capstone verdict logic from the independently tested host-lane resolver."""
    from merlin.targetgen import capsule_runner as CR
    monkeypatch.setattr(CR, "_resolve_model_host_lane", lambda target, dtype: (
        None, repo_root() / "frozen-test-host", {
            "package_sha256": "a" * 64,
            "dtype_strategy": "int8_w8a8" if dtype == "int8" else "fp32",
        }))


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
        # The must_accelerate assertions below turn on ELIGIBILITY, which comes from the target's
        # capability map. That map is derived into out/artifacts/, which is generated and untracked --
        # so on a checkout that has not derived this target it resolves to nothing, every region reads
        # ineligible, and `violated` is False for a reason that says nothing about the capstone. A pair
        # whose eligibility cannot be derived here is skipped rather than asserted: a check that could
        # not run must not report a failure.
        try:
            from merlin.targetgen import eligibility as _el
            if not _el.capability_map_for_target(t):
                continue
        except Exception:                                        # noqa: BLE001 — no derived contract
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

def _statuses(r):
    """Tier -> status, from the rich per-tier objects the merged row carries. The row is the same shape
    an op capsule produces, which is what capsule_result.schema.json requires and what routes it through
    the shared fail-closed gates; a tier that is honestly N/A for a whole model (L0/L1 interpret a command
    buffer, and a model has none) is reported as such rather than omitted."""
    return {t: (v or {}).get("status") for t, v in (r.get("tiers") or {}).items()}


def _passed(r):
    """The tiers that actually certified — the guarantee, independent of how the row is shaped."""
    return {t: v for t, v in _statuses(r).items() if v == "pass"}


def _grade_with(mesh_exec: dict, declared=("L0", "L1", "L2", "L3"), *, on_mesh=15, fallback=0):
    """Drive the tier-derivation block with a synthetic TILE-certification record.

    The tile record lives under ``mesh_tile_verification``; ``mesh_execution`` is the separate record of
    what happened to the MODEL's own layers. They shared a key once, and the tile record clobbered the
    model one — so these tests supply both, with the model fully on the mesh unless a case says otherwise.
    """
    from merlin import compile_cli as CCLI
    from merlin.targetgen import capsule_runner as CR

    capsule = {"name": "M_probe", "kind": "model",
               "operation": {"op": "model", "attributes": {"model": "probe", "compile_dtype": "int8",
                                                           "dtype": "i8"}},
               "required_oracle_tiers": list(declared),
               "semantic": {"semantic_family": "contraction", "must_accelerate": True}}
    out = {"status": "verified", "verify": {"gate_ok": True},
           "mesh_tile_verification": mesh_exec,
           "mesh_execution": {"target": "gemmini", "matmul_layers_routed": on_mesh + fallback,
                              "matmul_layers_on_mesh": on_mesh,
                              "matmul_layers_host_fallback": fallback}}
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
    assert _passed(r) == {"L3": "pass"}, r["tiers"]
    assert r["status"] == "pass", r


def test_a_tier_that_ran_and_failed_is_not_a_pass():
    """The contradiction this guards: `status: pass` printed beside `tiers: {L3: fail}`, with the
    flattering half being the one a reader takes away."""
    r = _grade_with({"n_tiles": 15, "n_passed": 14, "n_failed": 1,
                     "n_unavailable": 0, "n_unsynthesizable": 0})
    assert _statuses(r).get("L3") == "fail", r["tiers"]
    assert _passed(r) == {}, "a failing tile certifies nothing"
    assert r["status"] == "fail", r
    assert r["failure"]["category"] == "FUNCTIONAL_MISMATCH"


@pytest.mark.parametrize("key", ["n_unavailable", "n_unsynthesizable"])
def test_an_unrun_tile_is_not_counted_as_a_pass(key):
    """NOT-RUN-IS-NOT-PASS at tile granularity: a layer the oracle could not run leaves the model's
    accelerator claim unproven, so the tier cannot pass on the strength of the tiles that did run."""
    r = _grade_with({"n_tiles": 15, "n_passed": 14, "n_failed": 0,
                     "n_unavailable": 0, "n_unsynthesizable": 0, key: 1})
    assert _statuses(r).get("L3") == "fail", r["tiers"]
    assert _passed(r) == {}
    assert r["status"] == "fail", r


def test_nothing_ran_at_all_is_reported_unknown_not_failed():
    """Distinct from a failing tier: nothing ran, so there is no verdict to report either way. Nothing
    ran means BOTH records are empty — no certified tile, and no layer of the model itself on the mesh.
    A tile record alone was never the whole question, and the model's own accounting is now the half that
    decides, so this case has to withhold both."""
    r = _grade_with({"n_tiles": 0, "n_passed": 0, "n_failed": 0}, on_mesh=0)
    assert _passed(r) == {}
    assert r["status"] == "incomplete"
    assert r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


# --------------------------------------------------------------------------- model vs tile evidence
def _model_capsule(tmp_path, **semantic):
    import yaml
    d = tmp_path / "M0_probe"
    d.mkdir(parents=True, exist_ok=True)
    cap = {"name": "M0_probe", "kind": "model", "label": "public",
           "source_role": "pytorch_model_slice", "source_reference": "probe",
           "operation": {"op": "model", "attributes": {"model": "probe", "compile_dtype": "fp32",
                                                       "dtype": "fp32", "out": "Y0"}},
           "numeric_policy": {"compare": "tolerance_float", "dtype": "f32", "atol": 0.1, "rtol": 0.1},
           "expected": {"instruction_classes": []},
           "required_oracle_tiers": ["L3"],
           "semantic": {"generalization_axis": "model", **semantic},
           "__dir__": str(d)}
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap), encoding="utf-8")
    return cap


def _fake_compile(monkeypatch, *, on_mesh, fallback, tiles_pass=15):
    """A compile_model whose MODEL ran `on_mesh` layers on the accelerator while its synthesized TILE
    certification is perfect — the exact shape that used to read as a pass."""
    def _cm(*a, **k):
        return {"status": "verified",
                "verify": {"gate_ok": True, "fp32_cos": 1.0, "ok": True},
                "mesh_execution": {"target": k.get("target"), "matmul_layers_routed": on_mesh + fallback,
                                   "matmul_layers_on_mesh": on_mesh,
                                   "matmul_layers_host_fallback": fallback},
                "mesh_tile_verification": {"n_tiles": tiles_pass, "n_passed": tiles_pass, "n_failed": 0,
                                           "n_unavailable": 0, "n_unsynthesizable": 0, "per_tile": []}}
    import merlin.compile_cli as _cc
    monkeypatch.setattr(_cc, "compile_model", _cm)


def test_a_model_that_never_reached_the_mesh_cannot_pass(tmp_path, monkeypatch):
    """The defect this exists for: atlas routed 15 matmul layers, the dispatch runtime fell back to the
    host kernel on all 15, and the capstone still reported `pass` with `lane: mesh` -- because the tile
    record ("15 of 15 passed") had overwritten the model record under the same key."""
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    _fake_compile(monkeypatch, on_mesh=0, fallback=15)
    cap = _model_capsule(tmp_path, must_accelerate=True, eligible="auto",
                         semantic_family="contraction")
    res = _grade_model_capsule(cap, target="probe_target", timeout=1)
    assert res["status"] == "fail", res
    assert res["failure"]["category"] == "FALLBACK_ON_ELIGIBLE_REGION"
    assert "0 matmul" in res["failure"]["detail"] and "15 fell back" in res["failure"]["detail"]


def test_a_partial_fallback_is_also_a_failure(tmp_path, monkeypatch):
    """must_accelerate means EVERY eligible region accelerates; 14 of 15 is still a fallback."""
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    _fake_compile(monkeypatch, on_mesh=14, fallback=1)
    cap = _model_capsule(tmp_path, must_accelerate=True, eligible="auto",
                         semantic_family="contraction")
    res = _grade_model_capsule(cap, target="probe_target", timeout=1)
    assert res["status"] == "fail", res
    assert res["failure"]["category"] == "FALLBACK_ON_ELIGIBLE_REGION"


def test_a_model_fully_on_the_mesh_is_not_blocked(tmp_path, monkeypatch):
    """The guard must not reject the case it exists to certify."""
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    _fake_compile(monkeypatch, on_mesh=15, fallback=0)
    cap = _model_capsule(tmp_path, must_accelerate=True, eligible="auto",
                         semantic_family="contraction")
    res = _grade_model_capsule(cap, target="probe_target", timeout=1)
    assert res["status"] == "pass", res


def test_missing_per_layer_accounting_is_incomplete_not_pass(tmp_path, monkeypatch):
    """No accounting is not evidence of acceleration."""
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    def _cm(*a, **k):
        return {"status": "verified", "verify": {"gate_ok": True, "fp32_cos": 1.0, "ok": True},
                "mesh_execution": {"target": k.get("target")},        # no counts at all
                "mesh_tile_verification": {"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                                           "n_unavailable": 0, "n_unsynthesizable": 0, "per_tile": []}}
    import merlin.compile_cli as _cc
    monkeypatch.setattr(_cc, "compile_model", _cm)
    cap = _model_capsule(tmp_path, must_accelerate=True, eligible="auto",
                         semantic_family="contraction")
    res = _grade_model_capsule(cap, target="probe_target", timeout=1)
    assert res["status"] == "incomplete", res
    assert res["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_the_two_mesh_records_never_share_a_key():
    """Two different claims under one key is how the model record got clobbered by the tile record."""
    import inspect

    from merlin import compile_cli

    src = inspect.getsource(compile_cli.compile_model)
    assert 'out["mesh_tile_verification"] = _mesh_verify(' in src, \
        "the synthesized-tile record must not be written to the model-execution key"
    assert 'out["mesh_execution"] = _mesh_verify(' not in src


# --- the accelerator evidence must be collected by DEFAULT ---------------------------------------
# A third way the capstone came out vacuous, and the quietest: `MERLIN_MESH_VERIFY` defaulted OFF, so
# the tile certification that proves the matmul layers ran ON the mesh was simply never collected. The
# tier ladder then had nothing to record, and the capsule reported a verdict backed by the functional
# lane alone. The cost of collecting it is real -- it is bounded by MERLIN_MODEL_BUDGET_S, not by
# declining to look.

def _mesh_verify_default(capsule: dict, env: str | None) -> bool:
    """The decision under test, evaluated exactly as `_grade_model_capsule_inline` writes it."""
    sem = capsule.get("semantic") or {}
    req = [str(x) for x in ((capsule.get("lanes") or {}).get("require") or [])]
    demands = bool(sem.get("must_accelerate")) or "on_mesh" in req
    e = (env or "").strip().lower()
    return (e in ("1", "true", "yes", "on")) if e else demands


def test_mesh_verification_follows_the_capsules_own_demand():
    must = {"semantic": {"must_accelerate": True}}
    interop = {"semantic": {"must_accelerate": False},
               "lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}}
    neither = {"semantic": {"must_accelerate": False}}

    assert _mesh_verify_default(must, None) is True
    assert _mesh_verify_default(interop, None) is True, (
        "an interop capsule withholds must_accelerate on purpose — host work is the behaviour under "
        "test — but it REQUIRES on_mesh, and that requirement is unverifiable without the evidence")
    assert _mesh_verify_default(neither, None) is False

    # the env var still overrides, in BOTH directions, for a deliberate diagnostic run
    assert _mesh_verify_default(must, "0") is False
    assert _mesh_verify_default(neither, "1") is True


def test_every_capstone_either_demands_the_evidence_or_records_why_it_cannot():
    """The pair invariant. A capstone may withhold the demand -- six do, because the derivation could
    not ground one against the target's declared capabilities and role census -- but withholding must be
    RECORDED, never silent, and it must not buy a pass: a capsule that demands nothing collects no mesh
    verification, so every tier it declared is unexercised and the ladder fails it closed.

    What must never happen is the third state: no demand, no reason, and a verdict anyway.
    """
    from merlin.common.paths import merlin_dir

    roots = [merlin_dir() / "experiments/capsule_bench/harness/full_public_capsules",
             merlin_dir() / "contract/capsules"]
    demanding = withheld = 0
    for root in roots:
        for f in sorted(root.rglob("capsule.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            if c.get("kind") != "model":
                continue
            if _mesh_verify_default(c, None):
                demanding += 1
                continue
            withheld += 1
            reason = ((c.get("semantic") or {}).get("not_asserted_reason") or "").strip()
            assert reason, (
                f"{c.get('name')} ({f}) demands no accelerator evidence and records no reason — "
                f"silence here is indistinguishable from 'this target has no accelerator demand'")
    assert demanding >= 3, f"expected the grounded capstones, found {demanding}"
    assert withheld >= 1, "the withholding path is real and must stay exercised by the corpus"


def test_a_withheld_demand_cannot_buy_a_pass():
    """Withholding is honest, not free: with no demand there is no mesh verification, so no tier the
    capsule declared is exercised, and the ladder refuses to call that a pass."""
    ungrounded = {"semantic": {"must_accelerate": False, "not_asserted_reason": "could not derive"},
                  "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    assert _mesh_verify_default(ungrounded, None) is False
    # no mesh verification -> no tile record -> nothing exercised -> every declared tier unexercised
    exercised: dict[str, str] = {}
    unexercised = [t for t in ungrounded["required_oracle_tiers"] if t not in exercised]
    assert unexercised == ["L0", "L1", "L2", "L3"]
