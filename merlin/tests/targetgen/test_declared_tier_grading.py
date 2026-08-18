"""Grading must ride the tier the CAPSULE declared, and must name what actually happened.

Two defects, both measured on a real (voided) agent run, both regression-tested here.

1. **Tier substitution.** The per-round loop picked the endpoint's *fastest* tier and the materializer,
   when capping to that ceiling stripped every numeric tier, APPENDED the ceiling as the required tier.
   A corpus that declares the cycle-accurate cert tier was therefore graded against a cheaper, additive
   tier it had never declared — and that tier's runner hung, so every capsule failed on a gate the
   capsule never asked for while its declared tier ran fine. The loop tier is now the fastest endpoint
   tier the corpus DECLARES, and capping never substitutes; when no declared tier is reachable the
   harness fails closed and names the declared-vs-reachable sets.

2. **A hung program reported as a missing oracle.** The endpoint raised its "did not halt" verdict as a
   subclass of ``OracleUnavailable``, so a program that RAN and hung was recorded ``unavailable`` and
   surfaced as ``plane: oracle_unavailable`` / "mandatory tier L# did not run". Those are opposite
   instructions ("wait for infra" vs "emit the halt"). It now routes to its own named plane.

Every assertion here is driven by the corpus + endpoint, never by a target literal: the tests read
whichever tiers the descriptors declare rather than asserting a particular target reaches a particular
tier.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.contract import schemas
from merlin.targetgen.contract.materialize import (_cap_required, _cap_tiers, declared_oracle_tiers,
                                                   public_capsules_for)
from merlin.targetgen.program_oracle import OracleUnavailable, ProgramDidNotHalt
from merlin.targetgen.runner_config import RunnerConfig
from merlin.targetgen.target_experiment import load_target_experiment

TARGETS = repo_root() / "merlin/experiments/capsule_bench/targets"
CAPS = repo_root() / "merlin/contract/capsules"


def _te(name: str):
    d = TARGETS / name / "target_experiment.yaml"
    if not d.is_file():
        pytest.skip(f"no descriptor for {name}")
    return load_target_experiment(d)


# ---------------------------------------------------------------------------------------------
# 1. capping never substitutes
# ---------------------------------------------------------------------------------------------
def test_capping_is_a_pure_intersection_and_reports_what_it_dropped():
    kept, unreachable = _cap_required(["L0", "L1", "L3"], set(_cap_tiers("L2")))
    assert kept == ["L0", "L1"], "capping must not invent a tier the capsule never declared"
    assert unreachable == ["L3"], "the dropped DECLARED tier must be reported, not silently lost"


def test_capping_keeps_a_declared_tier_that_is_reachable():
    kept, unreachable = _cap_required(["L0", "L1", "L2", "L3"], set(_cap_tiers("L2")))
    assert kept == ["L0", "L1", "L2"]
    assert unreachable == ["L3"]


# ---------------------------------------------------------------------------------------------
# 2. the loop tier is chosen from what the corpus declares
# ---------------------------------------------------------------------------------------------
def test_loop_tier_is_a_tier_the_corpus_declares():
    """For every target with a descriptor and a resolvable endpoint, the per-round loop tier must be one
    the corpus actually declares — the property whose absence caused the substitution."""
    checked = 0
    for d in sorted(TARGETS.iterdir()):
        if not (d / "target_experiment.yaml").is_file():
            continue
        te = load_target_experiment(d / "target_experiment.yaml")
        try:
            full = CR.oracle_adapters(te.target, te.sim_via)
        except Exception:                       # noqa: BLE001 — endpoint not resolvable in this env
            continue
        declared = declared_oracle_tiers(*te.graded_roots())
        if not full or not declared:
            continue
        loop = CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared)
        checked += 1
        if set(full) & declared:
            assert loop, f"{te.target}: a declared tier is reachable but no loop tier was chosen"
            assert set(loop) <= declared, (
                f"{te.target}: loop tier {sorted(loop)} is NOT declared by its corpus "
                f"{sorted(declared)} — that is the silent substitution")
            assert len(loop) == 1, "the fast loop still rides exactly one tier"
        else:
            assert loop == {}, (
                f"{te.target}: no declared tier is reachable, so the loop must fail closed, not "
                f"substitute one of {sorted(full)}")
    if not checked:
        pytest.skip("no target's endpoint resolved in this environment")


def test_unreachable_declared_tier_fails_closed_and_names_both_sets(tmp_path):
    """When the endpoint reaches none of the declared tiers the loop returns nothing (never a
    substitute), and materialization raises naming declared-vs-reachable."""
    import yaml
    te = _te("gemmini")
    # L5 (firesim) is declared by no capsule here and reached by no endpoint in this environment.
    assert CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers={"L5"}) == {}

    corpus = tmp_path / "isa"
    (corpus / "X0_unreachable").mkdir(parents=True)
    (corpus / "X0_unreachable" / "capsule.yaml").write_text(yaml.safe_dump(
        {"name": "X0_unreachable", "label": "public", "required_oracle_tiers": ["L0", "L5"]}))

    class _Fake:                                 # a descriptor whose corpus declares an unreachable tier
        target, sim_via, capsule_corpus = te.target, te.sim_via, corpus

        def corpus_siblings(self):
            return []

    with pytest.raises(ValueError) as ei:
        public_capsules_for(_Fake())
    msg = str(ei.value)
    assert "declares required oracle tiers" in msg and "reaches" in msg, msg
    assert "L5" in msg, "the refusal must name the unreachable DECLARED tier"
    assert "never declared" in msg, "the refusal must say it will not substitute"


# ---------------------------------------------------------------------------------------------
# 3. a hung program is a verdict, not a missing oracle
# ---------------------------------------------------------------------------------------------
def _atlas_shaped_config(tier: str) -> RunnerConfig:
    """A float/self-hosted-ISA grading config whose single RTL tier is ``tier``. Shape, not identity —
    the target name is a label here, the behavior under test is tier-independent."""
    return RunnerConfig(
        target="atlas", suite="atlas-capsule-bench", dtype="fp8_e4m3",
        fourth_output_name="kernel.S", tier_sim={tier: "endpoint-sim"},
        rtl_tiers=frozenset({tier}), oracle_tiers=(tier,), perf_fields=(), trace_gate=None)


def _stub_front_half(monkeypatch):
    cb = {"tensors": {"Y0": {"role": "output", "base": 0, "shape": [32, 32], "dtype": "bf16"}}}
    monkeypatch.setattr(CR, "run_entrypoints", lambda *a, **k: (object(), cb, "# kernel.S (stub)\n"))
    return cb


def _capsule_declaring(tier: str):
    cap = load_capsule(CAPS / "atlas/isa/AT2_single_tile_matmul", contract="merlin/contract")
    cap = dict(cap)
    cap["required_oracle_tiers"] = ["L0", "L1", tier]
    return cap


def test_did_not_halt_is_its_own_plane_not_oracle_unavailable(tmp_path, monkeypatch):
    """The exact voided condition: the endpoint ran the program and it never halted."""
    _stub_front_half(monkeypatch)
    tier = "L3"

    def hung(cb, llvm_text, workdir, timeout):
        raise ProgramDidNotHalt("target program did not halt within 20000 instructions (functional)")

    res = CR.run_capsule(_capsule_declaring(tier), "unused-package", runs_root=tmp_path,
                         run_id="hung", config=_atlas_shaped_config(tier),
                         oracle_adapters={tier: hung})

    assert res["failure"]["plane"] == CR.DID_NOT_HALT_PLANE
    assert res["failure"]["plane"] != "oracle_unavailable", (
        "a program that RAN and hung must not be reported as an absent oracle")
    assert "did not halt" in res["failure"]["detail"]
    assert "halt/terminate instruction" in res["failure"]["detail"], (
        "the detail must name the fix, not just the symptom")
    # the tier itself records a FAIL (a verdict), never 'unavailable' (an absence).
    assert res["tiers"][tier]["status"] == "fail"
    assert res["status"] == "fail"
    schemas.validate(res, "capsule_result", contract="merlin/contract")


def test_absent_oracle_is_still_reported_as_unavailable(tmp_path, monkeypatch):
    """The distinction must cut both ways: a genuinely ABSENT oracle keeps the unavailable plane."""
    _stub_front_half(monkeypatch)
    tier = "L3"

    def absent(cb, llvm_text, workdir, timeout):
        raise OracleUnavailable("model venv python absent")

    res = CR.run_capsule(_capsule_declaring(tier), "unused-package", runs_root=tmp_path,
                         run_id="absent", config=_atlas_shaped_config(tier),
                         oracle_adapters={tier: absent})

    assert res["tiers"][tier]["status"] == "unavailable"
    assert res["status"] == "incomplete"
    assert res["failure"]["plane"] == "oracle_unavailable"


def test_did_not_halt_subclasses_unavailable_so_handlers_still_fail_closed():
    assert issubclass(ProgramDidNotHalt, OracleUnavailable)


# ---------------------------------------------------------------------------------------------
# 4. no change to a target whose declared tier already IS the fastest reachable one
# ---------------------------------------------------------------------------------------------
def test_declared_aware_selection_is_a_noop_when_the_fastest_tier_is_declared():
    """The regression guard for in-flight runs: where the corpus declares the endpoint's fastest tier,
    declared-aware selection must pick exactly what the legacy 'fastest tier' rule picked."""
    te = _te("gemmini")
    try:
        full = CR.oracle_adapters(te.target, te.sim_via)
    except Exception:                            # noqa: BLE001
        pytest.skip("endpoint not resolvable in this environment")
    if not full:
        pytest.skip("no oracle adapters in this environment")
    declared = declared_oracle_tiers(*te.graded_roots())
    assert min(full) in declared, (
        "precondition: this target's corpus declares its fastest reachable tier")
    legacy = sorted(CR.qa_loop_adapters(te.target, te.sim_via))
    aware = sorted(CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared))
    assert legacy == aware, f"loop tier changed for an in-flight target: {legacy} -> {aware}"


def test_materialized_required_tiers_are_a_subset_of_the_declared_ones():
    """Across every resolvable target: the graded set may only ever DROP declared tiers, never add one."""
    checked = 0
    for d in sorted(TARGETS.iterdir()):
        if not (d / "target_experiment.yaml").is_file():
            continue
        te = load_target_experiment(d / "target_experiment.yaml")
        try:
            dest = public_capsules_for(te)
        except Exception:                        # noqa: BLE001 — endpoint/corpus not resolvable here
            continue
        import yaml
        declared = declared_oracle_tiers(*te.graded_roots())      # the ORIGINAL corpus's declaration
        for cap_yaml in sorted(dest.glob("*/capsule.yaml")):
            doc = yaml.safe_load(cap_yaml.read_text()) or {}
            got = set(doc.get("required_oracle_tiers") or [])
            assert got <= declared, (
                f"{te.target}/{cap_yaml.parent.name}: graded tiers {sorted(got)} include one the corpus "
                f"never declared ({sorted(got - declared)}) — a substitution")
            checked += 1
    if not checked:
        pytest.skip("no target's corpus materialized in this environment")
