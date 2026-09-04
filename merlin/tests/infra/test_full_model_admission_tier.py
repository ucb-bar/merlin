"""A whole-model admission must demand the tier the model path can actually EMIT.

Two defects made an Arm-4 gemmini performance campaign unreachable, and both are pinned here.

1.  The full-model admission gate forced ``required_oracle_tiers = ["L2", "L3"]`` and required BOTH to
    report ``pass``.  A model capsule never enters the tier ladder: its tier block is synthesised by
    ``capsule_runner._model_tier_map``, which emits exactly ONE execution tier -- the last declared
    tier the target's capability manifest counts as RTL.  gemmini's RTL tiers are ``L3/L4/L5``, so
    ``status("L2")`` read ``None`` on a FLAWLESS model run, ``passed`` was ``False``, and the campaign
    raised before a single perf cell executed.  ``perf_reporting`` asserted the same impossible pair.

2.  The performance prompt advertised the agent-input manifest by its HOST path under the stage
    directory, while the sandbox binds that snapshot at ``AGENT_CORPUS_MOUNT``.  A round was measured
    in which the agent read the allowed-path list, found the declared mount absent, and correctly
    refused: exit 0, zero broker calls, zero files changed, 91.6 s of a round burned.

Every test here asserts the POSITIVE case fires as well as the refusal, and each fix carries a
vacuity check proving the pre-fix predicate really was unsatisfiable rather than merely redundant.
"""
from __future__ import annotations

import importlib.util
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_runner as CR

SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"

# The target whose ladder exposed the defect.  This is a per-target test edge, not shared library
# code: the point of the case is precisely that gemmini's RTL tiers begin ABOVE L2.
TARGET = "gemmini"


def _load(name: str, alias: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# The derivation itself
# ---------------------------------------------------------------------------------------------

def test_the_forced_l2_predicate_was_unsatisfiable_for_this_targets_ladder() -> None:
    """VACUITY GUARD for everything below: L2 is not an RTL tier here, so the old gate could not pass.

    If a future manifest change made L2 RTL for this target, the tests that follow would still pass
    while no longer pinning anything, so the impossibility is asserted directly.
    """
    rtl = CR.rtl_tiers_of(TARGET)
    assert rtl, "the target's capability manifest must declare RTL tiers for this case to mean anything"
    assert "L2" not in rtl, f"L2 became an RTL tier for {TARGET}; this whole case no longer applies"

    perfect = {"matmul_layers_on_mesh": 4, "matmul_layers_host_fallback": 0}
    tiers = CR._model_tier_map(["L2", "L3"], TARGET, perfect)
    assert "L2" not in tiers, "a model run emitted L2, so the old admission gate was satisfiable"
    assert {name: t.status for name, t in tiers.items()} == {"L3": "pass"}


# The derivation itself (which tier, on which target) is pinned in
# merlin/tests/targetgen/test_model_citable_tier.py, against the emitter it must agree with.


# ---------------------------------------------------------------------------------------------
# The runner's admission gate
# ---------------------------------------------------------------------------------------------

def _admission_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tiers: dict,
                       declared=("L0", "L1", "L2", "L3")):
    runner = _load("run_perf_bench", "_admission_gate_runner")
    package = tmp_path / "candidate"
    package.mkdir()
    source = tmp_path / "frozen-public" / "M3"
    source.mkdir(parents=True)
    contract = tmp_path / "frozen-contract"
    contract.mkdir()
    workspace = tmp_path / "admission"
    workspace.mkdir()
    PC = runner.PC
    sentinel = PC.FullModelSentinel(
        "M3", source, {"kind": "model", "required_oracle_tiers": list(declared)},
        "a" * 64, 2, 180_000)
    observed: dict = {}

    monkeypatch.setattr(runner.PC, "package_sandbox_policy", lambda *args: object())
    monkeypatch.setattr(runner.PC, "boxed_entrypoints", lambda _p: contextlib.nullcontext())
    monkeypatch.setattr(runner.CR, "load_capsule",
                        lambda source, contract: {"name": "M3", "kind": "model",
                                                  "required_oracle_tiers": list(declared)})

    def fake_grade(capsule, candidate, **kwargs):
        observed.update(capsule=capsule, kwargs=kwargs)
        return {"status": "pass", "numeric": {"status": "pass"}, "tiers": tiers}

    monkeypatch.setattr(runner.CR, "run_capsule", fake_grade)
    monkeypatch.setattr(runner, "_verify_frozen_contract", lambda *_a: None)
    target = SimpleNamespace(target=TARGET)
    return runner, observed, (package, sentinel, workspace, 90, target, contract, "b" * 64)


def test_a_flawless_model_run_is_ADMITTED_at_the_targets_citable_rtl_tier(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE POSITIVE CASE. The tier block below is exactly what a perfect model run emits."""
    runner, observed, args = _admission_harness(
        tmp_path, monkeypatch,
        tiers={"L0": {"status": "skipped", "not_applicable": True},
               "L1": {"status": "skipped", "not_applicable": True},
               "L3": {"status": "pass", "reason": "4 matmul layer(s) executed on the accelerator"}})
    evidence = runner.run_full_model_admission(*args)

    assert evidence["passed"] is True, "a flawless model run was still refused"
    assert evidence["required_tiers"] == ["L3"]
    assert evidence["tier_status"] == {"L3": "pass"}
    assert observed["capsule"]["required_oracle_tiers"] == ["L3"]
    assert evidence["tier_derivation"]["cert_tier"] == "L3"
    assert evidence["tier_derivation"]["declared"] == ["L0", "L1", "L2", "L3"]
    assert "L3" in evidence["tier_derivation"]["rtl_tiers"]
    assert evidence["cycles_recorded"] is False, "admission must never become a timing claim"

    # THE TEST ACTUALLY FIRES: the retired predicate would have refused this very evidence.
    assert evidence["tier_status"].get("L2") is None
    assert not (evidence["tier_status"].get("L2") == "pass"
                and evidence["tier_status"].get("L3") == "pass")


@pytest.mark.parametrize("tiers,why", [
    ({"L3": {"status": "skipped",
             "reason": "no matmul layer executed on the accelerator"}}, "no layer on the mesh"),
    ({"L3": {"status": "fail",
             "reason": "3 matmul layer(s) fell back to the host kernel"}}, "host fallback"),
    ({"L3": {"status": "unavailable",
             "reason": "no mesh execution counters were reported"}}, "counters absent"),
    ({}, "no tier block at all"),
    ({"L0": {"status": "skipped"}, "L1": {"status": "skipped"}}, "only not-applicable tiers"),
])
def test_a_model_run_that_never_reached_the_rtl_tier_is_still_REFUSED(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiers: dict, why: str) -> None:
    """Loosening the gate must not make it credulous: only an executed RTL tier admits a campaign."""
    runner, _observed, args = _admission_harness(tmp_path, monkeypatch, tiers=tiers)
    evidence = runner.run_full_model_admission(*args)
    assert evidence["passed"] is not True, f"admitted a model that {why}"


def test_a_correct_model_on_the_wrong_lane_is_refused_even_with_a_pass_grade(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured shape of the open compiler defect: numerics pass, nothing runs on the mesh."""
    runner, _observed, args = _admission_harness(
        tmp_path, monkeypatch,
        tiers={"L3": {"status": "skipped",
                      "reason": "no matmul layer executed on the accelerator"}})
    evidence = runner.run_full_model_admission(*args)
    assert evidence["grade_status"] == "pass" and evidence["numeric_status"] == "pass"
    assert evidence["passed"] is False


def test_a_sentinel_declaring_no_rtl_tier_is_refused_before_any_measurement(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _observed, args = _admission_harness(
        tmp_path, monkeypatch, tiers={"L2": {"status": "pass"}}, declared=("L0", "L1", "L2"))
    with pytest.raises(runner.PC.CampaignGateError, match="counts as RTL"):
        runner.run_full_model_admission(*args)


# ---------------------------------------------------------------------------------------------
# The reporting gate re-derives the same tier from recorded facts
# ---------------------------------------------------------------------------------------------

def _derivation(**over) -> dict:
    row = {"declared": ["L0", "L1", "L2", "L3"], "rtl_tiers": ["L3", "L4", "L5"],
           "cert_tier": "L3", "rule": "the last declared tier the manifest counts as RTL"}
    row.update(over)
    return row


def _staged(declared=("L0", "L1", "L2", "L3")):
    return SimpleNamespace(e2e_sentinel={"capsule": "M3", "required_tiers": list(declared)})


def test_reporting_gate_re_derives_the_admission_tier_from_the_recorded_inputs() -> None:
    PR = _load("perf_reporting", "_admission_gate_reporting")
    assert PR._full_model_cert_tier({"tier_derivation": _derivation()}, _staged()) == "L3"


@pytest.mark.parametrize("sentinel,staged,match", [
    ({}, _staged(), "records no tier derivation"),
    ({"tier_derivation": _derivation(cert_tier="L2")}, _staged(), "derive"),
    ({"tier_derivation": _derivation(declared=["L0", "L1", "L2"])},
     _staged(("L0", "L1", "L2")), "no tier the measured target counts as RTL"),
    ({"tier_derivation": _derivation(rtl_tiers=[])}, _staged(), "counts as RTL"),
    ({"tier_derivation": _derivation()}, _staged(("L2", "L3")), "sealed staging handoff"),
    ({"tier_derivation": _derivation()}, SimpleNamespace(e2e_sentinel={}), "sealed staging handoff"),
])
def test_reporting_gate_refuses_an_admission_tier_it_cannot_reproduce(
        sentinel: dict, staged: object, match: str) -> None:
    PR = _load("perf_reporting", "_admission_gate_reporting")
    with pytest.raises(PR.ReportingGateError, match=match):
        PR._full_model_cert_tier(sentinel, staged)


# ---------------------------------------------------------------------------------------------
# The sentinel is DECLARED, then graded -- it is never inherited as already-passed
# ---------------------------------------------------------------------------------------------

def test_no_public_functional_grade_contains_a_whole_model_capsule() -> None:
    """WHY the sentinel cannot be selected from the functional pass list, asserted against the corpus.

    ``select_full_model_sentinel`` used to be documented as picking the smallest ALREADY-PASSED public
    model, and a reader's natural repair is to make it consult ``public_score["per_capsule"]``.  That
    repair would refuse every campaign: the public functional corpus is kernel capsules, so the
    intersection is empty.  The sentinel is graded by the admission gate instead, and this test exists
    so the tempting "fix" fails loudly with the reason rather than silently zeroing the candidate set.
    """
    corpus = repo_root() / "merlin/contract/capsules"
    public_kinds = set()
    model_dirs = []
    for descriptor in sorted(corpus.glob("*/*/capsule.yaml")):
        import yaml
        doc = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
        if doc.get("label") != "public":
            continue
        public_kinds.add(str(doc.get("kind")))
        if doc.get("kind") == "model":
            model_dirs.append(descriptor.parent.name)

    assert public_kinds, "vacuous: no public capsules were read at all"
    assert model_dirs, "vacuous: the corpus declares no public whole-model capsule to select"
    # The models live in their own category directory, apart from the kernel capsules a functional
    # public grade actually covers -- which is exactly why a pass-list intersection comes back empty.
    assert "model" in public_kinds


# ---------------------------------------------------------------------------------------------
# Defect 1 -- the prompt names paths as the SANDBOX sees them
# ---------------------------------------------------------------------------------------------

def _prompt_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ST = _load("perf_agent_stage", "_admission_gate_stage")
    PP = ST.PP

    stage = tmp_path / "stage"
    agent_root = stage / "_agent_inputs"
    agent_root.mkdir(parents=True)
    manifest = agent_root / "agent_input_manifest.json"
    manifest.write_bytes(b"{}")
    agent_inputs = ST.AgentInputSnapshot(agent_root, manifest, "a" * 64, "b" * 64, 1, 2)

    host_package = tmp_path / "host-package"
    host = PP.HostLaneGrant(TARGET, "run42", str(host_package), "c" * 64,
                            str(host_package / "manifest.yaml"), "seam")
    sentinel = PP.E2ESentinel("M3", str(tmp_path / "repo" / "M3"), str(tmp_path / "snap" / "M3"),
                              "d" * 64, ("on_mesh", "scalar_rvv_lane"), ("L0", "L1", "L2", "L3"))
    grant = ST.FrozenGrant("capsules", Path("/repo/capsules"), tmp_path / "snap" / "capsules",
                           "e" * 64)
    frozen_functional = SimpleNamespace(marker_sha256="f" * 64, content_sha256="0" * 64,
                                        grants=(grant,))
    frozen_corpus = SimpleNamespace(capsules=(), manifest_sha256="1" * 64, capsules_sha256="2" * 64)
    functional = SimpleNamespace(run_id="fr-1", digest="3" * 64, public_capsules=20,
                                 hidden_capsules=5)

    monkeypatch.setattr(ST, "select_e2e_sentinel", lambda *a, **k: sentinel)
    monkeypatch.setattr(ST, "_host_lane_grant", lambda *a, **k: host)
    monkeypatch.setattr(ST, "_preflight_cohort", lambda _claim: ("r000", "r001", "r002"))
    monkeypatch.setattr(ST, "_family_declarations", lambda *a, **k: ())
    monkeypatch.setattr(ST.PC, "expected_perf_cells", lambda *a, **k: ())

    inputs = ST.prepare_prompt_inputs(
        functional, frozen_functional, frozen_corpus, agent_inputs,
        SimpleNamespace(target=TARGET), (),
        formal_claim={"declaration": {"replicates": {"n": 3}}},
        smoke_replicates=1, wall_budget_seconds=3600, rounds=2,
        round_timeout_seconds=600, max_tool_calls=50, tool_timeout_seconds=60)
    return ST, inputs, agent_inputs


def test_prompt_advertises_the_agent_manifest_at_its_IN_SANDBOX_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ST, inputs, agent_inputs = _prompt_inputs(tmp_path, monkeypatch)
    mount = str(ST.AGENT_CORPUS_MOUNT / "agent_input_manifest.json")

    assert mount in inputs.allowed_paths, "the prompt still does not name the mounted manifest"
    # THE DEFECT: the host stage path is not a path anything inside the sandbox can open, and an
    # agent that checks its declared mounts refuses the round rather than editing blind.
    assert str(agent_inputs.manifest_path) not in inputs.allowed_paths
    assert str(agent_inputs.manifest_path) != mount, "vacuous: host and mount paths coincide here"


def test_no_advertised_path_leaks_a_host_stage_location(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same defect class for EVERY path the prompt advertises, not just the one that bit us.

    A path is legitimate when the sandbox can open it: an absolute mount destination, a bind whose
    destination equals its source (the host-lane package is bound that way on purpose), or a
    workspace-relative name.  Nothing under the stage directory qualifies -- it is never bound.
    """
    _ST, inputs, agent_inputs = _prompt_inputs(tmp_path, monkeypatch)
    stage_root = agent_inputs.root.parent
    leaked = [path for path in inputs.allowed_paths
              if path.startswith(str(stage_root) + "/") or path == str(stage_root)]
    assert not leaked, f"prompt advertises host stage paths the sandbox never binds: {leaked}"
    assert inputs.allowed_paths, "vacuous: the prompt advertised no paths at all"
