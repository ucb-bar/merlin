"""A dead host and a failing model must not produce the same artifact.

Two silent no-ops are held shut here, and both presented as something else entirely:

* **Presence is not operability.** The performance stage gated its launch on ``shutil.which("bwrap")``
  + ``os.access(X_OK)``. On a host with ``kernel.apparmor_restrict_unprivileged_userns=1`` and a
  non-setuid ``bwrap`` that gate PASSES and every sandbox invocation then dies at ``setting up uid map:
  Permission denied``. Each agent round runs its whole process behind that wrapper, so each exits
  nonzero and the harness reports the outcome it has a name for -- "no seed reached" -- which reads as
  the model having failed. The launcher's ``preflight()`` asked nothing about the sandbox at all, so
  ``--dry-run`` said GO on exactly such a host.

* **A required path rule that matches nothing is vacuously satisfied.** ``answer_surfaces()`` can only
  mask existing paths, and ``coverage_gap()`` iterates only surfaces that exist. Explicit prior-backend
  names and exemptions therefore need a separate dropped-declaration check. Optional logical oracle
  identities are different: the access registry masks all physical installed copies, and an absent
  OOT implementation remains a recorded migration diagnostic rather than a fatal stale path.

Every test here is a MUTATION test: it breaks the property and asserts the guard fires, because a guard
that has never been observed to fire is indistinguishable from one that cannot.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import checkpoint_admission as ADMISSION
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT
from merlin_experiments.phase2 import telemetry as TEL
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.access import MODULE_ACCESS, module_locations, unresolved_modules
from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.targetgen.sandbox import build_sandbox
from merlin.targetgen.sandbox import preflight as PF

#: The answer-surface MODULE, not the same-named function the package re-exports over it. ``from
#: merlin.targetgen.sandbox import answer_surfaces`` binds the function, so the mutations below would
#: silently set attributes on a function object and assert nothing.
AS = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")

_GEMMINI_DESCRIPTOR = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"


def _probe(status: str, reason: str) -> PF.SandboxProbe:
    return PF.SandboxProbe(status=status, reason=reason, binary="/usr/bin/bwrap", returncode=1)


_OK_PROBE = PF.SandboxProbe(status=PF.SANDBOX_OK, reason="probe_command_ran", binary="/usr/bin/bwrap", returncode=0)
_DEAD_PROBE = _probe(PF.SANDBOX_INOPERABLE, "userns_uid_map_denied")


def _fake_bwrap(tmp_path: Path, *, stderr: str, code: int) -> Path:
    """An executable file that behaves like the real binary does on this host: present, and refusing."""
    binary = tmp_path / "bwrap"
    binary.write_text(f'#!/bin/sh\nprintf %s "{stderr}" >&2\nexit {code}\n', encoding="utf-8")
    binary.chmod(0o755)
    return binary


# --------------------------------------------------------------- presence vs operability, no mocking
def test_a_present_but_refusing_binary_passes_presence_and_fails_operability(tmp_path: Path) -> None:
    """The exact divergence that let a campaign launch on a host where no round could run.

    No monkeypatching: a real executable file that exits nonzero with the real message. The presence
    check the stage used to rely on accepts it; the probe names it inoperable and says why.
    """
    binary = _fake_bwrap(tmp_path, stderr="bwrap: setting up uid map: Permission denied", code=1)

    # The question about the FILE -- which is what the retired gate asked -- is satisfied.
    assert binary.resolve().is_file()
    assert os.access(binary, os.X_OK)

    # The question about the EXECUTION is not.
    probe = PF.probe_sandbox(binary, use_cache=False)
    assert probe.status == PF.SANDBOX_INOPERABLE
    assert probe.reason == "userns_uid_map_denied"
    assert probe.usable is False
    assert "uid map" in probe.describe()


def test_an_undetermined_probe_never_reads_as_a_pass(tmp_path: Path) -> None:
    """UNKNOWN is not usable. A sandbox not shown to work has not been shown to work."""
    unknown = _probe(PF.SANDBOX_UNKNOWN, "probe_timed_out")
    assert unknown.usable is False
    assert unknown.as_record()["usable"] is False
    with pytest.raises(PF.SandboxUnavailable):
        raise PF.SandboxUnavailable(unknown, "context")


def test_the_stage_launch_gate_refuses_a_dead_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: kill the sandbox and the stage refuses to start rather than blaming the agent."""
    monkeypatch.setattr(PF, "probe_sandbox", lambda *_a, **_k: _DEAD_PROBE)

    with pytest.raises(PF.SandboxUnavailable) as refusal:
        PF.require_working_sandbox("/bin/sh", context="agent stage")
    message = str(refusal.value)
    assert "agent stage" in message
    assert "userns_uid_map_denied" in message

    # Control: the SAME call succeeds when the probe says the sandbox is real, and hands back the probe
    # the run artifact records -- so the refusal above is about the sandbox, not about the argument.
    monkeypatch.setattr(PF, "probe_sandbox", lambda *_a, **_k: _OK_PROBE)
    probe = PF.require_working_sandbox("/bin/sh", context="agent stage")
    assert probe.as_record()["status"] == PF.SANDBOX_OK


# ------------------------------------------------------------------- the sealed record must say so
@pytest.mark.parametrize(
    "sandbox",
    [
        pytest.param({"dropped_declarations": []}, id="no probe recorded at all"),
        pytest.param(
            {"preflight": {"status": "inoperable", "usable": False}, "dropped_declarations": []},
            id="probe recorded and it refused",
        ),
        pytest.param(
            {"preflight": {"status": "unknown", "usable": False}, "dropped_declarations": []},
            id="probe could not be determined",
        ),
        pytest.param(
            {"preflight": {"status": "ok", "usable": True}, "dropped_declarations": [{"origin": "oracle"}]},
            id="sandbox ran but a deny rule masked nothing",
        ),
    ],
)
def test_a_record_without_evidence_of_real_containment_is_refused(sandbox: dict) -> None:
    """A policy digest proves an argv was CONSTRUCTED. Only the probe proves one could run."""
    with pytest.raises(StageGateError):
        RECORD.verify_sandbox_containment_evidence(sandbox)


def test_a_record_with_a_usable_probe_and_no_dropped_rules_is_accepted() -> None:
    RECORD.verify_sandbox_containment_evidence({"preflight": _OK_PROBE.as_record(), "dropped_declarations": []})


def test_a_record_predating_the_containment_fields_is_read_but_never_sealed() -> None:
    """The one allowance, and its exact boundary.

    Sealed schema-v3 records predate both fields. Refusing them in the pure validator would reject
    evidence this stage already produced rather than stop a bad run, so the validator's lenient form
    accepts a record that carries NEITHER field. It does not accept one that carries a field SAYING
    containment was absent -- that is a claim, not a gap -- and the strict form, which the sealing path
    uses, accepts neither, so nothing written from here on can omit them.
    """
    legacy = {"outer_codex_control_plane": {}, "inner_execution_plane": {}}
    RECORD.verify_sandbox_containment_evidence(legacy, require_present=False)
    with pytest.raises(StageGateError):
        RECORD.verify_sandbox_containment_evidence(legacy)
    with pytest.raises(StageGateError):
        RECORD.verify_sandbox_containment_evidence(
            {**legacy, "preflight": _DEAD_PROBE.as_record()}, require_present=False
        )
    with pytest.raises(StageGateError):
        RECORD.verify_sandbox_containment_evidence(
            {**legacy, "dropped_declarations": [{"origin": "oracle"}]}, require_present=False
        )


# ------------------------------------------------------------------ an inert deny rule is reported
def test_a_required_path_that_resolves_to_nothing_is_reported() -> None:
    """MUTATION: a nonexistent declared backend is named as a dropped rule."""
    te = _gemmini_experiment()
    missing = "a_package_nobody_minted_for_this_test"
    changed = replace(te, prior_backends=(*te.prior_backends, missing))
    new_drops = [d for d in AS.dropped_declarations(changed) if d.declared == missing]
    assert len(new_drops) == 1
    assert new_drops[0].origin == "prior_backend"
    assert new_drops[0].reason == AS.DROP_PATH_ABSENT
    assert new_drops[0].as_record()["reason"] == AS.DROP_PATH_ABSENT


def test_the_coverage_guard_alone_cannot_see_a_dropped_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY THE DROP REPORT HAD TO BE ADDED AT ALL, asserted rather than argued.

    The mask set and the coverage guard both agree the mask is complete while a declared rule is masking
    nothing -- because the guard can only iterate the surfaces that survived the existence filter. So an
    empty coverage gap is not evidence that the declared rules are in force, and any caller making a
    fairness claim has to check both. If this test ever fails because the gap became non-empty, the guard
    grew the ability to see drops and this file should say so.
    """
    missing = "a_package_nobody_minted_for_this_test"
    monkeypatch.setattr(AS, "prior_backend_exemptions", lambda _te: (missing,))
    sandbox = build_sandbox(_GEMMINI_DESCRIPTOR, tmp_path)
    assert [s.path for s in sandbox.coverage_gap()] == []
    assert any(d.declared == missing for d in sandbox.dropped_declarations())
    record = sandbox.containment_record()

    assert record["coverage_gap"] == []
    assert any(
        d["declared"] == missing and d["origin"] == "prior_backend_exemption" for d in record["dropped_declarations"]
    )


def test_installed_oracle_and_grader_copies_are_found_without_legacy_path_assumptions() -> None:
    """Moved private modules retain a logical identity and each physical copy is maskable."""
    root = repo_root()
    te = _gemmini_experiment()
    masked = {surface.path for surface in AS.answer_surfaces(te)}
    for identity, origin in (
        ("merlin.targetgen.program_oracle", "oracle"),
        ("merlin.targetgen.capsule_runner", "grader"),
    ):
        entry = next(item for item in MODULE_ACCESS if item.identity == identity)
        assert entry.origin == origin
        locations = module_locations(root, entry)
        assert locations, f"no installed copy of {identity}"
        assert set(locations) <= masked


def test_evicted_optional_oracle_identity_is_not_a_required_legacy_path() -> None:
    """Muon stays denied by identity, even if its OOT implementation is absent here."""
    root = repo_root()
    entry = next(item for item in MODULE_ACCESS if item.identity == "merlin.targetgen.muon_oracles")
    assert entry.origin == "oracle"
    assert entry.identity in AS.declared_oracle_modules()
    if not module_locations(root, entry):
        assert entry in unresolved_modules(root)
    assert all(d.declared != entry.identity for d in AS.dropped_declarations(_gemmini_experiment()))


def test_a_failed_eviction_sweep_is_recorded_as_unknown_not_as_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: break the plugin registry and the sweep says so instead of reporting zero routes.

    The sweep used to swallow every exception, so an unrelated import error anywhere in the registry's
    reach made it return an empty list -- indistinguishable from a checkout with no evicted routes.
    """
    import merlin.runtime.backends.base as backends_base

    def _explode(_key: str = "backend"):
        raise ImportError("registry unavailable")

    monkeypatch.setattr(backends_base, "_oot_plugin_modules", _explode)
    paths, failure = AS._evicted_oracle_modules_result()
    assert paths == []
    assert failure is not None and "ImportError" in failure

    dropped = AS.dropped_declarations(_gemmini_experiment())
    registry_drops = [d for d in dropped if d.origin == "evicted_oracle"]
    assert len(registry_drops) == 1
    assert registry_drops[0].reason.startswith(AS.DROP_DISCOVERY_FAILED)


# ------------------------------------------------ the whole codegen-package dir, not four of forty
def _gemmini_experiment():
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(_GEMMINI_DESCRIPTOR)


def test_every_package_under_the_targets_own_dir_is_withheld_not_just_the_named_ones() -> None:
    """The margin that was open: an enumeration covers what it enumerates and nothing minted since.

    Measured before this changed: the gemmini descriptor named 4 of 40 packages, and the hand-authored
    ``hand_v0`` -- as concentrated an answer as this target owns -- was covered by no surface at all.
    """
    te = _gemmini_experiment()
    package_root = artifacts_dir() / "targets" / te.target
    if not package_root.is_dir():
        pytest.skip("no codegen packages minted for this target in this checkout")
    surfaces = AS.answer_surfaces(te)
    withheld = {s.path for s in surfaces if s.origin == "prior_backend"}
    assert package_root in withheld, "the target's own codegen-package dir is not withheld wholesale"

    packages = sorted(p for p in package_root.iterdir() if p.is_dir())
    unlisted = [p for p in packages if p.name not in te.prior_backends]
    assert unlisted, "this checkout cannot demonstrate a newly minted package beyond the descriptor list"
    uncovered = [p for p in unlisted if p not in withheld and not any(a in withheld for a in p.parents)]
    assert uncovered == []


def test_only_this_targets_package_dir_is_withheld_so_a_cross_target_baseline_survives() -> None:
    """Withholding must not reach another target's packages: they are legitimately granted baselines
    (the gemmini descriptor grants ``out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8``) and they
    are not an answer to this target's question."""
    te = _gemmini_experiment()
    withheld = {s.path for s in AS.answer_surfaces(te) if s.origin == "prior_backend"}
    for path in withheld:
        assert (artifacts_dir() / "targets" / te.target) in (path, *path.parents)
    assert (artifacts_dir() / "targets") not in withheld


def test_a_way_into_the_withheld_package_dir_exists_only_where_the_descriptor_declares_one() -> None:
    """The exemption is a property of the SURFACE, so widening containment is a reviewed descriptor edit
    rather than a bundle grant nobody reads."""
    te = _gemmini_experiment()
    package_root = artifacts_dir() / "targets" / te.target
    if not package_root.is_dir():
        pytest.skip("no codegen packages minted for this target in this checkout")
    surface = next(s for s in AS.answer_surfaces(te) if s.path == package_root)
    assert surface.grantable == AS.prior_backend_exemptions(te)
    assert surface.grantable == (), "gemmini declares no exemption; a non-empty one needs review"


def test_an_exemption_naming_a_package_that_does_not_exist_is_a_dropped_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: an exemption list cannot rot into a silent no-op either."""
    te = _gemmini_experiment()
    monkeypatch.setattr(AS, "prior_backend_exemptions", lambda _te: ("a_package_nobody_minted",))
    dropped = AS.dropped_declarations(te)
    assert [(d.origin, d.declared) for d in dropped if d.origin == "prior_backend_exemption"] == [
        ("prior_backend_exemption", "a_package_nobody_minted")
    ]


def test_an_unreadable_descriptor_grants_no_exemption() -> None:
    """FAIL CLOSED: a descriptor that cannot be read withholds more, never less."""
    assert AS.prior_backend_exemptions(SimpleNamespace(path=Path("/nonexistent/target.yaml"))) == ()
    assert AS.prior_backend_exemptions(SimpleNamespace()) == ()


# -------------------------------------------------------------------- --dry-run on a dead host
def _orchestrator_config(orch, tmp_path: Path):
    price_table = tmp_path / "prices.yaml"
    price_table.write_text("gpt-model: [1, 2, 0.1, 1]\n", encoding="utf-8")
    price_table.chmod(0o444)
    source = HOLDOUT.HoldoutSourceContext(
        tmp_path, tmp_path / "catalog.yaml", tmp_path / "core", tmp_path / "experiments", tmp_path / "namespace"
    )
    context = orch.ExecutionContext(
        source_root=tmp_path,
        contract_root=tmp_path / "contract",
        functional_runs_root=tmp_path / "functional-runs",
        stage_root=tmp_path / "stage",
        measurement_root=tmp_path / "measurements",
        holdout_sources=source,
        chia_wrapper=tmp_path / "chia-wrapper",
        invocation=("merlin-experiments", "phase2", "run"),
        suite="test-suite",
    )
    return orch.Config(
        context=context,
        experiment_id="exp",
        root=tmp_path / "experiment",
        functional_run_id="functional",
        functional_submission_sha256="a" * 64,
        descriptor=tmp_path / "target.yaml",
        rtl_facts=tmp_path / "rtl.json",
        perf_profile=tmp_path / "perf.yaml",
        gsim_certificate=tmp_path / "certificate.json",
        gsim_certificate_sha256="b" * 64,
        model="gpt-model",
        effort="high",
        wall_budget_seconds=60,
        rounds=2,
        round_timeout_seconds=30,
        max_tool_calls=5,
        tool_timeout_seconds=10,
        smoke_replicates=1,
        holdout_count=4,
        measurement_timeout=90,
        gsim_max_cycles=9000,
        functional_gsim_certificate=tmp_path / "functional-certificate.json",
        functional_gsim_certificate_sha256="c" * 64,
        telemetry_price_table=price_table,
        chia_python=tmp_path / "chia-python",
    )


def _mock_everything_but_the_sandbox(orch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy every OTHER preflight dependency, so the only thing under test is the host condition."""
    binary = tmp_path / "pinned-gsim"
    binary.write_bytes(b"exact pinned gsim")
    pins = {name: {"path": str(binary), "sha256": orch._sha_file(binary)} for name in orch.GATE.REQUIRED_PINS}
    certificate = SimpleNamespace(target="gemmini", sha256="b" * 64, pins=pins)
    monkeypatch.setattr(orch, "load_target_experiment", lambda *_a, **_k: SimpleNamespace(target="gemmini"))
    monkeypatch.setattr(
        FC, "functional_grade_cohort_from_run", lambda *_a, **_k: FC.FunctionalGradeCohort((), (), 1, 1)
    )
    monkeypatch.setattr(FC, "declined_names", lambda _f: ())
    monkeypatch.setattr(
        orch.FI,
        "inspect_stage_functional_run",
        lambda *_a, **_k: SimpleNamespace(run_id="functional", digest="a" * 64),
    )
    monkeypatch.setattr(orch.GATE, "load_certificate", lambda *_a, **_k: certificate)
    monkeypatch.setattr(orch, "_verify_functional_certificate", lambda *_a: {"public_descriptors": 1})
    monkeypatch.setattr(orch, "_verify_functional_certificate_provenance", lambda *_a: {"declaration_sha256": "7" * 64})
    monkeypatch.setattr(
        orch, "_functional_qualification_descriptor", lambda *_a: (tmp_path / "frozen.yaml", {"sha256": "8" * 64})
    )
    monkeypatch.setattr(orch, "_verify_tuning_certificate", lambda *_a: {"members": 1})
    monkeypatch.setattr(orch, "_require_same_gsim_build", lambda *_a, **_k: None)
    monkeypatch.setattr(HOLDOUT, "derive_domain", lambda *_a, **_k: {"target": "gemmini"})
    monkeypatch.setattr(orch.HOLDOUT, "verify_rtl_facts_provenance", lambda *_a, **_k: {"replay_sha256": "d" * 64})
    monkeypatch.setattr(
        TEL,
        "prepare",
        lambda **_k: {
            "required": True,
            "model_resolution": {"requested_model": "gpt-model", "resolved_model": "gpt-model"},
            "sources": {},
        },
    )
    monkeypatch.setattr(TEL, "treatment_identity", lambda _p: {"resolved_model": "gpt-model"})
    monkeypatch.setattr(orch, "_chia_canary", lambda *_a, **_k: {"available": True})
    monkeypatch.setattr(orch, "dropped_declarations", lambda _te: [])


def test_dry_run_reports_no_go_on_a_host_where_the_sandbox_cannot_be_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE MUTATION THIS WHOLE FILE EXISTS FOR.

    The predeclaration is the last read-only checkpoint before a multi-day campaign commits. It used to
    ask nothing about the sandbox, so it answered GO on a host where every authoring round would die
    inside a wrapper that never starts -- and the campaign would then report the model as having
    produced nothing. Same config, two host conditions, two verdicts.
    """
    orch = ADMISSION
    _mock_everything_but_the_sandbox(orch, tmp_path, monkeypatch)
    config = _orchestrator_config(orch, tmp_path)

    monkeypatch.setattr(orch.SANDBOX_PREFLIGHT, "probe_sandbox", lambda *_a, **_k: _DEAD_PROBE)
    dead = orch.preflight(config, heldout_certificate_provider_available=True)
    assert dead["status"] == "NO_GO"
    assert any("cannot be built on this host" in blocker for blocker in dead["blockers"])
    assert dead["sandbox_preflight"]["status"] == PF.SANDBOX_INOPERABLE
    assert dead["sandbox_preflight"]["usable"] is False

    monkeypatch.setattr(orch.SANDBOX_PREFLIGHT, "probe_sandbox", lambda *_a, **_k: _OK_PROBE)
    alive = orch.preflight(config, heldout_certificate_provider_available=True)
    assert alive["status"] == "GO" and alive["blockers"] == []
    assert alive["sandbox_preflight"]["usable"] is True
    # The verdict is only meaningful together with the condition it was issued under, so the condition
    # travels inside the hashed declaration rather than beside it.
    assert dead["declaration_sha256"] != alive["declaration_sha256"]


def test_dry_run_reports_no_go_when_a_declared_deny_rule_masks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mis-declared answer surface does not weaken a result, it voids it -- so it blocks the launch."""
    orch = ADMISSION
    _mock_everything_but_the_sandbox(orch, tmp_path, monkeypatch)
    monkeypatch.setattr(orch.SANDBOX_PREFLIGHT, "probe_sandbox", lambda *_a, **_k: _OK_PROBE)
    config = _orchestrator_config(orch, tmp_path)

    assert orch.preflight(config, heldout_certificate_provider_available=True)["status"] == "GO"

    monkeypatch.setattr(
        orch,
        "dropped_declarations",
        lambda _te: [AS.DroppedDeclaration("oracle", "merlin/gone.py", AS.DROP_PATH_ABSENT)],
    )
    blocked = orch.preflight(config, heldout_certificate_provider_available=True)
    assert blocked["status"] == "NO_GO"
    assert any("masked nothing" in blocker for blocker in blocked["blockers"])
    assert blocked["dropped_answer_surface_declarations"] == [
        {"origin": "oracle", "declared": "merlin/gone.py", "reason": AS.DROP_PATH_ABSENT}
    ]


def test_replace_keeps_the_dataclass_contract_used_by_preflight() -> None:
    """``preflight`` rebuilds the cohort with ``dataclasses.replace``; keep that a real dataclass."""
    cohort = FC.FunctionalGradeCohort((), (), 1, 1)
    assert replace(cohort, declined=()) is not cohort


def test_the_operator_entry_point_answers_with_its_exit_status(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ops hand-off: fixing an inoperable host is somebody else's action, and they need to confirm
    it without writing a snippet. Exit 0 means usable, 1 means not, and the reason is printed."""
    monkeypatch.setattr(PF, "probe_sandbox", lambda *_a, **_k: _DEAD_PROBE)
    assert PF._main() == 1
    assert "userns_uid_map_denied" in capsys.readouterr().out

    monkeypatch.setattr(PF, "probe_sandbox", lambda *_a, **_k: _OK_PROBE)
    assert PF._main() == 0
