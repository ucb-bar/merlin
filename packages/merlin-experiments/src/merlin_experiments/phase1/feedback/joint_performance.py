"""Opt-in Phase 1 roofline and boxed native timing, without a second agent loop.

The operator selects immutable facts, capsule geometry and runtime bytes. The
author can change its compiler, never those inputs. Missing timing inputs retain
useful structural feedback. No structural ratio supplies attainable utilization.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.perf import capsule_roofline as R
from merlin.runtime import _native_files as F

from ...phase2.contracts import document_sha256

ENV = "MERLIN_JOINT_PERFORMANCE_SELECTION"
SCHEMA = "merlin.phase1.joint_performance_selection.v1"
_ENVIRONMENT = ContextVar("joint_performance_request_environment", default=None)


def environment():
    """A copy of this request's context, or its immutable process environment."""
    selected_environment = _ENVIRONMENT.get()
    return dict(os.environ if selected_environment is None else selected_environment)


@contextmanager
def environment_scope(value):
    """Bind a worker-thread callback without mutating process-global environment."""
    token = _ENVIRONMENT.set(dict(value))
    try:
        yield
    finally:
        _ENVIRONMENT.reset(token)


def capsule_identity(capsule):
    """Native discovery stamps a movable directory; bind the declaration bytes."""
    return document_sha256({k: v for k, v in (capsule or {}).items() if k != "__dir__"})



def selected_capsule(selection, capsule):
    """Resolve only the native execution-ceiling overlay to a pinned source.

    Required tiers, numerics, source files and every other declaration field
    remain part of the exact source identity. This grants no tier credit.
    """
    value = {k: v for k, v in (capsule or {}).items() if k != "__dir__"}
    member = selection.get("members", {}).get(value.get("name"))
    if not member or capsule_identity(value) == member.get("capsule_sha256"):
        return value
    if value.get("oracle_tier_ceiling") in {"L0", "L1", "L2", "L3", "L4"}:
        source = {k: v for k, v in value.items() if k != "oracle_tier_ceiling"}
        if capsule_identity(source) == member.get("capsule_sha256"):
            return source
        # Native public materialization also removes owner-only capture metadata.
        # Reopen the unchanged private policy source, and prove the entire
        # materialization projection before using its exact selected identity.
        for original in _policy_sources(capsule):
            if capsule_identity(original) != member.get("capsule_sha256"):
                continue
            projected = {k: v for k, v in original.items() if k != "capture_tool"}
            projected["oracle_tier_ceiling"] = value["oracle_tier_ceiling"]
            if projected == value:
                return original
    raise ValueError("joint roofline geometry belongs to different capsule bytes")


def _policy_sources(capsule):
    """Bounded lookup in the native owner's sibling policy snapshot only."""
    import yaml
    directory = capsule.get("__dir__")
    name = capsule.get("name")
    if not directory or not isinstance(name, str) or Path(name).name != name:
        return
    path = Path(directory)
    if path.name != name or path.parent.name != "public":
        return
    policy = path.parent.parent / "policy"
    if not policy.is_dir():
        return
    policy = F.canonical(policy)
    for group in F.children(policy, limit=16):
        source = policy / group / name / "capsule.yaml"
        if not source.is_file():
            continue
        before = F.pin(source, max_bytes=1 << 20)
        document = yaml.safe_load(source.read_text())
        if F.pin(source, max_bytes=1 << 20) != before:
            raise ValueError("native policy source changed during timing resolution")
        if isinstance(document, dict):
            yield document

def _read(pin):
    path = Path(pin["path"])
    before = F.pin(path, max_bytes=16 << 20)
    if before["sha256"] != pin["sha256"]:
        raise ValueError("joint performance input changed")
    value = json.loads(path.read_bytes())
    if F.pin(path, max_bytes=16 << 20) != before:
        raise ValueError("joint performance input changed while reading")
    return value


def selected():
    """Read only the explicitly pinned operator selection; no fallback discovery."""
    raw = environment().get(ENV)
    if not raw:
        return None
    pin = json.loads(raw)
    value = _read(pin)
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported joint performance selection")
    for key in ("per_request_seconds", "per_observation_seconds"):
        budget = value.get(key, 30)
        if type(budget) not in (int, float) or not 0 < budget <= 900:
            raise ValueError("joint performance observation budget must be in (0, 900]")
    return pin, value


def deadline(timeout):
    selection = selected()
    return time.monotonic() + min(timeout, selection[1].get("per_request_seconds", 30)) if selection else 0


def observation_budget(request_deadline):
    """Seconds ONE test case's observation may use: its own budget, never past the request's deadline.

    A single shared budget let the first few large test cases of a check spend it, after which every
    later case reported "budget exhausted" with no cycles: measured, 444 of 962 estimates returned to
    one agent were empty. ``per_observation_seconds`` (default ``per_request_seconds``) now bounds each
    case separately; the request's own deadline still bounds the whole check.
    """
    choice = selected()
    if choice is None:
        return 0
    value = choice[1]
    per_case = value.get("per_observation_seconds", value.get("per_request_seconds", 30))
    return max(0.0, min(per_case, request_deadline - time.monotonic()))


def _runtime(pin, *, facts_pin=None):
    from ...phase2.archive_spike_runtime import COUNT_OBSERVATION_900S_PROTOCOL, freeze_runtime

    spec = _read(pin)
    from .joint_timing_facts import verify

    alignment = _read(spec["facts_alignment"])
    if facts_pin is not None and facts_pin["sha256"] != alignment["facts"]["sha256"]:
        raise ValueError("roofline and timing runtime selected different RTL facts")
    facts = _read(alignment["facts"])
    model = _read({"path": spec["model"], "sha256": spec["model_sha256"]})
    verify(alignment, facts, model)
    for additional in spec.get("additional_fact_alignments", []):
        extra = _read(additional)
        for source in extra.get("source_pins", []):
            if F.pin(Path(source["path"]), max_bytes=256 << 20) != source:
                raise ValueError("additional timing fact source changed")
        verify(extra, _read(extra["facts"]), model)
    # Every pricing update gets a new selection and runtime identity. No inherited
    # historical coefficient digest is implicitly approved by this adapter.
    runtime = freeze_runtime(
        review_id=spec["review_id"], functional_argv=tuple(spec["functional_argv"]),
        observer=Path(spec["observer"]), disassembler=Path(spec["disassembler"]),
        model_path=Path(spec["model"]), source_manifest=Path(spec["source_manifest"]),
        libraries=tuple(map(Path, spec["libraries"])), dependencies=tuple(map(Path, spec["dependencies"])),
        entry_symbol=spec["entry_symbol"], completion_symbol=spec["completion_symbol"],
        timing_selection=spec["timing_selection"],
        observation_resource_protocol=COUNT_OBSERVATION_900S_PROTOCOL,
    )
    expected = spec["model_sha256"]
    if F.digest(runtime.model_path) != expected:
        raise ValueError("joint runtime pricing differs from pinned model")
    return runtime


def structural_feedback(selection, capsule, *, target):
    """Reopen source-specific geometry and derive arithmetic capacity from facts."""
    capsule = selected_capsule(selection, capsule)
    facts = _read(selection["facts"])
    machine = R.machine_bounds(target, facts=facts)
    member = selection.get("members", {}).get(capsule.get("name"), {})
    if member and member.get("capsule_sha256") != capsule_identity(capsule):
        raise ValueError("joint roofline geometry belongs to different capsule bytes")
    return R.partial_roofline(
        member.get("contractions"), machine,
        logical_read_bytes=member.get("logical_read_bytes"),
        logical_write_bytes=member.get("logical_write_bytes"),
    )


def attach(row, root, *, result, context, compiler, capsule, remaining_s, request_deadline=None):
    """Attach a public projection after the original complete numerical screen.

    No gSIM is launched. Exact native ELF producer custody, one-ELF namespace,
    plain/annotated console equivalence and bounded shutdown are inherited from
    the Phase 2 observer. Cleanup uncertainty propagates to the owning job.

    ``remaining_s`` is this case's floor budget. With ``request_deadline`` (the
    request's own monotonic deadline) a case whose predicted need exceeds the floor
    may use up to that need within the request (:mod:`.observation_need`); a case
    whose need does not fit is not started and says what it needs, and one beyond
    the selected count runtime's own bound says that instead.
    """
    from .joint_performance_report import project
    from .joint_timing_selection import choose, within_coverage

    choice = selected()
    if choice is None:
        return
    pin, selection = choice
    epoch = choose(selection, capsule_identity(capsule))
    report = None
    observation = {}
    structural = None
    model_review = None
    source_capsule = None
    request_model = None
    unavailable_stage = None
    stage = "source_binding"
    reason = "timing runtime not selected"
    unavailable_detail = None
    need = deferred = None
    try:
        source_capsule = selected_capsule(selection, capsule)
        epoch = choose(selection, capsule_identity(source_capsule))
        if result.get("owned_host_native_execution") is not None and epoch["mode"] == "untuned":
            # An owned call has a different scope identity. Never silently reuse
            # a kernel model or fall back from a covered tuned epoch.
            epoch = dict(epoch, runtime=selection.get("owned_model_runtime"))
        stage = "structural_roofline"
        structural = structural_feedback(selection, source_capsule, target=context.target)
        stage = "numerical_screen"
        numeric = result.get("numeric") or {}
        tier = (result.get("tiers") or {}).get("L2") or {}
        passed = numeric.get("status") == "pass" and tier.get("status") == "pass"
        from . import observation_need as ON

        need = ON.need_seconds(structural)
        budget = min(remaining_s, selection.get("per_request_seconds", 30))
        if need is not None and need > budget and request_deadline is not None:
            budget = max(budget, min(need, request_deadline - time.monotonic()))
        if not passed:
            reason = "original complete numerical screen did not pass"
        elif tier.get("engine") != "spike":
            reason = "timing observation only follows original Spike execution"
        elif budget < 1:
            # Nothing of this request's observation budget was left when the case's turn came.
            reason, deferred = "observation not started: shared budget exhausted", need
        elif need is not None and need > budget:
            reason, deferred = ON.deferred_reason(need, budget), need
        elif epoch.get("runtime"):
            if epoch["mode"] == "tuned":
                _read(epoch["calibration_evidence"])
            stage = "runtime_freeze"
            runtime = _runtime(epoch["runtime"], facts_pin=selection["facts"])
            model_review = runtime.review_id
            ceiling = ON.protocol_ceiling(runtime)
            if need is not None and ceiling is not None and need > ceiling:
                reason = (f"observation needs ~{need} s, beyond the selected count runtime's own bound "
                          f"({ceiling} s for its two commands)")
            else:
                stage = "same_elf_observation"
                report, observation = _observe(
                    root, context=context, compiler=compiler, runtime=runtime, result=result, capsule=capsule,
                    remaining_s=budget,
                )
                ON.record(structural, observation)
                reason = observation.get("reason", "native timing is provisional pending independent qualification")
                if observation.get("provider_reason"):
                    # The provider's own stated refusal (host paths already scrubbed by the observer).
                    unavailable_detail = str(observation["provider_reason"])[:240]
            stage = "coverage_check"
            if report is not None and not within_coverage(epoch, report):
                report = None
                reason = "selected tuned model outside observed coverage; no untuned fallback"
        stage = "selection_recheck"
        if selected() != choice:
            raise ValueError("joint selection changed during observation")
        stage = "request_model_binding"
        from ..joint_model_activation import request_binding
        request_model = request_binding(environment())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report = None
        # Details and paths are operator-only. Do not expose reference filenames.
        reason = "joint performance unavailable: " + type(error).__name__
        unavailable_stage = stage
        if stage == "same_elf_observation" and isinstance(error, ValueError):
            # These refusals describe the author's own executable or host code (e.g. an owned host
            # completion the analysis cannot follow); without the message the author cannot act on it.
            from merlin.common.path_scrub import scrub_host_paths

            unavailable_detail = scrub_host_paths(str(error))[:240]
    row["performance"] = project(
        report, observation, structural, selection_sha256=pin["sha256"], reason=reason,
    )
    row["performance"]["capsule_sha256"] = capsule_identity(source_capsule if source_capsule is not None else capsule)
    row["performance"]["staged_capsule_sha256"] = capsule_identity(capsule)
    row["performance"]["unavailability_stage"] = unavailable_stage
    if unavailable_detail:
        row["performance"]["unavailability_detail"] = unavailable_detail
    from .observation_need import coverage_limit

    row["performance"]["observation_need_s"] = need
    row["performance"]["observation_deferred"] = deferred is not None
    limit = coverage_limit(row["performance"])
    if limit:
        # The timing model's own bound, not the compiler's or the budget's: name it so it is visible.
        row["performance"]["model_coverage_limit"] = limit
        row["performance"]["reason"] = "model coverage limit: " + "; ".join(limit)
    row["performance"]["timing_mode"] = epoch["mode"]
    row["performance"]["timing_epoch"] = epoch["epoch"]
    row["performance"]["model_review"] = model_review
    row["performance"]["untuned_fallback_permitted"] = False
    row["performance"]["facts_sha256"] = selection["facts"]["sha256"]
    row["performance"]["request_model_binding"] = (dict(
        channel=request_model["channel"], request_id=request_model["request_id"],
        request_sha256=request_model["request_sha256"],
        selection_sha256=(request_model["selection"] or {}).get("sha256"),
        accepted_record_sha256=(request_model["accepted_record"] or {}).get("sha256"),
        activation_warning=request_model["activation_warning"],
    ) if request_model else None)
    row["performance"]["structural_comparison_binding"] = {
        key: row["performance"].get(key) for key in (
            "capsule_sha256", "facts_sha256", "selection_sha256", "model_sha256",
            "comparison_domain_sha256", "scope_sha256")}
    from .joint_performance_priority import project as search_priority
    row["performance"]["search_priority"] = search_priority(result, row["performance"])
    try:
        from .joint_diagnostics_feedback import attach as attach_diagnostics
        attach_diagnostics(row, root, report=report, observation=observation, result=result, compiler=compiler)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        row["compiler_diagnostics"] = {"status": "UNKNOWN", "reason": "diagnostic projection unavailable",
                                       "promotion_authority": False}
    try:
        from ..joint_calibration_service import register
        if source_capsule is not None:
            register(row, root, result=result, compiler=compiler, capsule=source_capsule)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        # Registration cannot change the original numerical/performance verdict.
        row["performance"]["calibration_registration"] = "unavailable"



def original_executable(root):
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    root = F.canonical(Path(root))
    elf = F.canonical(root / PACKAGE_ELF_NAME)
    if not elf.is_file():
        raise ValueError("original native producer executable is unavailable")
    return elf

def _observe(root, *, context, compiler, runtime, remaining_s, result=None, capsule=None):
    from merlin.targetgen.target_experiment import load_target_experiment

    from ...phase2 import archive_count_observation as OBS
    from ...phase2 import campaign as PC

    root = F.canonical(Path(root))
    # The native producer owns this exact slot. Retained execution snapshots
    # legitimately contain the same basename and are never alternative producers.
    # OBS still reopens the direct ELF's original recipe and invocation custody.
    owned_scope = None
    if (result or {}).get("owned_host_native_execution") is not None:
        from ...phase2.owned_model_observation import prepare
        owned_scope = prepare(result, target=context.target, evidence_root=root / "owned_scope_review")
        elf = owned_scope.elf
    else:
        elf = original_executable(root)
    compiler = Path(compiler)
    candidate = hash_tree(compiler)["sha256"]
    from .declared_bounds import binary_bytes

    # Self-contained ELFs embed their declared stimulus: the fixed bound is the floor.
    elf_bound = binary_bytes(capsule, 64 << 20)
    elf_before = F.pin(elf, max_bytes=elf_bound)
    target = load_target_experiment(context.descriptor)
    if target.target != context.target:
        raise ValueError("joint observation target mismatch")
    policy = PC.package_sandbox_policy(target, root, compiler)
    # These self-contained capsule ELFs embed their complete stimulus. The input
    # identity is explicitly ELF-bound, not a fabricated external tensor hash.
    observation = OBS.observe(
        elf, runtime=runtime, policy=policy, directory=root / "joint_timing",
        input_sha256=elf_before["sha256"], remaining_s=remaining_s, numerical_pass=True,
        **({"owned_scope": owned_scope} if owned_scope is not None else {}),
    )
    observation["input_identity_basis"] = "self-contained numerically checked ELF, including embedded stimulus"
    report = _read(observation["report"]) if observation.get("status") == "observed" else None
    if (F.pin(elf, max_bytes=elf_bound) != elf_before
            or hash_tree(compiler)["sha256"] != candidate):
        raise ValueError("candidate changed during joint observation")
    runtime.verify()
    if report is not None:
        model = json.loads(runtime.model_json)
        if report.get("model_sha256") != F.digest(runtime.model_path):
            raise ValueError("joint report model differs from selected runtime")
        if report.get("domain") != dict(model["domain"], target_model_sha256=F.digest(runtime.model_path)):
            raise ValueError("joint report comparison domain differs from selected runtime")
    return report, observation
