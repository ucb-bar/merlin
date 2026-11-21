"""Opt-in Phase 1 roofline and boxed native timing, without a second agent loop.

The operator selects immutable facts, capsule geometry and runtime bytes. The
author can change its compiler, never those inputs. Missing timing inputs retain
useful structural feedback. No structural ratio supplies attainable utilization.
"""

from __future__ import annotations

import contextvars
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from pathlib import Path

from merlin.analysis.perf.bounds import capsule_roofline as R
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.contracts import document_sha256
from merlin.runtime.views import native_simulator as F

# ---- joint_performance -------------------------------------------------------------------------

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
    # These are optional execution budgets, not observation evidence. Resolve
    # their public policy once here; all downstream consumers use this selection.
    if "per_request_seconds" not in value:
        value["per_request_seconds"] = 30
    if "per_observation_seconds" not in value:
        value["per_observation_seconds"] = value["per_request_seconds"]
    for key in ("per_request_seconds", "per_observation_seconds"):
        budget = value[key]
        if type(budget) not in (int, float) or not 0 < budget <= 900:
            raise ValueError("joint performance observation budget must be in (0, 900]")
    return pin, value


def deadline(timeout):
    selection = selected()
    return time.monotonic() + min(timeout, selection[1]["per_request_seconds"]) if selection else 0


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
    per_case = value["per_observation_seconds"]
    return max(0.0, min(per_case, request_deadline - time.monotonic()))


def _runtime(pin, *, facts_pin=None):
    from merlin.development.backend.optimization.diagnostics.spike_runtime import (
        COUNT_OBSERVATION_900S_PROTOCOL,
        freeze_runtime,
    )

    spec = _read(pin)

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
        review_id=spec["review_id"],
        functional_argv=tuple(spec["functional_argv"]),
        observer=Path(spec["observer"]),
        disassembler=Path(spec["disassembler"]),
        model_path=Path(spec["model"]),
        source_manifest=Path(spec["source_manifest"]),
        libraries=tuple(map(Path, spec["libraries"])),
        dependencies=tuple(map(Path, spec["dependencies"])),
        entry_symbol=spec["entry_symbol"],
        completion_symbol=spec["completion_symbol"],
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
        member.get("contractions"),
        machine,
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
    from merlin.development.backend.feedback.campaign.joint_performance_report import project

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
    passed = False
    seed = None
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
        # A carried screen did not compare outputs again (numeric stays "skipped"): the same executable
        # passed that comparison when its verdict was earned, so it is timed like one that just passed.
        carried = isinstance(tier.get("carried"), dict) and tier["carried"].get("carried") is True
        passed = tier.get("status") == "pass" and (numeric.get("status") == "pass" or carried)
        from merlin.development.backend.feedback.performance import idle_measurement as ON

        need = ON.need_seconds(structural)
        budget = min(remaining_s, selection["per_request_seconds"])
        if need is not None and need > budget and request_deadline is not None:
            budget = max(budget, min(need, request_deadline - time.monotonic()))
        if not passed:
            reason = "original complete numerical screen did not pass"
        elif tier.get("engine") != "spike":
            reason = "timing observation only follows original Spike execution"
        elif (seed := _seed_timing(root, epoch, selection, pin, source_capsule, result)) is not None:
            # A seeded run: the seed timed exactly these bytes under this model (.seed_timing).
            stage, reason = "seed_timing", seed["performance"]["reason"]
            model_review = seed["performance"].get("model_review")
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
                reason = (
                    f"observation needs ~{need} s, beyond the selected count runtime's own bound "
                    f"({ceiling} s for its two commands)"
                )
            else:
                stage = "same_elf_observation"
                report, observation = _observe(
                    root,
                    context=context,
                    compiler=compiler,
                    runtime=runtime,
                    result=result,
                    capsule=capsule,
                    remaining_s=budget,
                )
                ON.record(structural, observation)
                reason = observation.get("reason")
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
        from merlin.development.backend.campaign.calibration.activation import request_binding

        request_model = request_binding(environment())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report = None
        # Details and paths are operator-only. Do not expose reference filenames.
        reason = "joint performance unavailable: " + type(error).__name__
        unavailable_stage = stage
        if stage == "same_elf_observation" and isinstance(error, ValueError):
            # These refusals describe the author's own executable or host code (e.g. an owned host
            # completion the analysis cannot follow); without the message the author cannot act on it.
            from merlin.common.access import scrub_host_paths

            unavailable_detail = scrub_host_paths(str(error))[:240]
    row["performance"] = (
        seed["performance"]
        if seed is not None
        else project(
            report,
            observation,
            structural,
            selection_sha256=pin["sha256"],
            reason=reason,
        )
    )
    row["performance"]["capsule_sha256"] = capsule_identity(source_capsule if source_capsule is not None else capsule)
    row["performance"]["staged_capsule_sha256"] = capsule_identity(capsule)
    row["performance"]["unavailability_stage"] = unavailable_stage
    if unavailable_detail:
        row["performance"]["unavailability_detail"] = unavailable_detail
    from merlin.development.backend.feedback.performance.idle_measurement import coverage_limit

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
    if passed:
        # Every passing case carries a calibrated estimate from its emitted command census; a complete
        # timing-model estimate stays preferred and also calibrates it (:mod:`.fast_estimate`).
        from merlin.development.backend.feedback.performance import fast_estimate

        fast_estimate.attach(
            row["performance"], root, structural, facts_sha256=(selection.get("facts") or {}).get("sha256")
        )
    row["performance"]["request_model_binding"] = (
        dict(
            channel=request_model["channel"],
            request_id=request_model["request_id"],
            request_sha256=request_model["request_sha256"],
            selection_sha256=(request_model["selection"] or {}).get("sha256"),
            accepted_record_sha256=(request_model["accepted_record"] or {}).get("sha256"),
            activation_warning=request_model["activation_warning"],
        )
        if request_model
        else None
    )
    row["performance"]["structural_comparison_binding"] = {
        key: row["performance"].get(key)
        for key in (
            "capsule_sha256",
            "facts_sha256",
            "selection_sha256",
            "model_sha256",
            "comparison_domain_sha256",
            "scope_sha256",
        )
    }
    from merlin.development.backend.feedback.campaign.joint_performance_priority import (
        project as search_priority,
    )

    row["performance"]["search_priority"] = search_priority(result, row["performance"])
    if seed is not None:
        if seed.get("compiler_diagnostics") is not None:
            row["compiler_diagnostics"] = seed["compiler_diagnostics"]
        return
    try:
        from merlin.development.backend.feedback.campaign.joint_diagnostics_feedback import (
            attach as attach_diagnostics,
        )

        attach_diagnostics(row, root, report=report, observation=observation, result=result, compiler=compiler)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        row["compiler_diagnostics"] = {
            "status": "UNKNOWN",
            "reason": "diagnostic projection unavailable",
            "promotion_authority": False,
        }
    try:
        from merlin.development.backend.campaign.calibration.service import register

        if source_capsule is not None:
            register(row, root, result=result, compiler=compiler, capsule=source_capsule)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        # Registration cannot change the original numerical/performance verdict.
        row["performance"]["calibration_registration"] = "unavailable"


def _seed_timing(root, epoch, selection, pin, source_capsule, result):
    """A seeded run's carried model cycles for exactly this executable (:mod:`.seed_timing`), else None."""
    from merlin.development.backend.feedback.performance import seed_timing

    run_dir = seed_timing._run_dir()
    if (
        run_dir is None
        or not (run_dir / seed_timing.TIMING_DIR).is_dir()
        or not epoch.get("runtime")
        or result.get("owned_host_native_execution") is not None
        or not isinstance(source_capsule, dict)
    ):
        return None
    runtime = _runtime(epoch["runtime"], facts_pin=selection["facts"])
    key = {
        "capsule_sha256": capsule_identity(source_capsule),
        "selection_sha256": pin["sha256"],
        "model_sha256": F.digest(runtime.model_path),
        "facts_sha256": selection["facts"]["sha256"],
        "timing_epoch": epoch["epoch"],
    }
    return seed_timing.carried(original_executable(root), capsule=str(source_capsule.get("name")), key=key)


def original_executable(root):
    from merlin.development.targetgen.grading.elf_lanes import PACKAGE_ELF_NAME

    root = F.canonical(Path(root))
    elf = F.canonical(root / PACKAGE_ELF_NAME)
    if not elf.is_file():
        raise ValueError("original native producer executable is unavailable")
    return elf


def _observe(root, *, context, compiler, runtime, remaining_s, result=None, capsule=None):
    from merlin.development.backend.optimization.campaign import functional_boundary as PC
    from merlin.development.backend.optimization.campaign import package_sandbox as _merlin_package_sandbox_1
    from merlin.development.backend.optimization.diagnostics import archive_count_observation as OBS
    from merlin.target.registry.descriptor_loading import load_target_experiment

    root = F.canonical(Path(root))
    # The native producer owns this exact slot. Retained execution snapshots
    # legitimately contain the same basename and are never alternative producers.
    # OBS still reopens the direct ELF's original recipe and invocation custody.
    owned_scope = None
    if (result or {}).get("owned_host_native_execution") is not None:
        from merlin.development.backend.optimization.diagnostics.spike_runtime import prepare

        owned_scope = prepare(result, target=context.target, evidence_root=root / "owned_scope_review")
        elf = owned_scope.elf
    else:
        elf = original_executable(root)
    compiler = Path(compiler)
    candidate = hash_tree(compiler)["sha256"]
    from merlin.development.backend.feedback.scalability.declared_bounds import binary_bytes

    # Self-contained ELFs embed their declared stimulus: the fixed bound is the floor.
    elf_bound = binary_bytes(capsule, 64 << 20)
    elf_before = F.pin(elf, max_bytes=elf_bound)
    target = load_target_experiment(context.descriptor)
    if target.target != context.target:
        raise ValueError("joint observation target mismatch")
    policy = _merlin_package_sandbox_1.package_sandbox_policy(target, root, compiler)
    # These self-contained capsule ELFs embed their complete stimulus. The input
    # identity is explicitly ELF-bound, not a fabricated external tensor hash.
    observation = OBS.observe(
        elf,
        runtime=runtime,
        policy=policy,
        directory=root / "joint_timing",
        input_sha256=elf_before["sha256"],
        remaining_s=remaining_s,
        numerical_pass=True,
        **({"owned_scope": owned_scope} if owned_scope is not None else {}),
    )
    observation["input_identity_basis"] = "self-contained numerically checked ELF, including embedded stimulus"
    report = _read(observation["report"]) if observation.get("status") == "observed" else None
    if F.pin(elf, max_bytes=elf_bound) != elf_before or hash_tree(compiler)["sha256"] != candidate:
        raise ValueError("candidate changed during joint observation")
    runtime.verify()
    if report is not None:
        model = json.loads(runtime.model_json)
        if report.get("model_sha256") != F.digest(runtime.model_path):
            raise ValueError("joint report model differs from selected runtime")
        if report.get("domain") != dict(model["domain"], target_model_sha256=F.digest(runtime.model_path)):
            raise ValueError("joint report comparison domain differs from selected runtime")
    return report, observation


# ---- joint_timing_selection --------------------------------------------------------------------
# Select an immutable timing epoch; covered tuned models never fall back.


def choose(selection, capsule_sha256):
    """Exact capsule coverage selects the newest declared tuned epoch.

    Dynamic feature coverage is checked after observation. A mismatch or failed
    observation stays UNKNOWN on the selected epoch; retrying an older or untuned
    model would silently change the performance objective.
    """
    epochs = selection.get("timing_epochs", [])
    if not isinstance(epochs, list) or len(epochs) > 64:
        raise ValueError("joint timing epochs need a bounded explicit roster")
    found = []
    seen = set()
    for row in epochs:
        number = row.get("epoch")
        coverage = row.get("coverage") or {}
        if (
            type(number) is not int
            or number < 1
            or number in seen
            or row.get("mode") != "tuned"
            or not row.get("runtime")
            or not row.get("calibration_evidence")
            or not coverage.get("scope_sha256")
            or not coverage.get("memory_regime_sha256")
            or not isinstance(coverage.get("capsule_sha256"), list)
            or not isinstance(coverage.get("feature_ranges"), dict)
        ):
            raise ValueError("tuned epoch lacks explicit runtime, evidence or coverage")
        seen.add(number)
        if capsule_sha256 in coverage["capsule_sha256"]:
            found.append(row)
    if found:
        return max(found, key=lambda row: row["epoch"])
    return dict(epoch=0, mode="untuned", runtime=selection.get("runtime"), coverage=None)


def within_coverage(epoch, report):
    if epoch["mode"] == "untuned":
        return True
    coverage = epoch["coverage"]
    if (
        report.get("scope_sha256") != coverage["scope_sha256"]
        or report.get("domain", {}).get("memory_regime_sha256") != coverage["memory_regime_sha256"]
    ):
        return False
    counts = report.get("feature_counts") or {}
    for name, interval in coverage["feature_ranges"].items():
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(v) not in (int, float) for v in interval)
            or interval[0] < 0
            or interval[1] < interval[0]
        ):
            raise ValueError("tuned model feature coverage is malformed")
        value = counts.get(name)
        if type(value) not in (int, float) or not interval[0] <= value <= interval[1]:
            return False
    return True


# ---- joint_timing_facts ------------------------------------------------------------------------
# Reopen explicit operator mappings from extracted facts to timing model slots.
#
# Selectors and semantic roles come from the target's operator intake. This reader
# checks arithmetic consistency; it cannot turn unsupported prices or queue-role
# substitutions into physical qualification.


def _at(document, path):
    if not isinstance(path, list) or not 1 <= len(path) <= 32:
        raise ValueError("fact/model selector must be a bounded nonempty path")
    for key in path:
        if type(key) not in (str, int):
            raise ValueError("fact/model selector has an unsupported key")
        document = document[key]
    return document


def verify(alignment, facts, model):
    """Refuse inconsistent direct geometry/credit mappings; never price traffic."""
    if alignment.get("schema") != "merlin.joint_timing_fact_alignment.v1":
        raise ValueError("unsupported timing fact alignment")
    rows = alignment.get("mappings")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("timing fact alignment needs explicit bounded mappings")
    checked = []
    for row in rows:
        selectors = row["fact_paths"]
        if not isinstance(selectors, list) or not 1 <= len(selectors) <= 8:
            raise ValueError("timing mapping has invalid factor roster")
        values = [_at(facts, path) for path in selectors]
        if any(type(v) is not int or not 0 < v < 2**63 for v in values):
            raise ValueError("timing mapping needs positive extracted integer facts")
        expected = Fraction(math.prod(values))
        # An absent conversion leaves the extracted product in its original units.
        # An explicit conversion must be admitted independently.
        if "unit_divisor" in row:
            divisor = row["unit_divisor"]
            if type(divisor) is not int or divisor not in (1, 8):
                raise ValueError("timing mapping only supports identity or bit-to-byte conversion")
            expected /= divisor
        actual = _at(model, row["model_path"])
        if type(actual) not in (int, float) or actual != expected:
            raise ValueError("selected timing model contradicts extracted facts at " + str(row["model_path"]))
        checked.append(dict(model_path=row["model_path"], value=actual, role=row["role"]))
    return dict(
        status="consistent",
        checked=checked,
        timing_qualified=False,
        unsupported=alignment.get("unsupported", []),
        fit_from_measurement=alignment.get("fit_from_measurement", []),
    )


# ---- joint_performance_batch -------------------------------------------------------------------
# Attach a self-check's timing observations, each test case on its own budget, a few at a time.
#
# Observations ran one after another against one shared budget, so a check of many cases returned cycles
# for the first few only. Each case now gets :func:`joint_performance.observation_budget` of its own, and
# when the calling thread holds no CPU slot the observations run concurrently: each leases its own slot
# from the shared pool (as it always did) and the calling thread waits holding none. A thread that already
# holds a slot keeps the sequential order, so no slot is ever held while waiting for another.

MAX_CONCURRENT = 4


def attach_all(jobs, *, context, request_deadline):
    """``joint_performance.attach`` for every job dict (row, root, result, compiler, capsule)."""
    from merlin.common.process import cpu_pool

    def one(job):
        started = time.monotonic()
        attach(
            job["row"],
            job["root"],
            result=job["result"],
            context=context,
            compiler=job["compiler"],
            capsule=job["capsule"],
            remaining_s=observation_budget(request_deadline),
            request_deadline=request_deadline,
        )
        performance = job["row"].get("performance")
        if isinstance(performance, dict):
            performance["observation_seconds"] = round(time.monotonic() - started, 1)

    jobs = list(jobs)
    width = min(MAX_CONCURRENT, cpu_pool.compute_capacity(), len(jobs))
    if width <= 1 or cpu_pool.current_thread_owns_cpu_lease():
        for job in jobs:
            one(job)
        return
    with ThreadPoolExecutor(width, thread_name_prefix="joint-observation") as pool:
        # Each worker carries the caller's lease priority (a context variable threads do not inherit).
        futures = [pool.submit(contextvars.copy_context().run, one, job) for job in jobs]
        for future in futures:
            future.result()
