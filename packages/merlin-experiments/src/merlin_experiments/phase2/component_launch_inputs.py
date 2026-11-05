"""Assemble component launch data only from live independent experiment owners.

The declaration selects no compiler, observer, backend or calibration authority.
Every input path must identify the fresh compiler and measured development cohort
already held by issued owners. Assembly never runs an author or mints qualification.
"""

from __future__ import annotations

from pathlib import Path

from merlin_experiments.phase1.component_origin import FreshCompilerOrigin

from . import contracts as C
from .component_execution import ComponentIndependentExecution
from .component_launch import ComponentLaunchInputs, ComponentReadinessProbe
from .component_measurement_qualification import IndependentMeasurementQualification
from .component_providers import (
    build_independent_component_analytical_provider,
    build_independent_component_cca_provider,
)
from .component_runtime_authority import IndependentComponentRuntime, require_independent_runtime
from .component_variants import ComponentVariantSnapshot
from .component_workflow import ComponentOnlyPolicy

_FIELDS = frozenset(
    {
        "schema",
        "candidate",
        "view",
        "corpus",
        "descriptor",
        "source_root",
        "contract_root",
        "stage_root",
        "qualification_root",
        "edit_authority_root",
        "edit_contract",
        "runtime",
        "control_runtime",
        "readiness",
        "codex_binary",
        "codex_destination",
        "auth_source",
        "price_table",
        "analytical",
    }
)
_ANALYTICAL_FIELDS = frozenset(
    {
        "calibration_adapter",
        "qualification",
        "objective",
        "max_workers",
        "memory_per_worker_bytes",
        "engine_slots",
        "output",
        "lease_path",
    }
)
_RTL_FIELDS = frozenset(
    {
        "certificate_path",
        "certificate_sha256",
        "rtl_facts_path",
        "rtl_facts_sha256",
        "work_root",
    }
)


def read_component_launch_declaration(configuration: Path) -> dict:
    """Decode data only; never resolve providers, grade a compiler or create output."""
    document = C.mapping_file(Path(configuration))
    if not (
        (set(document) == _FIELDS and document["schema"] == "merlin.component_launch_inputs.v1")
        or (set(document) == _FIELDS | {"rtl"} and document["schema"] == "merlin.component_launch_inputs.v2")
    ):
        raise C.StageGateError("component launch configuration must use a complete closed v1 or v2 schema")
    return document


def _path(value, *, field):
    if not isinstance(value, str) or not value or "\0" in value:
        raise C.StageGateError("component launch needs an explicit canonical path: " + field)
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path.resolve() != path
        or any(member.is_symlink() for member in (path, *path.parents))
    ):
        raise C.StageGateError("component launch refuses indirect or relative paths: " + field)
    return path


def _same_path(document, field, expected):
    path = _path(document[field], field=field)
    if path != expected:
        raise C.StageGateError("component launch selects another independently admitted input: " + field)
    return path


def _owners(phase1_origin, measurement_support):
    if type(phase1_origin) is not FreshCompilerOrigin or type(measurement_support) is not IndependentComponentRuntime:
        raise C.StageGateError(
            "from-scratch component launch requires live independent Phase 1 origin and measurement authority"
        )
    runtime = require_independent_runtime(
        measurement_support,
        required_roles=("grade", "stage_verifier", "feature_provider", "cca_provider", "rtl_executor"),
        target_descriptor=measurement_support.target_descriptor,
    )
    measured = runtime.qualification
    if type(measured) is not IndependentMeasurementQualification:
        raise C.StageGateError("component launch requires independently qualified complete measurement roles")
    measured.verify()
    admission = measured.baseline_admission
    admission.verify()
    qualification = admission.qualification
    if qualification.compiler_origin is not phase1_origin or qualification.compiler_lineage is not None:
        raise C.StageGateError("component launch baseline differs from the exact fresh Phase 1 origin")
    phase1_origin.verify(candidate=admission.baseline)
    measured.verify_component_binding(admission)
    if (
        measured.functional_runtime is not qualification.runtime_authority
        or measured.functional_runtime is not phase1_origin.inputs.execution_support
        or runtime.hardware_intake is not phase1_origin.inputs.hardware
    ):
        raise C.StageGateError("component launch functional, measurement and fresh hardware owners disagree")
    variants = measured.variants
    if (
        not isinstance(variants, tuple)
        or not variants
        or any(type(row) is not ComponentVariantSnapshot for row in variants)
    ):
        raise C.StageGateError("component launch lacks independently issued compiler variants")
    execution = variants[0].execution
    if type(execution) is not ComponentIndependentExecution:
        raise C.StageGateError("component launch lacks independently issued candidate execution authority")
    for variant in variants:
        variant.verify()
        if variant.execution is not execution:
            raise C.StageGateError("component launch variants belong to different candidate edit owners")
    execution.verify()
    if (
        execution.baseline_admission is not admission
        or execution.independent_runtime is not measured.functional_runtime
        or execution.scope != measured.scope
        or execution.view != qualification.view
        or execution.runtime != qualification.runtime
        or execution.contract_root != qualification.contract_root
    ):
        raise C.StageGateError("component launch execution differs from its fresh baseline or measurement scope")
    qualification.verify(candidate=execution.candidate)
    return runtime, measured, admission, execution


def _grant_records(grants):
    return [{"source": str(row.source), "destination": row.destination, "sha256": row.sha256} for row in grants]


def _tools(document, origin, execution):
    original = origin.inputs
    if (
        execution.runtime != original.runtime
        or document["runtime"] != _grant_records(execution.runtime)
        or document["control_runtime"] != _grant_records(original.control_runtime)
    ):
        raise C.StageGateError("component launch tool grants differ from the exact fresh authoring closure")
    for field in ("codex_binary", "auth_source"):
        _same_path(document, field, getattr(original, field))
    if document["codex_destination"] != original.codex_destination:
        raise C.StageGateError("component launch control executable destination changed")
    for grant in (*execution.runtime, *original.control_runtime):
        grant.verify()
    rows = document["readiness"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise C.StageGateError("component launch needs the exact five live tool readiness probes")
    probes, destinations = [], {row.destination for row in execution.runtime}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"capability", "command", "stdout_sha256"}:
            raise C.StageGateError("component readiness probe must use the closed declaration")
        if (
            not isinstance(row["capability"], str)
            or not isinstance(row["stdout_sha256"], str)
            or not isinstance(row["command"], list)
        ):
            raise C.StageGateError("component readiness command must use explicit argv")
        probe = ComponentReadinessProbe(row["capability"], tuple(row["command"]), row["stdout_sha256"])
        probe.verify()
        if probe.command[0] not in destinations:
            raise C.StageGateError("component readiness executable is outside the independently admitted tool grants")
        probes.append(probe)
    if {row.capability for row in probes} != {"compiler", "linker", "simulator", "isa", "cca"}:
        raise C.StageGateError("component launch readiness omitted a granted capability")
    return original.control_runtime, tuple(probes)


def _stage(document, origin, measured, execution):
    stage = _path(document["stage_root"], field="stage_root")
    if stage.exists():
        raise C.StageGateError("component launch requires a fresh private stage root")
    qualification = execution.baseline_admission.qualification
    private_roots = (
        origin.inputs.output,
        measured.receipt.parent,
        execution.output,
        execution.edit_authority.output,
        qualification.receipt.parent,
    )
    public_roots = (
        execution.candidate,
        execution.view.root,
        execution.baseline_admission.baseline,
        execution.baseline_admission.corpus.root,
        qualification.corpus_root,
        qualification.source_root,
        execution.contract_root,
    )
    if any(stage.is_relative_to(root) or root.is_relative_to(stage) for root in (*private_roots, *public_roots)):
        raise C.StageGateError("component launch private stage overlaps an admitted input or private evidence owner")
    price = _path(document["price_table"], field="price_table")
    if not price.is_file() or any(
        price.is_relative_to(root) for root in (execution.candidate, execution.view.root, stage)
    ):
        raise C.StageGateError("component launch price table must remain an explicit private host input")
    return stage, price


def _analytical(document, runtime, measured, admission, stage):
    selected = document["analytical"]
    if not isinstance(selected, dict) or set(selected) != _ANALYTICAL_FIELDS:
        raise C.StageGateError("component analytical selection must use the complete closed declaration")
    objective = selected["objective"]
    if objective not in ("cold", "warm"):
        raise C.StageGateError("component launch requires an explicit qualified cold or warm objective")
    reports = [path for regime, path, _digest in measured.screen_reports if regime == objective]
    if len(reports) != 1:
        raise C.StageGateError("component launch objective has no exact independently held screen")
    adapter = _same_path(selected, "calibration_adapter", measured.calibration_adapter)
    report = _same_path(selected, "qualification", reports[0])
    output = _same_path(selected, "output", stage / "analytical")
    lease = _same_path(selected, "lease_path", stage / "worker-leases.json")
    for field in ("max_workers", "memory_per_worker_bytes", "engine_slots"):
        if type(selected[field]) is not int or selected[field] < 1:
            raise C.StageGateError("component analytical resource budgets must be positive integers")
    measured.verify_feedback_binding(
        baseline_admission=admission,
        scope=measured.scope,
        calibration_adapter=adapter,
        qualification=report,
        objective=objective,
    )
    return build_independent_component_analytical_provider(
        independent_runtime=runtime,
        baseline_admission=admission,
        dependencies={},
        calibration_adapter=adapter,
        scope=measured.scope,
        qualification=report,
        output=output,
        lease_path=lease,
        objective=objective,
        max_workers=selected["max_workers"],
        memory_per_worker_bytes=selected["memory_per_worker_bytes"],
        engine_slots=selected["engine_slots"],
    )


def load_component_launch_inputs(
    configuration: Path,
    *,
    phase1_origin=None,
    measurement_support=None,
) -> ComponentLaunchInputs:
    """Assemble already issued owners; never grade a compiler or run an author.

    A correctness-only runtime, arbitrary compiler tree, serialized qualification,
    reference backend and imported callback cannot supply experimental authority.
    Live physical roles must have closed the independent measurement controls.
    """
    configuration = _path(str(configuration), field="configuration")
    digest = C.sha256_file(configuration)
    document = read_component_launch_declaration(configuration)
    runtime, measured, admission, execution = _owners(phase1_origin, measurement_support)
    qualification, authority = admission.qualification, execution.edit_authority
    expected = {
        "candidate": execution.candidate,
        "view": execution.view.root,
        "corpus": admission.corpus.root,
        "descriptor": admission.target_descriptor,
        "source_root": qualification.source_root,
        "contract_root": execution.contract_root,
        "qualification_root": qualification.receipt.parent,
        "edit_authority_root": authority.output,
        "edit_contract": authority.output / "compiler_edit_authority.json",
    }
    for field, path in expected.items():
        _same_path(document, field, path)
    control_runtime, readiness = _tools(document, phase1_origin, execution)
    stage, price = _stage(document, phase1_origin, measured, execution)
    analytical = _analytical(document, runtime, measured, admission, stage)
    cca = build_independent_component_cca_provider(independent_runtime=runtime, baseline_admission=admission)
    target = phase1_origin.inputs.target_experiment
    if target.path != admission.target_descriptor:
        raise C.StageGateError("component launch target differs from fresh authoring descriptor ownership")
    rtl = None
    if document["schema"] == "merlin.component_launch_inputs.v2":
        from .component_providers import build_independent_component_rtl_provider

        selected = document["rtl"]
        if not isinstance(selected, dict) or set(selected) != _RTL_FIELDS:
            raise C.StageGateError("component RTL selection requires its complete closed declaration")
        rtl = build_independent_component_rtl_provider(
            independent_runtime=runtime,
            baseline_admission=admission,
            target_experiment=target,
            certificate_path=_path(selected["certificate_path"], field="certificate_path"),
            certificate_sha256=selected["certificate_sha256"],
            rtl_facts_path=_path(selected["rtl_facts_path"], field="rtl_facts_path"),
            rtl_facts_sha256=selected["rtl_facts_sha256"],
            work_root=_same_path(selected, "work_root", stage / "rtl-feedback"),
        )
    policy = ComponentOnlyPolicy(
        candidate=execution.candidate,
        target_experiment=target,
        receipt_path=stage / "receipts" / "private.jsonl",
        component_corpus=admission.corpus,
        baseline_admission=admission,
        independent_runtime=runtime,
        component_analytical=analytical,
        component_cca=cca,
        component_rtl=rtl,
    )
    inputs = ComponentLaunchInputs(
        execution.view,
        execution.candidate,
        policy,
        authority,
        qualification,
        execution.runtime,
        control_runtime,
        readiness,
        phase1_origin.inputs.codex_binary,
        phase1_origin.inputs.codex_destination,
        phase1_origin.inputs.auth_source,
        stage,
        price,
    )
    inputs.verify(require_baseline=True)
    if C.sha256_file(configuration) != digest:
        raise C.StageGateError("component launch declaration changed during independent owner assembly")
    return inputs
