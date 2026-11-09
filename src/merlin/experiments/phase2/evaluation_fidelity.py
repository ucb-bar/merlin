"""Explicit program scope and CPU engine provenance beside legacy probe policy.

An engine name, build receipt or output-equivalence certificate supplies no
physical timing authority. This owner delegates public numerical execution to
the existing independently issued ordinary evaluator. Full-model support is
not inferred from a component qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes
from merlin.common.invocation_record import verify as verify_invocation
from merlin.common.strict_json import loads

from .contracts import StageGateError, document_sha256, sha256_file

SCHEMA = "merlin.program_evaluation_fidelity.v1"
REGISTRY_SCHEMA = "merlin.program_access_registry.v1"
CPU_ENGINES = ("spike", "verilator", "gsim", "firesim_verilator_metasim", "native_fir")
SCOPES = ("component_program", "whole_model")
PHASES = ("public_development", "protected_final")


def _pin(path, digest):
    path = Path(path)
    if (
        not is_sha256(digest)
        or not path.is_absolute()
        or path.resolve() != path
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or sha256_file(path) != digest
    ):
        raise StageGateError("program evaluation selected source/configuration bytes changed")
    return path


@dataclass(frozen=True)
class ProgramAccessRegistry:
    """Host-selected immutable public/protected membership; no grading authority.

    Both sets are explicit before authoring. Public requests cannot supply a
    visibility label to override this selection. Protected execution remains
    with the original post-freeze final evaluator.
    """

    path: Path
    sha256: str

    def verify(self):
        path = _pin(self.path, self.sha256)
        if path.stat().st_size > 1 << 20:
            raise StageGateError("program access registry exceeds its metadata budget")
        data = path.read_bytes()
        if sha256_bytes(data) != self.sha256:
            raise StageGateError("program access registry changed while reading")
        row = loads(data.decode("utf-8"))
        if type(row) is not dict or set(row) != {"schema", "public", "protected"} or row["schema"] != REGISTRY_SCHEMA:
            raise StageGateError("program access registry has unsupported membership fields")
        for name in ("public", "protected"):
            values = row[name]
            if (
                type(values) is not list
                or len(values) > 10000
                or any(not is_sha256(value) for value in values)
                or len(set(values)) != len(values)
            ):
                raise StageGateError("program access registry requires complete distinct source identities")
        if not row["public"] or set(row["public"]) & set(row["protected"]):
            raise StageGateError("public and protected program membership must be nonempty and disjoint")
        return row

    def require_public(self, source_sha256):
        row = self.verify()
        if source_sha256 in row["protected"]:
            raise StageGateError("protected programs are unavailable to public development before compiler freeze")
        if source_sha256 not in row["public"]:
            raise StageGateError("program is outside the original public source registry")


@dataclass(frozen=True)
class CpuTimingConfiguration:
    """Exact public memory/cache/clock inputs, without a timing qualification.

    Equality of these inputs is necessary for a matched study. Their source
    identity alone does not prove the engine consumed them or modelled their
    physical semantics.
    """

    memory: tuple[str, str]
    cache: tuple[str, str]
    clock: tuple[str, str]

    def record(self):
        result = {}
        for name in ("memory", "cache", "clock"):
            pair = getattr(self, name)
            if type(pair) is not tuple or len(pair) != 2:
                raise StageGateError("CPU timing needs explicit memory, cache and clock source selections")
            _pin(*pair)
            result[name] = {"path": pair[0], "sha256": pair[1]}
        return result

    @property
    def sha256(self):
        return document_sha256(self.record())


@dataclass(frozen=True)
class CpuEngineSelection:
    """Selection reopens the actual backend identity on each use.

    The backend is selected by the existing target registry. Public definitions
    cannot provide a callback, import string or replacement command.
    """

    target: str
    engine: str
    identity_sha256: str

    def identity(self):
        from merlin.runtime.backends.base import get_backend

        if type(self.target) is not str or not self.target or self.engine not in CPU_ENGINES:
            raise StageGateError("CPU evaluation requires an explicit supported engine and target")
        backend = get_backend(self.target)
        reader = getattr(backend, "execution_engine_identity", None)
        if reader is None:
            raise StageGateError("selected backend has no ordinary engine provenance adapter")
        identity = reader(self.engine)
        if (
            type(identity) is not dict
            or identity.get("engine") != self.engine
            or not is_sha256(identity.get("selection_sha256"))
            or type(identity.get("binary")) is not dict
            or set(identity["binary"]) != {"path", "sha256"}
        ):
            raise StageGateError("ordinary selected engine returned incomplete provenance")
        _pin(identity["binary"]["path"], identity["binary"]["sha256"])
        if document_sha256(identity) != self.identity_sha256:
            raise StageGateError("CPU engine selection changed since planning")
        return identity


def inspect_program_fidelity(*, scope, engine, phase, public, configuration_matched=False):
    """Report possible observation scopes, never infer an evaluated capability.

    ``public`` and ``configuration_matched`` are descriptive planning inputs.
    Execution independently rechecks the private registry and live evaluator;
    neither input can grant it. Legacy model/probe refusal is unchanged.
    """
    if scope not in SCOPES or engine not in CPU_ENGINES or phase not in PHASES:
        raise StageGateError("unknown program scope, CPU engine or evaluation phase")
    if type(public) is not bool or type(configuration_matched) is not bool:
        raise StageGateError("fidelity planning needs explicit visibility and configuration state")
    reasons = ["original independent source/runtime qualification remains required"]
    if not public and phase == "public_development":
        reasons.append("protected program unavailable before compiler freeze")
    if scope == "whole_model":
        reasons.append("component qualification does not establish whole-program execution coverage")
    if engine == "spike":
        reasons.append("ISA execution and counts are numerical/logical observations, not physical timing or traffic")
    if engine == "native_fir":
        reasons.append("fixed-memory native FIR execution is conditional functional evidence only")
    if not configuration_matched:
        reasons.append("final public memory/cache/clock configuration is not matched")
    reasons.append("output or source equivalence alone does not establish cycle equivalence or timing rankings")
    return {
        "schema": SCHEMA,
        "program_scope": scope,
        "phase": phase,
        "engine": engine,
        "execution_substrate": "CPU",
        "numerical_status": "UNKNOWN",
        "timing_status": "UNKNOWN",
        "physical_traffic_status": "UNKNOWN",
        "promotion": "NONE",
        "reasons": reasons,
    }


def evaluate_public_program_cpu(*, execution, selection, registry, compiler, member, corpus, workspace, timeout_s):
    """Invoke the fixed ordinary independently issued public numerical evaluator.

    This route deliberately does not call legacy ``require_probe_execution``.
    Its source/runtime obligations are additional, and it has no protected final
    or full-model fallback. Source-labelled full models remain UNKNOWN until an
    independently issued ordinary whole-program evaluator exists.
    """
    from .component_execution import ComponentIndependentExecution

    if (
        type(execution) is not ComponentIndependentExecution
        or type(selection) is not CpuEngineSelection
        or type(registry) is not ProgramAccessRegistry
    ):
        raise StageGateError("public program evaluation requires fixed ordinary live execution and engine owners")
    execution.verify()
    registry.require_public(member.source_sha256)
    if member.descriptor.get("operation", {}).get("op") != "component_program":
        raise StageGateError(
            "whole-program CPU numerical qualification is UNKNOWN; no ordinary issuer covers this source"
        )
    if selection.target != execution.independent_runtime.hardware_intake.target:
        raise StageGateError("public program engine targets another independently evaluated runtime")
    before = selection.identity()
    result = execution.execute(
        compiler=compiler, member=member, corpus=corpus, workspace=workspace, timeout_s=timeout_s
    )
    # A matching name is insufficient: join the actual ordinary process to the
    # selected binary and the ELF produced by this invocation, including output.
    records = [
        verify_invocation(path)
        for path in Path(workspace).rglob("invocation.json")
        if path.is_file() and not path.is_symlink()
    ]
    actual = [row for row in records if row.get("stage") == "selected_rocc_process"]
    if not actual or any(row.get("executable") != before["binary"] for row in actual):
        raise StageGateError("ordinary public numerical execution is not joined to the selected actual CPU engine")
    if selection.identity() != before:
        raise StageGateError("selected CPU engine changed during ordinary numerical execution")
    registry.require_public(member.source_sha256)
    return {
        "schema": SCHEMA,
        "program_scope": "component_program",
        "phase": "public_development",
        "engine": before,
        "execution_substrate": "CPU",
        "source_sha256": member.source_sha256,
        "registry_sha256": registry.sha256,
        "numerical_status": "PASS",
        "timing_status": "UNKNOWN",
        "physical_traffic_status": "UNKNOWN",
        "promotion": "NONE",
        "ordinary_result": result,
    }


def require_whole_program_cpu_timing(*, runtime, selection, configuration, final_configuration, scope):
    """Check live prerequisites without widening an existing component issuer.

    No ordinary whole-program physical issuer currently exists. Retain this
    explicit refusal after the original runtime/measurement checks, so a later
    producer must add its actual same-program/configuration evaluation rather
    than turning a saved certificate or matching hashes into authority.
    """
    from .component_measurement_qualification import IndependentMeasurementQualification
    from .component_runtime import require_independent_runtime

    if type(selection) is not CpuEngineSelection or any(
        type(value) is not CpuTimingConfiguration for value in (configuration, final_configuration)
    ):
        raise StageGateError("whole-program CPU timing requires exact engine and public configuration selections")
    selection.identity()
    if selection.engine in {"spike", "native_fir"}:
        raise StageGateError("selected CPU engine is functional only; matched physical timing is UNKNOWN")
    if configuration.sha256 != final_configuration.sha256:
        raise StageGateError("CPU timing differs from the final public memory/cache/clock configuration")
    from .component_runtime import IndependentComponentRuntime

    if type(runtime) is not IndependentComponentRuntime:
        raise StageGateError("CPU timing requires the original live independent measurement qualification")
    require_independent_runtime(
        runtime, required_roles=("feature_provider",), target_descriptor=runtime.target_descriptor
    )
    if type(runtime.qualification) is not IndependentMeasurementQualification:
        raise StageGateError("CPU timing requires the original live independent measurement qualification")
    runtime.qualification.verify()
    if scope != runtime.qualification.scope:
        raise StageGateError("CPU timing selected another complete-cost scope")
    raise StageGateError(
        "whole-program CPU timing is UNKNOWN: original component controls do not issue whole-program authority"
    )
