"""Located and portable identities for the global host verifier's source policy.

Creation checks loaded Python owners. Historical verification reads only the
recorded files beneath an independently admitted snapshot, never current imports.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common.paths import module_source_path
from merlin.common.source_membership import python_members

from . import contracts

V2_SCHEMA = "global_host_verification_policy_v2"
SCHEMA = "global_host_verification_policy_v3"
NAMESPACE = "merlin_experiments.phase2"
NAMESPACES = (NAMESPACE, "merlin.perf")
RESOURCE_FILES = (
    "gate_phases.yaml",
    "hardware_pins.yaml",
    "schemas/manifest.schema.json",
    "schemas/command_buffer.schema.json",
)
V2_MODULES = (
    "merlin.targetgen.sandbox.toolchain",
    "merlin.common.access",
    "merlin.common.digest",
    "merlin_experiments.source_snapshot",
    "merlin.targetgen.sandbox.answer_surfaces",
    "merlin.targetgen.sandbox.bwrap",
    "merlin.targetgen.sandbox.build_dependencies",
    "merlin.targetgen.sandbox.executable_dependencies",
    "merlin.targetgen.contract.build_service",
    "merlin.targetgen.contract.build_recipe",
    "merlin.perf.compiler_plan_evidence",
    "merlin.perf.task_cfg_evidence",
    "merlin.perf.task_instruction_evidence",
    "merlin.perf.task_route_presence",
    "merlin.perf.storage_encoding",
    "merlin.perf.structural_transitions",
    "merlin.perf.physical_transition_evidence",
    "merlin.perf.external_objective",
    "merlin.perf.model_placement",
    "merlin.perf.model_macs",
    "merlin.runtime.storage_binding",
    "merlin.runtime.prepack_authority",
    "merlin.runtime.captured_constants",
    "merlin.frontends.argument_identity",
    "merlin.perf.host_cfg_activity",
    "merlin.perf.analysis_worker",
    "merlin.perf.isolated_probe_provider",
    "merlin.perf.primitive_probe",
    "merlin.perf.instruction_motif",
    "merlin.perf.probe_relevance",
    "merlin.perf.structural_delta",
    "merlin.perf.completion_delta",
    "merlin.perf.context_probe",
    "merlin.perf.context_program",
    "merlin.perf.controlled_context_provider",
    "merlin.perf.fixed_work_context",
    "merlin.perf.paired_context_provider",
    "merlin.perf.static_imports",
    "merlin.perf.compiler_edit_scope",
    "merlin.perf.agent_guidance",
    "merlin.perf.agent_guidance_vocabulary",
    "merlin.kernels.cca_contract",
    "merlin.perf.host_region_qualifier",
    "merlin.perf.host_source_witness",
    "merlin.perf.source_convolution_witness",
    "merlin.perf.source_convolution_preparation",
    "merlin.perf.source_program_pair",
    "merlin.perf.source_contraction_witness",
    "merlin.perf.source_contraction_preparation",
    "merlin.perf.source_program_pair_provider",
    "merlin.perf.source_initializer_elision",
    "merlin.targetgen.conv_geometry",
    "merlin.targetgen.capsule_golden",
    "merlin.runtime.commandbuffer",
    "merlin.runtime.tensor",
    "merlin.perf.host_physical_transition_qualifier",
    "merlin.perf.native_host_witness_runner",
    "merlin.perf.mechanism_probe",
    "merlin.perf.historical_reference",
    "merlin.perf.harvest",
    "merlin.perf.work_volume",
    "merlin.perf.execution_policy",
    "merlin.targetgen.rocc.decode",
    "merlin.common.source_membership",
)


MODULES = V2_MODULES
V3_MODULES = tuple(name for name in V2_MODULES if not name.startswith("merlin.perf."))


def _module_identity(relative: str) -> str:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative or path.suffix != ".py":
        raise ValueError("host policy closure member is not a relative Python source")
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if any(not part.isidentifier() for part in parts):
        raise ValueError("host policy closure member has an invalid module identity")
    return "python/" + ".".join((NAMESPACE, *parts))


def location_sha256(record: Mapping[str, Any]) -> str:
    """Digest of source locations and their explicit ownership relationships."""
    return contracts.document_sha256({key: record[key] for key in ("sources", "identities", "closures")})


def _portable_v2_sha256(record: Mapping[str, Any]) -> str:
    return contracts.document_sha256(
        {
            "schema": V2_SCHEMA,
            "sources": {identity: record["sources"][path] for identity, path in record["identities"].items()},
            "closures": {name: closure["members"] for name, closure in record["closures"].items()},
        }
    )


def _plain_path(value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str):
        raise ValueError("host policy source location is malformed")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or (not path.is_dir() if directory else not path.is_file())
    ):
        raise ValueError("host policy source is absent, linked, or non-absolute")
    return path


def _content_v2_sha256(record: Mapping[str, Any], *, source_root: Path) -> str:
    """Verify v2 bytes and exact ownership inside an independently admitted source snapshot."""
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema", "sources", "identities", "closures", "sha256", "location_sha256"}
        or record.get("schema") != V2_SCHEMA
    ):
        raise ValueError("host verification policy is malformed")
    sources, identities, closures = (record[key] for key in ("sources", "identities", "closures"))
    if not all(isinstance(value, Mapping) for value in (sources, identities, closures)):
        raise ValueError("host verification policy ownership is malformed")
    root = _plain_path(str(Path(source_root).absolute()), directory=True)
    for location, digest in sources.items():
        path = _plain_path(location)
        if not path.is_relative_to(root):
            raise ValueError("host verification policy source escaped its sealed source root")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("host verification policy source digest is malformed")
        if contracts.sha256_file(path) != digest:
            raise ValueError("host verification policy source bytes changed")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in identities.items()):
        raise ValueError("host verification policy logical identities are malformed")
    if set(identities.values()) != set(sources) or set(closures) != {NAMESPACE}:
        raise ValueError("host verification policy source ownership differs")
    closure = closures[NAMESPACE]
    if (
        not isinstance(closure, Mapping)
        or set(closure) != {"root", "members"}
        or not isinstance(closure["members"], Mapping)
    ):
        raise ValueError("host verification policy closure is malformed")
    package_root = _plain_path(closure["root"], directory=True)
    if not package_root.is_relative_to(root):
        raise ValueError("host verification policy closure escaped its sealed source root")
    actual = python_members(package_root)
    if "__init__.py" not in actual or set(actual) != set(closure["members"]):
        raise ValueError("host verification policy closure membership changed")
    expected_ids = (
        {"python/" + name for name in V2_MODULES}
        | {"controller/global"}
        | {"resource/contract/" + relative for relative in RESOURCE_FILES}
    )
    for relative, path in actual.items():
        identity = _module_identity(relative)
        if identity in expected_ids:
            raise ValueError("host policy has duplicate logical source identities")
        expected_ids.add(identity)
        if closure["members"][relative] != identity or identities.get(identity) != str(path):
            raise ValueError("host verification policy closure ownership changed")
    if set(identities) != expected_ids:
        raise ValueError("host verification policy logical source membership changed")
    portable = _portable_v2_sha256(record)
    if record["sha256"] != portable or record["location_sha256"] != location_sha256(record):
        raise ValueError("host verification policy contradicts its source identities")
    return portable


def _namespace_roots(namespace: str) -> tuple[Path, ...]:
    """Inspect active namespace locations without importing their implementations."""
    package = importlib.import_module(namespace)
    roots = tuple(_plain_path(str(path), directory=True) for path in package.__path__)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("host policy namespace has absent or duplicate source roots")
    return tuple(sorted(roots))


def _namespace_members(namespace: str, roots: tuple[Path, ...]) -> dict[str, tuple[str, Path]]:
    members: dict[str, tuple[str, Path]] = {}
    identities: set[str] = set()
    paths: set[Path] = set()
    plain_modules: set[str] = set()
    for root in roots:
        for relative, path in python_members(root).items():
            identity = _module_identity(relative).replace("python/" + NAMESPACE, "python/" + namespace, 1)
            if relative in members or identity in identities or path in paths:
                raise ValueError("host policy has duplicate logical source identities")
            identities.add(identity)
            paths.add(path)
            members[relative] = (identity, path)
            if PurePosixPath(relative).name != "__init__.py":
                plain_modules.add(identity)
    if "__init__.py" not in members:
        raise ValueError("host policy package closure lacks its initializer")
    if any(other.startswith(module + ".") for module in plain_modules for other in identities):
        raise ValueError("host policy module shadows a package source identity")
    return dict(sorted(members.items()))


def _portable_sha256(record: Mapping[str, Any]) -> str:
    if record.get("schema") == V2_SCHEMA:
        return _portable_v2_sha256(record)
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported host verification policy schema")
    return contracts.document_sha256(
        {
            "schema": SCHEMA,
            "sources": {identity: record["sources"][path] for identity, path in record["identities"].items()},
            "closures": {name: closure["members"] for name, closure in record["closures"].items()},
        }
    )


def build_record(*, controller_source: Path, contract_root: Path) -> dict[str, Any]:
    """Capture V3 namespace membership and actual loaded owners, not a checkout layout."""
    identities: dict[str, str] = {}

    def add(identity: str, source: Path) -> None:
        if identity in identities:
            raise ValueError("host policy has duplicate logical source identities")
        source = Path(source)
        if source.is_symlink():
            raise ValueError("host policy source is linked")
        location = str(_plain_path(str(source.resolve())))
        if identity.startswith("python/") and any(
            name.startswith("python/") and value == location for name, value in identities.items()
        ):
            raise ValueError("host policy has duplicate Python source owners")
        identities[identity] = location

    for module in V3_MODULES:
        add("python/" + module, module_source_path(module))
    add("controller/global", controller_source)
    for relative in RESOURCE_FILES:
        add("resource/contract/" + relative, Path(contract_root) / relative)
    closures = {}
    all_roots: set[Path] = set()
    for namespace in NAMESPACES:
        roots = _namespace_roots(namespace)
        if any(root in all_roots for root in roots):
            raise ValueError("host policy namespaces share a source root")
        all_roots.update(roots)
        members = _namespace_members(namespace, roots)
        for identity, path in members.values():
            add(identity, path)
        closures[namespace] = {
            "roots": [str(root) for root in roots],
            "members": {relative: identity for relative, (identity, _path) in members.items()},
        }
    module_owners = {
        identity.removeprefix("python/"): Path(path)
        for identity, path in identities.items()
        if identity.startswith("python/")
    }
    for name, expected in module_owners.items():
        # Check the namespace initializer without importing its member modules.
        if name in NAMESPACES and module_source_path(name).resolve() != expected:
            raise ValueError(f"namespace initializer differs from host policy owner: {name}")
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        canonical = getattr(loaded, "__name__", name)
        origin = getattr(loaded, "__file__", None)
        if (
            module_owners.get(canonical) != expected
            or origin is None
            or sys.modules.get(canonical) is not loaded
            or Path(origin).resolve() != expected
        ):
            raise ValueError(f"loaded source origin differs from host policy owner: {name}")
    record = {
        "schema": SCHEMA,
        "sources": {path: contracts.sha256_file(Path(path)) for path in sorted(set(identities.values()))},
        "identities": dict(sorted(identities.items())),
        "closures": closures,
    }
    record["sha256"] = _portable_sha256(record)
    record["location_sha256"] = location_sha256(record)
    return record


def content_sha256(record: Mapping[str, Any], *, source_root: Path) -> str:
    """Decode historical V2 or V3 using only independently admitted snapshot bytes."""
    if isinstance(record, Mapping) and record.get("schema") == V2_SCHEMA:
        return _content_v2_sha256(record, source_root=source_root)
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema", "sources", "identities", "closures", "sha256", "location_sha256"}
        or record.get("schema") != SCHEMA
    ):
        raise ValueError("host verification policy is malformed")
    sources, identities, closures = (record[key] for key in ("sources", "identities", "closures"))
    if not all(isinstance(value, Mapping) for value in (sources, identities, closures)):
        raise ValueError("host verification policy ownership is malformed")
    root = _plain_path(str(Path(source_root).absolute()), directory=True)
    for location, digest in sources.items():
        path = _plain_path(location)
        if not path.is_relative_to(root):
            raise ValueError("host verification policy source escaped its sealed source root")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("host verification policy source digest is malformed")
        if contracts.sha256_file(path) != digest:
            raise ValueError("host verification policy source bytes changed")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in identities.items()):
        raise ValueError("host verification policy logical identities are malformed")
    if set(identities.values()) != set(sources) or set(closures) != set(NAMESPACES):
        raise ValueError("host verification policy source ownership differs")
    python_locations = [value for name, value in identities.items() if name.startswith("python/")]
    if len(set(python_locations)) != len(python_locations):
        raise ValueError("host policy has duplicate Python source owners")
    expected_ids = (
        {"python/" + name for name in V3_MODULES}
        | {"controller/global"}
        | {"resource/contract/" + relative for relative in RESOURCE_FILES}
    )
    all_roots: set[Path] = set()
    for namespace in NAMESPACES:
        closure = closures[namespace]
        if (
            not isinstance(closure, Mapping)
            or set(closure) != {"roots", "members"}
            or not isinstance(closure["members"], Mapping)
            or not isinstance(closure["roots"], list)
            or not closure["roots"]
        ):
            raise ValueError("host verification policy closure is malformed")
        roots = tuple(_plain_path(value, directory=True) for value in closure["roots"])
        if len(set(roots)) != len(roots) or tuple(sorted(roots)) != roots or any(r in all_roots for r in roots):
            raise ValueError("host policy namespace has duplicate or unordered source roots")
        all_roots.update(roots)
        if any(not path.is_relative_to(root) for path in roots):
            raise ValueError("host verification policy closure escaped its sealed source root")
        actual = _namespace_members(namespace, roots)
        if set(actual) != set(closure["members"]):
            raise ValueError("host verification policy closure membership changed")
        for relative, (identity, path) in actual.items():
            if identity in expected_ids:
                raise ValueError("host policy has duplicate logical source identities")
            expected_ids.add(identity)
            if closure["members"][relative] != identity or identities.get(identity) != str(path):
                raise ValueError("host verification policy closure ownership changed")
    if set(identities) != expected_ids:
        raise ValueError("host verification policy logical source membership changed")
    portable = _portable_sha256(record)
    if record["sha256"] != portable or record["location_sha256"] != location_sha256(record):
        raise ValueError("host verification policy contradicts its source identities")
    return portable
