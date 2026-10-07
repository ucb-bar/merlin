"""Read-only Phase 0 evidence selection and exact run-owned snapshots."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _existing_receipt_path(member: Mapping) -> Path | None:
    path = member.get("path") if isinstance(member, Mapping) else None
    if not isinstance(path, str) or not path:
        return None
    try:
        return Path(path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None


def _same_selected_file(left: Mapping, right: Mapping) -> bool:
    """Compare byte-bound file identities across equivalent symlink spellings."""
    from merlin.common.digest import is_sha256

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    digest = left.get("sha256")
    resolved = _existing_receipt_path(left)
    return (
        is_sha256(digest)
        and digest == right.get("sha256")
        and resolved is not None
        and resolved == _existing_receipt_path(right)
    )


def _same_source_consistency(left: Mapping, right: Mapping) -> bool:
    """Compare full receipts after resolving only their selected source paths."""

    def resolved_sources(document: Mapping) -> dict | None:
        if not isinstance(document, Mapping) or not isinstance(document.get("sources"), list):
            return None
        result = copy.deepcopy(document)
        for source in result.get("sources", []):
            resolved = _existing_receipt_path(source)
            if resolved is None:
                return None
            source["path"] = str(resolved)
        return result

    normalized_left, normalized_right = resolved_sources(left), resolved_sources(right)
    return normalized_left is not None and normalized_right is not None and normalized_left == normalized_right


def _reference_inventory_root(role: str, root: Path, software_doc: Mapping) -> Path:
    """Select the source package used by a declared numerical engine.

    The SpecIR adapter imports ``root/specir``. Snapshotting the entire project
    would also bind unrelated targets, builds and generated ``out`` artifacts;
    those are not inputs to its pure reduction model.
    """
    model = (software_doc.get("numerical_semantics") or {}).get("model") or {}
    if role == "numerical_model" and model.get("engine") == "specir_fp_reduce":
        package = root / "specir"
        if not (package / "__init__.py").is_file():
            raise ValueError(f"selected SpecIR source package is absent: {package}")
        return package
    return root


def _reference_inventory_members(
    role: str, root: Path, software_doc: Mapping, extensions: set[str], excluded: set[str]
) -> tuple[Path, ...]:
    """Inventory the files loaded by a numerical engine, preserving owner-relative paths."""
    model = (software_doc.get("numerical_semantics") or {}).get("model") or {}
    if role == "numerical_model" and model.get("engine") == "mx_block_reference":
        # Both MX numerical consumers load this file directly by path. Keeping
        # its owner-relative location lets the frozen launcher replay that load.
        reference = root / "mlc" / "validate" / "mx_ref.py"
        if not reference.is_file():
            raise ValueError(f"selected MX numerical reference is absent: {reference}")
        return (reference,)
    inventory_root = _reference_inventory_root(role, root, software_doc)
    return tuple(
        leaf
        for leaf in sorted(inventory_root.rglob("*"))
        if leaf.is_file()
        and leaf.suffix in extensions
        and not excluded.intersection(leaf.relative_to(inventory_root).parts)
    )


def _native_baseline_observations(selections, applications, observe) -> dict:
    """Select exact generated host checks, never generalize them into host admission."""
    import numpy as np

    from merlin.common.mlir_query import forward_signature
    from merlin.runtime.dispatch_runtime import resolve_forward_args
    from merlin.targetgen.application_inventory import verify_capture_receipt
    from merlin_experiments import model_qualification as qualification

    if not isinstance(selections, Mapping) or set(selections) - set(applications):
        raise ValueError("native qualifications require explicit selected application labels")
    results = {}
    for label, location in sorted(selections.items()):
        path = qualification._plain(Path(location))
        document = json.loads(observe(path, f"native-qualification:{label}", required=True))
        request, observed = document.get("request") or {}, document.get("observations") or {}
        runtime = observed.get("runtime") or {}
        if (
            document.get("schema") != qualification.SCHEMA
            or document.get("status") != "completed"
            or document.get("worker_returncode") != 0
            or document.get("error") is not None
            or document.get("native_host_numerical_verified") is not True
            or document.get("whole_workload_validation_verified") is not False
            or document.get("target") is not None
            or request.get("native_host_only") is not True
            or any(request.get(key) is not None for key in ("package", "target", "evidence_bundle"))
            or request.get("execute") is not False
            or runtime.get("target_executed") is not False
            or runtime.get("native_host_executed") is not True
            or runtime.get("status") != "numerically_matched"
            or observed.get("native_host_numerical_verified") is not True
            or observed.get("full_native_lowering_verified") is not True
        ):
            raise ValueError("selected native qualification is not a completed finite native-only check")
        app = applications[label]
        model = qualification._plain(Path(app["capture_source_path"]))
        bundle = model.parent
        if qualification._plain(Path(request["bundle"]), directory=True) != bundle:
            raise ValueError("native qualification selects a different capture bundle")
        workflow = qualification.inspect_workflow(bundle)
        if workflow != request.get("workflow") or workflow != observed.get("workflow"):
            raise ValueError("native qualification workflow/input bytes differ from selected capture")
        programs = workflow["programs"]
        rows = observed.get("native_lowerings") or []
        if workflow["session"] or len(programs) != 1 or len(rows) != 1:
            raise ValueError("native qualification must describe one standalone captured program")
        row = rows[0]
        if row.get("status") != "numerically_matched":
            raise ValueError("native qualification program is not numerically matched")
        if row.get("model_sha256") != app["capture_sha256"] or programs[0]["model_sha256"] != app["capture_sha256"]:
            raise ValueError("native qualification capture identity differs from selected inventory")
        receipt = verify_capture_receipt(model)
        # A newer verifier may project byte-bound integerization facts from
        # meta.json. The installed native qualifier records the original stable
        # capture identity, not that optional projection; compare every stable
        # field exactly, then require the projection to agree with this selected
        # application's independently inventoried bytes.
        identity_fields = {"status", "receipt_sha256", "source_closure_verified", "errors"}
        identity = {key: receipt.get(key) for key in identity_fields}
        qualified_receipt = row.get("capture_receipt")
        if (
            receipt["status"] != "verified_materialized"
            or not isinstance(qualified_receipt, dict)
            or not identity_fields <= set(qualified_receipt)
            or set(qualified_receipt) - identity_fields - {"capture_integerization"}
            or any(qualified_receipt[key] != identity[key] for key in identity_fields)
            or app.get("capture_receipt") != identity
            or app.get("capture_integerization") != receipt.get("capture_integerization")
            or (
                "capture_integerization" in qualified_receipt
                and qualified_receipt["capture_integerization"] != receipt.get("capture_integerization")
            )
        ):
            raise ValueError("native qualification capture receipt differs from selected capture")
        for member, identity in workflow["members"].items():
            raw = observe(qualification._plain(bundle / member), f"native-input:{label}", required=True)
            if _digest(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
                raise ValueError("native qualification input changed during selection")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict) or not {
            "request.json",
            "observations.json",
            "program-000/output.npy",
        } <= set(artifacts):
            raise ValueError("native qualification lacks its generated observation artifacts")
        saved = {}
        for member, identity in artifacts.items():
            relative = Path(member)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("native qualification artifact escapes its owner")
            raw = observe(qualification._plain(path.parent / relative), f"native-artifact:{label}", required=True)
            if _digest(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
                raise ValueError("native qualification artifact differs from its recorded bytes")
            saved[member] = raw
        if json.loads(saved["request.json"]) != request or json.loads(saved["observations.json"]) != observed:
            raise ValueError("native qualification embedded records differ from generated artifacts")
        compiler = document.get("compiler") or {}
        producer = compiler.get("producer") or {}
        producer_raw = observe(
            qualification._plain(Path(producer["path"])), "native-qualification-producer", required=True
        )
        if compiler.get("route") != "merlin.llvmlower.lower_model_file:host" or _digest(producer_raw) != producer.get(
            "sha256"
        ):
            raise ValueError("native qualification producer identity changed or selects another route")
        policy = request.get("numeric_policy")
        if (
            not isinstance(policy, dict)
            or set(policy) != {"atol", "rtol"}
            or any(
                type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in policy.values()
            )
            or row.get("numeric_policy") != policy
        ):
            raise ValueError("native qualification requires its exact finite numerical policy")
        inputs, outputs = forward_signature(model)
        input_abi = [{"shape": shape, "dtype": dtype} for shape, dtype in inputs]
        output_abi = [{"shape": shape, "dtype": dtype} for shape, dtype in outputs]
        arguments = resolve_forward_args(bundle)
        storage = {"f32": "float32", "i64": "int64", "i32": "int32", "i8": "int8", "i1": "bool"}
        if (
            len(outputs) != 1
            or outputs[0][1] != "f32"
            or len(arguments) != len(inputs)
            or any(
                list(value.shape) != shape or str(value.dtype) != storage.get(dtype)
                for value, (shape, dtype) in zip(arguments, inputs, strict=True)
            )
        ):
            raise ValueError("native qualification saved arguments differ from its exact supported ABI")
        if (
            row.get("input_abi") != input_abi
            or row.get("output_abi") != output_abi
            or row.get("input_buffer_sha256") != [hashlib.sha256(value.tobytes()).hexdigest() for value in arguments]
        ):
            raise ValueError("native qualification typed ABI/argument bytes differ from selected capture")
        actual = np.load(io.BytesIO(saved["program-000/output.npy"]), allow_pickle=False)
        golden = np.load(
            io.BytesIO(observe(bundle / "golden.npy", f"native-input:{label}", required=True)), allow_pickle=False
        )
        if (
            actual.shape != golden.shape
            or list(actual.shape) != outputs[0][0]
            or str(actual.dtype) != "float32"
            or actual.dtype != golden.dtype
            or not np.isfinite(actual).all()
            or not np.isfinite(golden).all()
        ):
            raise ValueError("native qualification output/reference ABI or finiteness differs")
        error = float(np.max(np.abs(actual.astype(np.float64) - golden)))
        if (
            not np.allclose(actual, golden, **policy, equal_nan=False)
            or row.get("max_absolute_error") != error
            or row.get("nonfinite_output_elements") != 0
        ):
            raise ValueError("native qualification saved output fails its recorded numerical comparison")
        results[label] = {
            "status": "whole_program_numerically_matched",
            "executor": "native_cpu",
            "placement": "host",
            "capture_sha256": app["capture_sha256"],
            "receipt_sha256": _digest(observe(path, f"native-qualification:{label}", required=True)),
            "compiler_identity": copy.deepcopy(compiler),
            "input_abi": input_abi,
            "output_abi": output_abi,
            "numeric_policy": policy,
            "max_absolute_error": error,
            "target_executed": False,
            "source": str(path),
            "qualification": (
                "exact saved standalone program and inputs only; "
                "no RVV, accelerator or independent per-operation support grant"
            ),
        }
    return results


@dataclass(frozen=True)
class EvidenceSource:
    path: Path
    role: str
    content: bytes

    @property
    def sha256(self) -> str:
        return _digest(self.content)


@dataclass(frozen=True)
class EvidenceSelection:
    target: str
    source_snapshots: tuple[EvidenceSource, ...]
    views_json: bytes
    raw_facts: bytes | None
    archived_artifacts: tuple[tuple[str, bytes], ...] = ()
    instruction_semantics_source: bytes | None = None

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(source.path for source in self.source_snapshots)

    @property
    def raw_facts_sha256(self) -> str | None:
        return _digest(self.raw_facts) if self.raw_facts is not None else None

    @property
    def derivation_identity(self) -> dict[str, str | None]:
        """Capability inputs that corpus derivation and capsule writing must share.

        Support paths are relative to their selected package so an installed copy
        with identical bytes has the same identity. Readout facets bind the
        effective provider semantics, not just the target contract or raw RTL.
        """
        support = [source for source in self.source_snapshots if source.role == "support-source"]
        root = Path(os.path.commonpath([str(source.path.absolute().parent) for source in support])) if support else None
        support_rows = [
            {"path": str(source.path.absolute().relative_to(root)), "sha256": source.sha256} for source in support
        ]
        return {
            "contract_sha256": _digest(_json(self.contract)),
            "raw_facts_sha256": self.raw_facts_sha256,
            "readout_facets_sha256": _canonical_digest(self.readout_facets),
            "support_sources_sha256": _canonical_digest(sorted(support_rows, key=lambda row: row["path"])),
            "instruction_semantics_sha256": (
                _digest(self.instruction_semantics_source) if self.instruction_semantics_source is not None else None
            ),
        }

    def __getattr__(self, name: str) -> Any:
        # Every consumer receives a fresh value. The authoritative views are
        # immutable serialized bytes, not mutable objects shared with callers.
        views = json.loads(self.views_json)
        if name not in views:
            raise AttributeError(name)
        return views[name]


def select_evidence(
    target: str,
    *,
    descriptor=None,
    capture_python=None,
    capability_contract_path=None,
    facts_path=None,
    hardware_spec=None,
    software_spec=None,
    conformance_spec=None,
    inventory_path=None,
    native_qualifications=None,
    prohibited_roles=(),
) -> EvidenceSelection:
    """Observe selected source bytes once; never extract facts or modify a checkout.

    Missing native evidence produces diagnostic unknowns. Malformed explicitly
    selected documents fail rather than selecting another source implicitly.
    """
    from merlin.perf.profile import derive_profile
    from merlin.runtime.backends.base import execution_capability_facts
    from merlin.targetgen import readout_facet, target_registry
    from merlin.targetgen.rtl import facts as rtl_facts

    sources: dict[Path, EvidenceSource] = {}
    diagnostics: list[dict[str, Any]] = []
    excluded = {".git", "__pycache__", "build", ".venv", "out"}
    extensions = {".py", ".json", ".yaml", ".yml", ".mlir", ".h", ".hpp", ".cpp", ".c", ".inc", ".S"}

    def observe(path, role, *, required=False) -> bytes | None:
        path = Path(path).absolute()
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError("environment configuration must not enter a Phase 0 evidence bundle")
        if path in sources:
            return sources[path].content
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            if required:
                raise
            diagnostics.append({"component": role, "status": "unknown", "reason": f"unavailable: {path}"})
            return None
        sources[path] = EvidenceSource(path, role, raw)
        return raw

    def document(value, role, *, required=True) -> dict:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        raw = observe(value, role, required=required)
        if raw is None:
            return {}
        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"{value}: {role} must be a mapping")
        return loaded

    descriptor_path = getattr(descriptor, "path", descriptor)
    descriptor_doc = document(descriptor_path, "descriptor") if descriptor_path is not None else {}
    application_inventory = None
    frontend_traces, application_graphs, framework_catalogs, host_capabilities = {}, {}, {}, {}
    inventory_identity = {"status": "not_available", "reason": "no selected detailed application inventory"}
    capture_attestations: dict[str, Any] = {}
    whole_program_admission = None
    if conformance_spec is not None or inventory_path is not None:
        from .profiles import application_inventory_path

        requirement = (
            document(conformance_spec, "conformance-spec", required=False) if conformance_spec is not None else {}
        )
        if requirement.get("target", target) != target:
            raise ValueError("selected conformance requirement target differs from selected target")
        sidecar = (
            application_inventory_path(conformance_spec, document=requirement) if conformance_spec is not None else None
        )
        if inventory_path is not None:
            selected_inventory = Path(inventory_path).absolute()
            if conformance_spec is not None and (sidecar is None or sidecar.absolute() != selected_inventory):
                raise ValueError("explicit application inventory differs from selected conformance sidecar")
            if selected_inventory.is_symlink():
                raise ValueError("selected application inventory may not be a symlink")
            sidecar = selected_inventory
        declared_whole_program = requirement.get("whole_program_admission")
        if conformance_spec is not None and isinstance(declared_whole_program, Mapping):
            from .program_admission import operations_digest

            name = declared_whole_program.get("sidecar")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("whole-program admission sidecar must be one adjacent basename")
            location = Path(conformance_spec).parent / name
            if location.is_symlink():
                raise ValueError("whole-program admission sidecar may not be a symlink")
            raw_whole_program = observe(location, "whole-program-admission", required=True)
            whole_program_admission = json.loads(raw_whole_program)
            if operations_digest(whole_program_admission) != declared_whole_program.get("sha256"):
                raise ValueError("whole-program admission sidecar differs from the requirement's commitment")
        if sidecar is not None:
            raw_inventory = observe(sidecar, "application-inventory", required=False)
            if raw_inventory is not None:
                application_inventory = json.loads(raw_inventory)
                if not isinstance(application_inventory, dict):
                    raise ValueError("selected detailed application inventory must be a mapping")
                actual = _canonical_digest(application_inventory)
                expected = (requirement.get("application_demands") or {}).get("full_inventory_sha256")
                if expected is not None and expected != actual:
                    raise ValueError("selected application inventory differs from conformance requirement")
                declared = (descriptor_doc.get("workload_spec") or {}).get("applications") or []
                observed = application_inventory.get("applications")
                if not isinstance(observed, dict):
                    raise ValueError("selected detailed application inventory requires an application mapping")
                # A directory selector is not an immutable list of model labels.
                # Do not enumerate it here or silently treat its characters as labels.
                labels = [str(label) for label in declared] if isinstance(declared, (list, tuple, dict)) else None
                roster_matches = labels is not None and len(labels) == len(set(labels)) and set(observed) == set(labels)
                inventory_identity = {
                    "status": "digest_bound" if expected is not None and roster_matches else "unverified",
                    "raw_sha256": _digest(raw_inventory),
                    "content_sha256": actual,
                    "requirement_digest_present": expected is not None,
                    "declared_roster_matches": roster_matches,
                    "declared_applications": sorted(labels) if labels is not None else None,
                    "observed_applications": sorted(observed),
                }
                # Attestations issued at derivation travel with the byte-bound
                # requirement; the coverage gate re-verifies each one from disk.
                capture_attestations = requirement.get("capture_execution_attestations") or {}
                if not isinstance(capture_attestations, dict) or (
                    capture_attestations and set(capture_attestations) != set(observed)
                ):
                    raise ValueError("capture execution attestations must cover exactly the selected applications")
                from merlin.targetgen.application_inventory import application_graph_inventory

                for label, application in sorted(observed.items()):
                    # Only fresh inventories retain the exact capture owner.
                    # Do not guess historical store paths from basename tags.
                    selected_capture = application.get("capture_source_path")
                    if not isinstance(selected_capture, str) or not selected_capture:
                        continue
                    capture_path = Path(selected_capture)
                    if not capture_path.is_absolute():
                        capture_path = sidecar.parent / capture_path
                    capture_raw = observe(capture_path, f"application-capture:{label}")
                    if capture_raw is None:
                        continue
                    if _digest(capture_raw) != application.get("capture_sha256"):
                        raise ValueError(f"selected application capture bytes changed: {label}")
                    application_graphs[label] = application_graph_inventory(capture_path)
                    if application_graphs[label]["capture_sha256"] != _digest(capture_raw):
                        raise ValueError(f"application graph was derived from changed source bytes: {label}")
                    normalized = (application.get("capture_normalization") or {}).get("output_sha256")
                    if normalized and application_graphs[label]["normalized_mlir_sha256"] != normalized:
                        raise ValueError(
                            f"selected application normalization differs from the derived inventory: {label}; "
                            "use the same frontend/MLIR environment for derivation and Phase 0 execution"
                        )
                    for name, filename, destination in (
                        ("frontend-trace", "frontend-trace.json", frontend_traces),
                        ("framework-catalog", "pytorch-opset.json", framework_catalogs),
                    ):
                        record = application.get(name.replace("-", "_")) or {}
                        location = (
                            (record.get("source_path") or record.get("path")) if isinstance(record, Mapping) else None
                        )
                        member = Path(location) if location else capture_path.with_name(filename)
                        if not member.is_absolute():
                            member = capture_path.parent / member
                        if member.is_symlink():
                            raise ValueError(f"selected {name} may not be symlinked: {label}")
                        raw_member = observe(member, f"application-{name}:{label}")
                        if raw_member is None:
                            continue
                        expected_member = (
                            (record.get("raw_sha256") or record.get("sha256")) if isinstance(record, Mapping) else None
                        )
                        if expected_member is not None and expected_member != _digest(raw_member):
                            raise ValueError(f"selected application {name} bytes changed: {label}")
                        parsed = json.loads(raw_member)
                        if not isinstance(parsed, dict):
                            raise ValueError(f"selected application {name} must be a mapping: {label}")
                        destination[label] = parsed
    if inventory_identity["status"] != "digest_bound":
        diagnostics.append(
            {
                "component": "application-inventory",
                "status": "unknown",
                "reason": inventory_identity.get("reason", "inventory digest or declared roster is unverified"),
            }
        )
    hardware_doc = document(hardware_spec, "hardware-spec")
    if hardware_doc.get("target", target) != target:
        raise ValueError("hardware spec target differs from selected target")
    if "schema" in hardware_doc and hardware_doc["schema"] != "merlin.hardware_selection.v1":
        raise ValueError("unsupported hardware selection schema")
    software_doc = document(software_spec, "software-spec")
    contract = (
        document(capability_contract_path, "target-contract", required=True)
        if capability_contract_path is not None
        else {}
    )
    residual, provider = {}, None
    try:
        provider = target_registry.resolve(target)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        diagnostics.append({"component": "support", "status": "unknown", "reason": str(exc)})
    if provider is not None:
        # Capability input selection is independent of executable support ownership.
        # The existing explicit contract selector must not be ignored merely because
        # support code was selected from an external provider.
        if capability_contract_path is None:
            contract = document(rtl_facts.target_contract_path(target), "target-contract", required=False)
        residual = document(provider.base / "contracts" / "residual.yaml", "residual", required=False)
        # Bind support code and its declarative/header dependencies before hooks
        # execute. This is an inventory, not a candidate grant or qualification.
        for path in sorted(provider.base.rglob("*")):
            if (
                path.is_file()
                and path.suffix in extensions
                and not excluded.intersection(path.relative_to(provider.base).parts)
            ):
                observe(path, "support-source")
    if contract and contract.get("name") != target:
        raise ValueError("selected backend capability contract differs from selected target")
    datapath = {}
    if software_spec is not None:
        from merlin.targetgen.software_spec import (
            capability_contract,
            numerical_datapath,
            software_spec_references,
            validate_software_spec,
        )

        software_doc = validate_software_spec(software_doc, target=target, source=str(software_spec))
        if not isinstance(software_spec, Mapping):
            try:
                references = software_spec_references(software_spec, document=software_doc)
            except ValueError as exc:
                references = {}
                diagnostics.append({"component": "numerical-model", "status": "unknown", "reason": str(exc)})
            for role, path in references.items():
                if path.is_dir():
                    for leaf in _reference_inventory_members(role, path, software_doc, extensions, excluded):
                        observe(leaf, f"software-reference:{role}")
                else:
                    observe(path, f"software-reference:{role}", required=True)
        contract = capability_contract(software_doc, base_contract=contract)
        datapath = numerical_datapath(software_doc)
    # The hardware selection is a source declaration, not an inferred dtype.
    for field, value in hardware_doc.items():
        if field.endswith("_path") and isinstance(value, str):
            owner = Path(hardware_spec).parent if not isinstance(hardware_spec, Mapping) else Path.cwd()
            observe(owner / value, f"hardware-reference:{field}")
    try:
        selected_path = rtl_facts.find_facts(target, explicit=facts_path)
    except FileNotFoundError as exc:
        selected_path = None
        diagnostics.append({"component": "facts", "status": "unknown", "reason": str(exc)})
    raw_facts = observe(selected_path, "rtl-facts") if selected_path is not None else None
    loaded_facts = json.loads(raw_facts) if raw_facts is not None else {}
    if not isinstance(loaded_facts, dict) or not isinstance(loaded_facts.get("facts", {}), dict):
        raise ValueError("selected RTL facts must contain a facts mapping")
    if not loaded_facts.get("facts"):
        diagnostics.append({"component": "facts", "status": "unknown", "reason": "no populated RTL facts selected"})
    instruction_semantics_source = None
    instruction_semantics = {
        "schema": "merlin.instruction_semantics.v1",
        "target": target,
        "status": "UNKNOWN",
        "unknowns": ["selected_target_contract_has_no_instruction_semantics_resource"],
        "instructions": [],
    }
    instruction_resource = contract.get("instruction_semantics")
    if instruction_resource is not None:
        if provider is None or not isinstance(instruction_resource, str) or not instruction_resource.strip():
            raise ValueError("instruction_semantics requires a selected support provider and relative resource")
        from merlin.targetgen.instruction_semantics import normalize_instruction_semantics
        from merlin.targetgen.providers import contained_resource

        selected_instruction_path = contained_resource(provider.base, instruction_resource)
        instruction_semantics_source = observe(selected_instruction_path, "instruction-semantics", required=True)
        authored_instructions = yaml.safe_load(instruction_semantics_source)
        instruction_semantics = normalize_instruction_semantics(
            authored_instructions,
            software_spec=software_doc,
            rtl_facts=loaded_facts,
            target=target,
            source_bytes=instruction_semantics_source,
            software_source_bytes=(
                sources[Path(software_spec).absolute()].content
                if software_spec is not None and not isinstance(software_spec, Mapping)
                else None
            ),
            rtl_source_bytes=raw_facts,
        )
    # Only explicit full paths identify actual RTL owners. Basenames and widths
    # are not enough to reconstruct provenance or declare numeric support.
    inputs = loaded_facts.get("inputs") or {}
    body = loaded_facts.get("facts") or {}
    origin = body.get("source") or {}
    source_records = [
        inputs,
        origin,
        *(inputs.get("firrtl_inputs") or []),
        *(inputs.get("reader_sources") or []),
        *((loaded_facts.get("source_consistency") or {}).get("sources") or []),
        ((loaded_facts.get("source_consistency") or {}).get("production") or {}).get("input_preparation") or {},
    ]
    for record in source_records:
        if not isinstance(record, Mapping):
            continue
        for key, value in record.items():
            if (key.endswith("_path") or key == "path") and isinstance(value, str) and value:
                path = Path(value)
                if path.is_absolute():
                    observed = observe(path, f"rtl-source:{key}")
                    expected = record.get("sha256" if key == "path" else key.removesuffix("_path") + "_sha256")
                    if (
                        observed is not None
                        and isinstance(expected, str)
                        and len(expected) == 64
                        and expected != _digest(observed)
                    ):
                        diagnostics.append(
                            {
                                "component": "rtl-source",
                                "status": "contradiction",
                                "reason": f"recorded source digest differs from observed bytes: {path}",
                            }
                        )
    for record in body.get("interfaces") or []:
        if isinstance(record, Mapping) and record.get("name") == "register_bundle_layouts" and record.get("source"):
            observe(record["source"], "readout-register-source")
    taxonomy = copy.deepcopy(body.get("isa_taxonomy") or {})
    if descriptor_path is not None:
        from merlin.targetgen import isa_taxonomy
        from merlin.targetgen.target_experiment import load_target_experiment

        try:
            te = descriptor if hasattr(descriptor, "isa_headers") else load_target_experiment(descriptor_path)
            from merlin.common.paths import repo_root, resolve_grant

            source_root = te.source_root if te.source_root is not None else repo_root()
            board_catalog = te.selected_board_catalog()
            if board_catalog is not None:
                observe(board_catalog, "host-board-catalog", required=True)
            if te.host_lanes is not None:
                for name, lane in sorted(te.host_lanes.profiles.items()):
                    try:
                        capability_path, capabilities, identity = lane.resolve_capabilities(
                            root=source_root, descriptor=te.path
                        )
                        if capability_path is not None:
                            observe(capability_path, f"host-capability-spec:{name}", required=True)
                        host_capabilities[name] = {**identity, "capability_spec": capabilities}
                        # Preserve the selected immutable payload identity, not
                        # only the authored claim or a mutable checkout path.
                        package_root = source_root / lane.package
                        for member in sorted(package_root.rglob("*")):
                            if member.is_file():
                                observe(member, f"host-package:{name}", required=True)
                    except (OSError, ValueError) as exc:
                        host_capabilities[name] = {"status": "unknown", "reason": str(exc)}
                        diagnostics.append(
                            {"component": f"host-capabilities:{name}", "status": "unknown", "reason": str(exc)}
                        )
            headers = [str(resolve_grant(header, root=source_root)) for header in te.isa_headers]
            for header in headers:
                observe(header, "isa-source")
            # The taxonomy's historical memo is path-keyed. A fresh selection
            # must derive from the source bytes just observed, not that memo.
            for source in sources.values():
                if source.role == "isa-source":
                    isa_taxonomy._CACHE.pop(str(source.path), None)
            model = (contract.get("runner") or {}).get("model_ext")
            taxonomy = dict(
                isa_taxonomy.derive_isa_taxonomy(SimpleNamespace(target=target, isa_headers=headers), model_ext=model)
            )
        except Exception as exc:  # noqa: BLE001 -- absent ISA tooling is unknown evidence
            taxonomy = {
                "status": "unknown",
                "by_class": {},
                "by_mnemonic": {},
                "asm_mnemonics": {},
                "unknown": {"taxonomy": f"{type(exc).__name__}: {exc}"},
            }
            diagnostics.append(
                {"component": "isa-taxonomy", "status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
            )
    # Replace physical shape declarations only where one extracted array
    # settles them. Never reinterpret a port width as numeric support.
    arrays = [array for array in body.get("arrays") or [] if isinstance(array, Mapping)]
    caps = contract.get("capabilities") or {}
    if isinstance(caps, dict) and any(key in caps for key in ("mesh", "tile")):
        for key in ("mesh", "tile"):
            declared = caps.get(key)
            if not isinstance(declared, dict):
                continue
            if len(arrays) == 1 and arrays[0].get("rows") and arrays[0].get("cols"):
                geometry = {field: arrays[0][field] for field in ("rows", "cols")}
                if any(declared.get(field) not in (None, value) for field, value in geometry.items()):
                    diagnostics.append(
                        {
                            "component": "physical-contract",
                            "status": "contradiction",
                            "reason": f"{key} declaration differs from selected array geometry",
                        }
                    )
                caps[key] = {**declared, **geometry}
            else:
                del caps[key]
                diagnostics.append(
                    {
                        "component": "physical-contract",
                        "status": "unknown",
                        "reason": f"{key} declaration has no unique selected array geometry",
                    }
                )
    snapshot_bytes = {str(path): source.content for path, source in sources.items()}
    refreshed_facts = dict(readout_facet.with_current_register_layouts(loaded_facts, source_bytes=snapshot_bytes))
    with rtl_facts.observed_facts(target, refreshed_facts, selected_path):
        readout_inputs = readout_facet.capture_inputs(target, facts=refreshed_facts, include_taxonomy=False)
    readout_inputs["taxonomy"] = taxonomy
    facets = readout_facet.for_target(target, contract=contract, facts=refreshed_facts, readout_inputs=readout_inputs)
    # The selected contract's scale claim must agree with the selected RTL
    # readout. Keep a mismatch diagnostic: narrowing a declaration or certifying
    # the software semantics requires its own reviewed input and fresh run.
    units = [unit for unit in contract.get("compute_units") or () if isinstance(unit, Mapping)]
    selected_facets = facets if units else []
    for unit, facet in zip(units, selected_facets, strict=True):
        for finding in readout_facet.reconcile(unit, facet):
            diagnostics.append(
                {
                    "component": "readout-scaling",
                    "status": "contradiction" if finding["kind"] == "scaling_exceeds_readout" else "unknown",
                    "reason": finding["why"],
                    "finding": finding,
                    "contract_sha256": _canonical_digest(contract),
                    "raw_facts_sha256": _digest(raw_facts) if raw_facts is not None else None,
                    "readout_facet_sha256": _canonical_digest(facet.to_dict()),
                }
            )
    from merlin.targetgen.quant_recipe import derive_candidates

    quantization_candidates = [candidate.to_dict() for candidate in derive_candidates(contract, facets)]
    # Fill the authored spec's fact-derived declarations from the same fact view the drift check
    # compares against, under the experiment's prohibited roles. Everything downstream of this
    # selection (screens, quantization contract, recipes, the frozen corpus) sees the resolved spec;
    # the authored bytes stay the selected source identity.
    software_derivation = None
    if software_spec is not None:
        from merlin.targetgen import spec_fact_drift

        # Some capability helpers still resolve by target name. Their reads
        # must use this selection too, never extract from an ambient provider.
        with (
            target_registry.observed_contract(target, contract),
            rtl_facts.observed_facts(target, refreshed_facts, selected_path),
        ):
            fact_view = spec_fact_drift.fact_capabilities(
                target=target,
                contract=contract,
                raw_facts=loaded_facts,
                readout_facets=[facet.to_dict() for facet in facets],
                quantization_candidates=quantization_candidates,
                taxonomy=taxonomy,
                prohibited_roles=prohibited_roles,
            )
        software_doc, software_derivation = spec_fact_drift.resolve_spec(software_doc, fact_view)
        software_derivation["prohibited_instruction_roles"] = sorted(set(prohibited_roles))
        for row in software_derivation["unresolved"]:
            diagnostics.append(
                {
                    "component": "software-spec-derivation",
                    "status": "unknown",
                    "reason": row["reason"],
                    "declaration": row.get("id") or row.get("format"),
                }
            )
    target_profile = derive_profile(target, facts=refreshed_facts, residual=residual, contract=contract).to_dict()
    with rtl_facts.observed_facts(target, refreshed_facts, selected_path):
        execution = execution_capability_facts(target)
    performance_doc = {"target_profile": target_profile, "execution_capabilities": execution}
    performance_facts = {
        "target": target,
        "traits": target_profile["traits"],
        "execution_capabilities": execution,
        "target_profile_sha256": _canonical_digest(target_profile),
        "execution_capabilities_sha256": _canonical_digest(execution),
        "sha256": _canonical_digest(performance_doc),
        "raw_facts_sha256": _digest(raw_facts) if raw_facts is not None else None,
    }
    # These observations do not establish common elaboration or refmodel
    # equivalence. Keep absent production receipts visible in diagnostic runs.
    consistency = loaded_facts.get("source_consistency") or {}
    if inputs.get("source_bundle_path"):
        from merlin.targetgen.rtl import source_selection

        observe(Path(source_selection.__file__), "rtl-production-reader", required=True)
        try:
            production = source_selection.load_selection(inputs["source_bundle_path"], target=target)
            consistency = source_selection.production_consistency(production)
            from .elaboration_evidence import snapshot_elaboration

            snapshot_elaboration(production, consistency, observe)
            # Generic serialization is a separate consumer edge after source
            # production. Validate it rather than discard its exact parser input
            # or falsely contradict a coherent source bundle with an added edge.
            genericization = (loaded_facts.get("source_consistency") or {}).get("genericization")
            if genericization is not None:
                from merlin.common.digest import sha256_file

                if not isinstance(genericization, Mapping):
                    raise ValueError("invalid CIRCT genericization receipt")
                source, output, tool = (genericization[key] for key in ("input", "output", "tool"))
                command = genericization["command"]
                if (
                    genericization.get("kind") != "circt_generic_serialization"
                    or genericization.get("returncode") != 0
                    or not _same_selected_file(source, production["sources"]["core_hw"])
                    or output != {"path": inputs.get("generic_hw_path"), "sha256": inputs.get("generic_hw_sha256")}
                    or not isinstance(command, list)
                    or len(command) != 5
                    or command[1] != "--mlir-print-op-generic"
                    or command[3] != "-o"
                    or Path(command[0]).resolve() != Path(tool["path"]).resolve()
                    or Path(command[2]).resolve() != Path(source["path"]).resolve()
                    or Path(command[4]).resolve() != Path(output["path"]).resolve()
                    or any(sha256_file(member["path"]) != member["sha256"] for member in (source, output, tool))
                ):
                    raise ValueError("CIRCT genericization source/output/tool binding differs")
                consistency["genericization"] = copy.deepcopy(genericization)
                consistency["sources"].append({"role": "core_hw_generic", **output})
            if not _same_source_consistency(consistency, loaded_facts.get("source_consistency") or {}):
                diagnostics.append(
                    {
                        "component": "source-consistency",
                        "status": "contradiction",
                        "reason": "recorded source consistency differs from independent validation",
                    }
                )
                consistency = {"status": "unverified"}
            historical = ((production.get("diagnostics") or {}).get("authored_hierarchy") or {}).get("path")
            if historical is not None:
                observe(historical, "rtl-historical-hierarchy", required=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append(
                {
                    "component": "source-consistency",
                    "status": "contradiction",
                    "reason": f"production source validation failed: {exc}",
                }
            )
            consistency = {"status": "unverified"}
    if consistency.get("status") != "verified":
        diagnostics.append(
            {
                "component": "source-consistency",
                "status": "unknown",
                "reason": "no verified common-elaboration receipt selected",
            }
        )
    # Fresh facts are not enough: an rtl-source-audit report beside them must verify them, bound to the
    # exact facts and hardware-spec bytes selected here. And every selected application capture must
    # carry an admitted sealed-runner attestation that still matches its bytes on disk.
    from .evidence_status import AUDIT_MEMBER, capture_attestation_diagnostics, rtl_audit_diagnostic

    audit_path = Path(selected_path).with_name(AUDIT_MEMBER) if selected_path is not None else None
    audit_raw = observe(audit_path, "rtl-source-audit") if audit_path is not None and audit_path.is_file() else None
    hardware_source = (
        sources.get(Path(hardware_spec).absolute())
        if hardware_spec is not None and not isinstance(hardware_spec, Mapping)
        else None
    )
    audit_refusal = rtl_audit_diagnostic(
        audit_raw, facts_raw=raw_facts, hardware_raw=hardware_source.content if hardware_source else None
    )
    if audit_refusal is not None:
        diagnostics.append(audit_refusal)
    if application_inventory is not None:
        diagnostics.extend(
            capture_attestation_diagnostics(application_inventory.get("applications") or {}, capture_attestations)
        )
    if software_doc.get("status") != "reviewed":
        diagnostics.append(
            {
                "component": "software-review",
                "status": "unknown",
                "reason": "selected numerical software semantics are not reviewed",
            }
        )
    try:
        from merlin.targetgen.rtl.mlc_bridge import toolchain_identity

        observed_toolchain = toolchain_identity()
    except Exception as exc:  # noqa: BLE001 -- unavailable tool identity stays unknown
        observed_toolchain = {"unknown_reason": f"{type(exc).__name__}: {exc}"}
    toolchain_inputs = {
        "recorded_by_extractor": inputs.get("toolchain"),
        "observed_at_selection": observed_toolchain,
        "scope": "current tool observations do not identify the tools that produced unstamped facts",
    }
    from merlin.targetgen import aten_coverage

    # Observe the framework selected for capture, never a convenient host torch.
    # Replays use the serialized result below and do not relaunch this worker.
    framework_catalog = aten_coverage.observe_opset(python=capture_python, timeout=30)
    observe(Path(aten_coverage.__file__).with_name("_aten_opset_worker.py"), "framework-catalog-reader")
    if framework_catalog.get("status") != "available":
        diagnostics.append(
            {
                "component": "framework-catalog",
                "status": "unknown",
                "reason": "selected capture framework operator catalog is unavailable or incomplete",
            }
        )
    for field in hardware_doc.get("required_fields") or []:
        value = loaded_facts
        for part in str(field).split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is None or value == [] or value == {}:
            diagnostics.append(
                {
                    "component": "required-hardware-field",
                    "status": "unknown",
                    "reason": f"selected facts do not establish {field}",
                }
            )
    native_baselines = (
        _native_baseline_observations(
            native_qualifications, (application_inventory or {}).get("applications", {}), observe
        )
        if native_qualifications
        else {}
    )
    for source in sources.values():
        if source.path.read_bytes() != source.content:
            raise ValueError(f"evidence source changed during selection: {source.path}")
    from .evidence_status import status as evidence_status

    views = {
        "target": target,
        # Operator policy (evidence_status): verified only when nothing is on record against it.
        "status": evidence_status(diagnostics),
        "diagnostics": diagnostics,
        "descriptor": descriptor_doc,
        "hardware_spec": hardware_doc,
        "software_spec": software_doc,
        "software_spec_derivation": software_derivation,
        "instruction_semantics": instruction_semantics,
        "instruction_semantics_source_sha256": (
            _digest(instruction_semantics_source) if instruction_semantics_source is not None else None
        ),
        "datapath": datapath,
        "application_inventory": application_inventory,
        "application_inventory_identity": inventory_identity,
        "capture_execution_attestations": capture_attestations,
        "whole_program_admission": whole_program_admission,
        "frontend_traces": frontend_traces,
        "application_graphs": application_graphs,
        "framework_catalogs": framework_catalogs,
        "host_capabilities": host_capabilities,
        "native_baseline_observations": native_baselines,
        "loaded_facts": loaded_facts,
        "refreshed_facts": refreshed_facts,
        "contract": contract,
        "residual": residual,
        "target_profile": target_profile,
        "execution_capabilities": execution,
        "performance_facts": performance_facts,
        "readout_inputs": readout_inputs,
        "readout_facets": [facet.to_dict() for facet in facets],
        "readout_numerics": [facet.numerics_handoff() for facet in facets],
        "isa_taxonomy": taxonomy,
        "toolchain_inputs": toolchain_inputs,
        "framework_catalog": framework_catalog,
        "qualification_blockers": [
            "source-production consistency is checked when an explicit source bundle is selected; "
            "numerical reference-model qualification and reviewed operation support remain separate obligations"
        ],
        "quantization_snapshot": {
            "quantization_candidates": quantization_candidates,
            "numerical_semantics": copy.deepcopy(software_doc.get("numerical_semantics")),
            "readout_numerics": [facet.numerics_handoff() for facet in facets],
            "declared_units": [
                {key: copy.deepcopy(unit.get(key)) for key in ("name", "dtypes", "scaling", "requant")}
                for unit in contract.get("compute_units") or []
                if isinstance(unit, Mapping)
            ],
        },
    }
    return EvidenceSelection(
        target,
        tuple(sources.values()),
        _json(views),
        raw_facts,
        instruction_semantics_source=instruction_semantics_source,
    )


def _coverage_readme(accounting: dict, quantization: dict) -> bytes:
    """Human navigation generated from the same JSON views, never a second census."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    overall, universe = accounting["overall"], accounting["framework_universe"]
    operation_count = overall["n_mlir_operations"] if overall["n_mlir_operations"] is not None else "unavailable"
    registry_count = (
        universe["n_registered_aten_operators"]
        if universe["n_registered_aten_operators"] is not None
        else "unavailable"
    )
    inventory_identity = accounting.get("selected_inventory") or {}
    lines = [
        "# Phase 0 operation and quantization accounting",
        "",
        "Generated diagnostic views; no compiler lowering or TorchAO realization is certified.",
        "",
        f"Inventory status: **{accounting['status']}**. Normalized MLIR operations: **{operation_count}**.",
        f"Inventory binding: **{inventory_identity.get('status', 'not_available')}**; "
        f"declared roster matches: **{inventory_identity.get('declared_roster_matches', 'unknown')}**.",
        f"Selected framework catalog: **{universe['status']}**; registered ATen overloads: **{registry_count}**.",
        "",
        "Frontend counts are static captured call sites, not dynamic execution frequencies.",
        "Missing source traces remain unknown; repeated lowering provenance is never counted as a frontend call.",
        "",
        "## Combined normalized-IR split",
        "",
        "| Partition | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall["classification_counts"].items()]
    operation_breakdown: dict[tuple[str, str], int] = {}
    for application in accounting["applications"].values():
        for signature in application["signatures"]:
            key = (signature["classification"], signature["observed_signature"]["mlir_operation"])
            operation_breakdown[key] = operation_breakdown.get(key, 0) + signature["count"]
    lines += [
        "",
        "## Normalized-IR operations by partition",
        "",
        "These are static occurrences from the same digest-bound accounting, not new support claims.",
        "",
        "| Partition | MLIR operation | Occurrences |",
        "| --- | --- | ---: |",
    ]
    for (partition, operation), count in sorted(
        operation_breakdown.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        lines.append(f"| {cell(partition)} | {cell(operation)} | {count} |")
    lines += [
        "",
        "## Selected hardware declaration screen",
        "",
        "This separate split exposes hardware-declared admission even when SW review is unresolved.",
        "It is not executed compiler lowering or hardware qualification.",
        "",
        "| Admission | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall["hardware_admission_counts"].items()]
    lines += [
        "",
        "## Independent host and accelerator support",
        "",
        "These declaration screens are independent; unreviewed inputs remain unknown.",
        "Neither screen implies actual dispatch or execution.",
        "A host placement request without a matching pinned host capability remains unknown.",
        "",
        "| Support partition | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall.get("support_partition_counts", {}).items()]
    lines += [
        "",
        "## Per-application split",
        "",
        "| Application | Role / scope | Original calls | Quantized calls | Prepared calls | "
        "MLIR ops | Source correspondence |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, application in accounting["applications"].items():
        source = (application.get("completeness") or {}).get("source_trace") or {}
        identity = application.get("workload_identity") or {}
        counts = [source.get(f"{stage}_invocation_count") for stage in ("original", "quantized", "prepared")]
        rendered = " | ".join("unknown" if value is None else str(value) for value in counts)
        scope = f"{identity.get('workload_role', 'unknown')} / {identity.get('coverage_scope', 'unknown')}"
        lines.append(
            f"| {cell(name)} | {cell(scope)} | {rendered} | {application['n_mlir_operations']} | "
            f"{cell(source.get('status', 'unknown'))} |"
        )
    lines += [
        "",
        "## Precision and typed-edge obligations",
        "",
        "`operation-accounting.json` retains ordered storage, compute, accumulator and output types",
        "for each exact operation ordinal, plus independent host/accelerator signature decisions.",
        "`completeness.transfer_obligations` lists typed SSA edges with conditional lane crossings;",
        "no crossing is claimed until placement is selected and a reviewed transfer contract matches.",
        "Inspect [frontend/index.json](../software/frontend/index.json) for exact trace and SSA graph files.",
        "Unknown lineage, precision, capability, transfer or exact capsule coverage blocks verified",
        "whole-workload admission. Unit tests and representative subsets do not certify a headline model.",
    ]
    native = {
        label: app["native_baseline_observation"]
        for label, app in accounting["applications"].items()
        if "native_baseline_observation" in app
    }
    if native:
        lines += [
            "",
            "## Selected finite native baselines",
            "",
            "These complete-program CPU checks do not change RVV host or accelerator admission.",
            "",
            "| Application | Input precisions | Maximum absolute error | Target executed |",
            "| --- | --- | ---: | --- |",
        ]
        for label, observation in native.items():
            precision = ", ".join(sorted({item["dtype"] for item in observation["input_abi"]}))
            lines.append(f"| {cell(label)} | {cell(precision)} | {observation['max_absolute_error']} | false |")
        lines += [
            "",
            "Inspect [native-baseline-observations.json](../software/native-baseline-observations.json)",
            "for exact ABIs, compiler identities, policies and links to byte-identical frozen receipts and outputs.",
            "Agreement is scoped to the saved inputs; individual-operation execution is not independently traced.",
        ]
    lines += [
        "",
        "## Format and operation decisions",
        "",
        "| Format | Hardware status | Operation decisions |",
        "| --- | --- | --- |",
    ]
    for row in quantization["formats"]:
        decisions = ", ".join(f"{entry['operation_id']}: {entry['status']}" for entry in row["operation_eligibility"])
        lines.append(f"| {cell(row['id'])} | {cell(row['status'])} | {cell(decisions)} |")
    lines += [
        "",
        "Inspect [operation-accounting.json](operation-accounting.json) for exact signature ordinals,",
        "per-application provenance groups, registry sets and unobserved declarations.",
        "Inspect [quantization-contract.json](../software/quantization-contract.json) for parameters,",
        "format-specific recipes, conflicts and reasons each operation is unknown or ineligible.",
        "The selected detailed inventory is copied as `application-inventory.json` when available.",
        "",
    ]
    return "\n".join(lines).encode()


def export_evidence(selection: EvidenceSelection, artifact_root: str | Path) -> dict:
    """Write only already-observed bytes beneath the explicit Phase 0 run root."""
    root = Path(artifact_root)
    if selection.archived_artifacts:
        # A newer implementation must not rewrite historical reports or widen
        # their qualification. Re-export the already validated original bytes.
        outputs = dict(selection.archived_artifacts)
        manifest = json.loads(outputs["evidence-manifest.json"])
        _materialize_evidence(root, outputs)
        return manifest
    outputs: dict[str, bytes] = {}
    outputs["software/selection.json"] = selection.views_json
    if selection.raw_facts is not None:
        outputs["hardware/circt/facts.json"] = selection.raw_facts
    hardware_views = {
        "loaded-facts": selection.loaded_facts,
        "refreshed-facts": selection.refreshed_facts,
        "target-profile": selection.target_profile,
        "execution-capabilities": selection.execution_capabilities,
        "performance-facts": selection.performance_facts,
        "readout-inputs": selection.readout_inputs,
        "readout-facets": selection.readout_facets,
        "readout-numerics": selection.readout_numerics,
        "isa-taxonomy": selection.isa_taxonomy,
        "toolchain-inputs": selection.toolchain_inputs,
        "quantization": selection.quantization_snapshot,
    }
    for name, value in hardware_views.items():
        outputs[f"hardware/effective-views/{name}.json"] = _json(value)
    for name in ("contract", "residual", "hardware_spec", "software_spec", "datapath", "diagnostics"):
        outputs[f"software/{name.replace('_', '-')}.json"] = _json(getattr(selection, name))
    outputs["software/instruction-semantics.json"] = _json(selection.instruction_semantics)
    if selection.instruction_semantics_source is not None:
        outputs["software/instruction-semantics-authored.yaml"] = selection.instruction_semantics_source
    from merlin.targetgen.operation_accounting import build_operation_accounting
    from merlin.targetgen.quantization_spec import build_quantization_contract, capture_recipe_candidates

    accounting = build_operation_accounting(
        getattr(selection, "application_inventory", None),
        selection.software_spec or None,
        capability_contract=selection.contract,
        framework_catalog=getattr(selection, "framework_catalog", None),
        framework_catalogs=getattr(selection, "framework_catalogs", None),
        frontend_traces=getattr(selection, "frontend_traces", None),
        application_graphs=getattr(selection, "application_graphs", None),
        host_capabilities=getattr(selection, "host_capabilities", None),
        capture_execution_attestations=getattr(selection, "capture_execution_attestations", None),
    )
    accounting["selected_inventory"] = getattr(
        selection,
        "application_inventory_identity",
        {
            "status": "not_available",
            "reason": "snapshot predates selected application inventory",
        },
    )
    native_baselines = getattr(selection, "native_baseline_observations", {})
    for label, observation in native_baselines.items():
        destination = f"software/native-baselines/{_digest(label.encode())}"
        observation["frozen_artifact_root"] = destination
        original_root = Path(observation["source"]).parent
        for source in selection.source_snapshots:
            if source.role == f"native-qualification:{label}":
                outputs[f"{destination}/qualification.json"] = source.content
            elif source.role == f"native-artifact:{label}":
                outputs[f"{destination}/{source.path.relative_to(original_root).as_posix()}"] = source.content
        application = accounting["applications"][label]
        application["native_baseline_observation"] = copy.deepcopy(observation)
        application["precision_execution"] = {
            "status": observation["status"],
            "executor": "native_cpu",
            "target_executed": False,
            "input_abi": observation["input_abi"],
            "output_abi": observation["output_abi"],
            "scope": "whole saved program; individual operations and RVV board support remain unqualified",
        }
    quantization = build_quantization_contract(
        selection.software_spec,
        {
            "contract": selection.contract,
            "quantization_candidates": selection.quantization_snapshot.get("quantization_candidates", []),
            "readout_facets": selection.readout_facets,
            "readout_numerics": selection.readout_numerics,
        },
        accounting,
    )
    outputs["coverage/operation-accounting.json"] = _json(accounting)
    outputs["software/quantization-contract.json"] = _json(quantization)
    recipes = []
    for candidate in capture_recipe_candidates(selection.software_spec, quantization):
        payload = _json(candidate["recipe"])
        path = f"software/quantization-recipes/{_digest(payload)}.json"
        outputs[path] = payload
        recipes.append(
            {key: value for key, value in candidate.items() if key != "recipe"}
            | {"path": path, "sha256": _digest(payload), "recipe_sha256": candidate["recipe"]["recipe_sha256"]}
        )
    outputs["software/quantization-recipes.json"] = _json(
        {
            "schema": "merlin.phase0.capture_recipes.v1",
            "target": selection.target,
            "software_review": selection.software_spec.get("status"),
            "recipes": recipes,
            "qualification": "diagnostic transformation inputs; calibrate and verify actual precision before admission",
        }
    )
    outputs["software/host-capabilities.json"] = _json(getattr(selection, "host_capabilities", {}))
    outputs["software/native-baseline-observations.json"] = _json(native_baselines)
    for label, graph in sorted(getattr(selection, "application_graphs", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"coverage/application-graphs/{identity}.json"] = _json(graph)
    # Application labels are metadata, never paths: no traversal or collisions.
    for label, trace in sorted(getattr(selection, "frontend_traces", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"software/frontend/{identity}/frontend-trace.json"] = _json(trace)
    for label, catalog in sorted(getattr(selection, "framework_catalogs", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"software/frontend/{identity}/pytorch-opset.json"] = _json(catalog)
    outputs["software/frontend/index.json"] = _json(
        {
            "schema": "merlin.phase0.frontend_index.v1",
            "applications": {
                label: {
                    "frontend_trace": f"software/frontend/{_digest(label.encode())}/frontend-trace.json"
                    if label in getattr(selection, "frontend_traces", {})
                    else None,
                    "application_graph": f"coverage/application-graphs/{_digest(label.encode())}.json"
                    if label in getattr(selection, "application_graphs", {})
                    else None,
                    "framework_catalog": f"software/frontend/{_digest(label.encode())}/pytorch-opset.json"
                    if label in getattr(selection, "framework_catalogs", {})
                    else None,
                }
                for label in sorted((getattr(selection, "application_inventory", None) or {}).get("applications", {}))
            },
        }
    )
    outputs["coverage/README.md"] = _coverage_readme(accounting, quantization)
    outputs["software/framework/pytorch-opset.json"] = _json(
        getattr(
            selection,
            "framework_catalog",
            {
                "schema": "merlin.pytorch_opset.v1",
                "status": "not_available",
                "reason": "snapshot predates framework catalog observation",
            },
        )
    )
    for source in selection.source_snapshots:
        if source.role == "application-inventory":
            outputs["coverage/application-inventory.json"] = source.content
    source_index = []
    for index, source in enumerate(selection.source_snapshots):
        destination = f"software/source-snapshots/{index:04d}-{source.sha256}.bin"
        outputs[destination] = source.content
        source_index.append(
            {
                "source": str(source.path),
                "role": source.role,
                "path": destination,
                "sha256": source.sha256,
                "size_bytes": len(source.content),
            }
        )
    manifest = {
        "schema": "phase0_evidence_v1",
        "target": selection.target,
        "status": selection.status,
        "raw_facts_sha256": selection.raw_facts_sha256,
        "target_profile_sha256": selection.performance_facts["target_profile_sha256"],
        "performance_facts_sha256": selection.performance_facts["sha256"],
        "diagnostics": selection.diagnostics,
        "sources": source_index,
        "qualification_blockers": selection.qualification_blockers,
        "artifacts": {name: {"sha256": _digest(raw), "size_bytes": len(raw)} for name, raw in outputs.items()},
        "consumers": {
            "corpus_binding": [
                "software/contract.json",
                "hardware/effective-views/loaded-facts.json",
                "software/datapath.json",
                "hardware/effective-views/isa-taxonomy.json",
            ],
            "performance_gates": ["hardware/effective-views/performance-facts.json"],
            "memory_regime_axes": ["hardware/effective-views/refreshed-facts.json"],
            "readout": [
                "hardware/effective-views/refreshed-facts.json",
                "hardware/effective-views/readout-inputs.json",
            ],
            "operation_accounting": [
                "software/software-spec.json",
                "software/contract.json",
                "software/framework/pytorch-opset.json",
                "coverage/operation-accounting.json",
                "software/host-capabilities.json",
                "software/native-baseline-observations.json",
                "software/frontend/index.json",
                *sorted(
                    name
                    for name in outputs
                    if name.startswith(("coverage/application-graphs/", "software/frontend/"))
                    and name != "software/frontend/index.json"
                ),
                *(["coverage/application-inventory.json"] if "coverage/application-inventory.json" in outputs else []),
            ],
            "quantization": [
                "software/datapath.json",
                "hardware/effective-views/quantization.json",
                "software/quantization-contract.json",
                "software/quantization-recipes.json",
                *sorted(name for name in outputs if name.startswith("software/quantization-recipes/")),
                "coverage/operation-accounting.json",
            ],
            "instruction_selection": [
                "software/software-spec.json",
                "software/contract.json",
                "software/instruction-semantics.json",
                *(
                    ["software/instruction-semantics-authored.yaml"]
                    if selection.instruction_semantics_source is not None
                    else []
                ),
                *(["hardware/circt/facts.json"] if selection.raw_facts is not None else []),
            ],
        },
        "qualification": "byte snapshots and declared/derived views only; no compiler or hardware verdict",
    }
    outputs["evidence-manifest.json"] = _json(manifest)
    _materialize_evidence(root, outputs)
    return manifest


def _materialize_evidence(root: Path, outputs: dict[str, bytes]) -> None:
    """Validate all destinations before adding immutable saved evidence members."""
    for name in outputs:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"evidence output may not be symlinked: {path}")
        if path.exists() and path.read_bytes() != outputs[name]:
            raise FileExistsError(f"changed evidence output already exists: {path}")
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"evidence output traverses a symlink: {path}")
    for name, raw in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        with path.open("xb") as handle:
            handle.write(raw)


def load_exported_evidence(artifact_root: str | Path) -> EvidenceSelection:
    """Verify and reopen saved views without reading any original source path."""
    root = Path(artifact_root).absolute()
    manifest_path = root / "evidence-manifest.json"
    if manifest_path.is_symlink() or any(parent.is_symlink() for parent in manifest_path.parents):
        raise ValueError("evidence manifest traverses a symlink")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "phase0_evidence_v1":
        raise ValueError("unsupported Phase 0 evidence manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("evidence manifest requires an artifact inventory")
    observed = {}
    for name, record in artifacts.items():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts or not member.parts:
            raise ValueError(f"invalid evidence member: {name}")
        path = root / member
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"evidence member traverses a symlink: {name}")
        raw = path.read_bytes()
        if not isinstance(record, dict) or record.get("sha256") != _digest(raw) or record.get("size_bytes") != len(raw):
            raise ValueError(f"evidence member changed: {name}")
        observed[name] = raw
    views = observed.get("software/selection.json")
    if views is None:
        raise ValueError("evidence selection snapshot is missing")
    document = json.loads(views)
    if document.get("target") != manifest.get("target"):
        raise ValueError("evidence selection target differs from manifest")
    snapshots = []
    for source in manifest.get("sources") or []:
        raw = observed.get(source.get("path"))
        if raw is None or source.get("sha256") != _digest(raw) or source.get("size_bytes") != len(raw):
            raise ValueError("evidence source snapshot changed or is missing")
        snapshots.append(EvidenceSource(Path(source["source"]), source["role"], raw))
    selection = EvidenceSelection(
        manifest["target"],
        tuple(snapshots),
        views,
        observed.get("hardware/circt/facts.json"),
        tuple(sorted({**observed, "evidence-manifest.json": manifest_raw}.items())),
        observed.get("software/instruction-semantics-authored.yaml"),
    )
    if selection.raw_facts_sha256 != manifest.get("raw_facts_sha256"):
        raise ValueError("raw facts identity differs from evidence manifest")
    return selection
