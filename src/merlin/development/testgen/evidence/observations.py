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
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

# ---- evidence ----------------------------------------------------------------------------------


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

    from merlin.compiler.ir.mlir_query import forward_signature
    from merlin.development.targetgen.application_inventory import verify_capture_receipt
    from merlin.experiments import model_qualification as qualification
    from merlin.runtime.dispatch_runtime import resolve_forward_args

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
        if compiler.get("route") != "merlin.compiler.lower.model.lower_model_file:host" or _digest(
            producer_raw
        ) != producer.get("sha256"):
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

    @cached_property
    def _view_bytes(self) -> tuple[tuple[str, bytes], ...]:
        # Decode the immutable selection once. Cache only immutable field bytes,
        # so consumers still receive independently decoded mutable values.
        return tuple((key, json.dumps(value).encode()) for key, value in json.loads(self.views_json).items())

    def __getattr__(self, name: str) -> Any:
        # Every consumer receives a fresh value. The authoritative views are
        # immutable serialized bytes, not mutable objects shared with callers.
        for key, raw in self._view_bytes:
            if key == name:
                return json.loads(raw)
        raise AttributeError(name)


def _record_scaling_contradictions(
    diagnostics: list, contract: Mapping, provider_raw: bytes, *, target: str, selected_source
) -> None:
    """Compare an explicit capability contract's unit scaling with the selected provider's.

    The two declarations have different owners. A readout check against only the selected one can
    miss a stronger, incompatible scaling claim made by the executable provider, so each unit whose
    scaling differs (or which the provider lacks) is recorded; neither declaration is replaced.
    """
    provider_contract = yaml.safe_load(provider_raw)
    if not isinstance(provider_contract, dict) or provider_contract.get("name") != target:
        raise ValueError("selected support provider contract differs from selected target")

    def units(document: Mapping) -> dict:
        return {
            unit["name"]: unit
            for unit in document.get("compute_units") or ()
            if isinstance(unit, Mapping) and isinstance(unit.get("name"), str)
        }

    provider_units = units(provider_contract)
    for name, unit in sorted(units(contract).items()):
        provider_unit = provider_units.get(name)
        if provider_unit is not None and unit.get("scaling") == provider_unit.get("scaling"):
            continue
        diagnostics.append(
            {
                "component": "support-contract",
                "status": "contradiction",
                "unit": name,
                "selected_scaling": unit.get("scaling"),
                "provider_scaling": provider_unit.get("scaling") if provider_unit is not None else None,
                "reason": (
                    f"selected compute unit {name!r} has no matching support-provider unit"
                    if provider_unit is None
                    else f"selected compute unit {name!r} declares scaling {unit.get('scaling')!r}, but the "
                    f"support provider declares {provider_unit.get('scaling')!r}; review the provider against "
                    "the selected RTL readout and freeze a fresh corpus"
                ),
                "selected_contract_sha256": (
                    selected_source.sha256 if selected_source is not None else _canonical_digest(contract)
                ),
                "provider_contract_sha256": _digest(provider_raw),
            }
        )
