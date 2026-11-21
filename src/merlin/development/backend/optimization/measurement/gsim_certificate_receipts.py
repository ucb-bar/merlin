"""Produce certificates consumed by ``merlin.development.backend.optimization.measurement.gsim_gate``.

Legacy cross-validation JSON is intentionally not upgraded: it did not record the shared ELF digest or
the simulator/model/FIRRTL identities.  Cases must be captured again through :func:`capture_case`, which
builds one ELF and passes that same path to both registered backend engines.  Certificate production is
then a pure validation/assembly step over those v1 captures and an independently sealed model-build
receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import gsim_workload as WORKLOAD

# ---- gsim_certificate --------------------------------------------------------------------------

CAPTURE_SCHEMA = "merlin.gsim-xval-case.v1"

MODEL_MANIFEST_SCHEMA = "merlin.gsim-generated-model.v1"

BUILD_RECEIPT_SCHEMA_V2 = "merlin.gsim-model-build.v2"

BUILD_RECEIPT_SCHEMA = "merlin.gsim-model-build.v3"

FIRRTL_BOUNDARY_ELABORATED = "elaborated_in_build"

FIRRTL_BOUNDARY_ADOPTED = "adopted_preexisting"

ADOPTED_FIRRTL_WARNING = (
    "FIRRTL was adopted as a pre-existing byte-pinned input; this receipt does not establish "
    "the RTL revision or command that originally elaborated it."
)

REFUSAL_SCHEMA = "merlin.gsim-certificate-refusal.v1"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _document_sha(value: Any) -> str:
    return WORKLOAD._sha_bytes(GATE.canonical_json(value).encode("utf-8"))


def _artifact_pin(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise WORKLOAD.ProducerError(f"build input is not a regular non-symlink file: {resolved}")
    return {"path": str(resolved), "sha256": _sha_file(resolved), "n_bytes": resolved.stat().st_size}


def build_model_manifest(model_root: str | Path, relative_files: Sequence[str]) -> dict[str, Any]:
    """Content-address an explicit, deterministic set of generated model sources."""
    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise WORKLOAD.ProducerError(f"generated-model root is not a directory: {root}")
    names = sorted(set(relative_files))
    if not names or len(names) != len(relative_files):
        raise WORKLOAD.ProducerError("generated-model source list is empty or contains duplicates")
    rows = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise WORKLOAD.ProducerError(f"unsafe/noncanonical generated-model path {name!r}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise WORKLOAD.ProducerError(f"generated-model source is absent or a symlink: {path}")
        rows.append({"path": name, "sha256": _sha_file(path), "n_bytes": path.stat().st_size})
    body = {"schema_version": MODEL_MANIFEST_SCHEMA, "files": rows}
    return {**body, "files_sha256": _document_sha(rows)}


def write_model_manifest(model_root: str | Path, relative_files: Sequence[str], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        GATE.canonical_json(build_model_manifest(model_root, relative_files)) + "\n", encoding="utf-8"
    )
    return output_path


def validate_model_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the sealed generated-source inventory without trusting its filename."""
    manifest_path = Path(path)
    doc = dict(WORKLOAD._load_mapping(manifest_path))
    rows = doc.get("files")
    if doc.get("schema_version") != MODEL_MANIFEST_SCHEMA or not isinstance(rows, list) or not rows:
        raise WORKLOAD.ProducerError("GSIM generated-model manifest is absent, empty, or has the wrong schema")
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise WORKLOAD.ProducerError(f"GSIM generated-model manifest row {index} is malformed")
        name, digest, n_bytes = row.get("path"), row.get("sha256"), row.get("n_bytes")
        relative = Path(str(name or ""))
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
        ):
            raise WORKLOAD.ProducerError(f"GSIM generated-model manifest row {index} has an unsafe path")
        if not GATE._is_sha256(digest) or isinstance(n_bytes, bool) or not isinstance(n_bytes, int) or n_bytes < 0:
            raise WORKLOAD.ProducerError(f"GSIM generated-model manifest row {index} lacks its byte identity")
        names.append(name)
    if names != sorted(set(names)) or doc.get("files_sha256") != _document_sha(rows):
        raise WORKLOAD.ProducerError("GSIM generated-model manifest is not canonical or its digest is invalid")
    return doc


def _firrtl_circuit(path: Path) -> str | None:
    """Read a circuit header in the first 32 lines, preserving FIRRTL identifier spelling."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for _ in range(32):
            line = stream.readline()
            if not line:
                break
            text = line.lstrip()
            keyword = "circuit"
            if not text.startswith(keyword) or len(text) == len(keyword) or not text[len(keyword)].isspace():
                continue
            declaration = text[len(keyword) :].lstrip()
            identifier, separator, _ = declaration.partition(":")
            identifier = identifier.rstrip()
            if not separator or not identifier:
                continue
            initial = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
            continuation = initial + "0123456789$"
            if identifier[0] in initial and all(character in continuation for character in identifier[1:]):
                return identifier
    return None


def model_top(path: str | Path) -> str | None:
    """Return the unique generated model top named by its sealed header, if explicit."""
    rows = validate_model_manifest(path)["files"]
    headers = [Path(row["path"]).stem for row in rows if Path(row["path"]).suffix == ".h"]
    return headers[0] if len(headers) == 1 else None


def build_receipt_document(
    *,
    firrtl: str | Path,
    model_manifest: str | Path,
    binary: str | Path,
    emitter: str | Path,
    cxx_wrapper: str | Path,
    cxx_compiler: str | Path,
    inputs: Sequence[tuple[str, str | Path]],
    commands: Sequence[Mapping[str, Any]],
    firrtl_boundary: str = FIRRTL_BOUNDARY_ELABORATED,
) -> dict[str, Any]:
    """Seal the complete, ordered native-model build lineage.

    ``inputs`` includes every source repair, harness/support source, static library, and upstream
    elaboration artifact not already covered by the generated-model manifest.  ``commands`` is the
    actual ordered build transcript: each row has ``stage``, absolute ``cwd``, and exact ``argv``.
    Merely naming one final linker command is deliberately insufficient.
    """
    firrtl_pin = _artifact_pin(firrtl)
    manifest_pin = _artifact_pin(model_manifest)
    binary_pin = _artifact_pin(binary)
    validate_model_manifest(model_manifest)
    tool_pins = {
        "gsim_emitter": _artifact_pin(emitter),
        "cxx_wrapper": _artifact_pin(cxx_wrapper),
        "cxx_compiler": _artifact_pin(cxx_compiler),
    }
    input_rows = []
    for role, input_path in inputs:
        if not isinstance(role, str) or not role.strip():
            raise WORKLOAD.ProducerError("build input role must be a non-empty string")
        input_rows.append({"role": role.strip(), **_artifact_pin(input_path)})
    if len({row["role"] for row in input_rows}) != len(input_rows):
        raise WORKLOAD.ProducerError("build input roles must be unique")
    input_rows.sort(key=lambda row: row["role"])

    command_rows = []
    for index, row in enumerate(commands):
        stage, cwd, argv = row.get("stage"), row.get("cwd"), row.get("argv")
        if not isinstance(stage, str) or not stage.strip():
            raise WORKLOAD.ProducerError(f"build command {index} has no stage")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise WORKLOAD.ProducerError(f"build command {index} cwd is not absolute")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise WORKLOAD.ProducerError(f"build command {index} has no exact argv")
        command_rows.append({"stage": stage.strip(), "cwd": cwd, "argv": list(argv)})
    stages = [row["stage"] for row in command_rows]
    if "emit" not in stages or "compile" not in stages or not stages or stages[-1] != "link":
        raise WORKLOAD.ProducerError("build transcript must contain emit, compile, and final link stages")
    if firrtl_boundary == FIRRTL_BOUNDARY_ELABORATED:
        if "elaborate" not in stages:
            raise WORKLOAD.ProducerError("an elaborated-in-build receipt must contain an elaborate command")
        provenance = {"firrtl_boundary": firrtl_boundary, "elaboration_performed": True}
    elif firrtl_boundary == FIRRTL_BOUNDARY_ADOPTED:
        if "elaborate" in stages:
            raise WORKLOAD.ProducerError("an adopted-preexisting receipt must not claim an elaborate command")
        provenance = {
            "firrtl_boundary": firrtl_boundary,
            "elaboration_performed": False,
            "warning": ADOPTED_FIRRTL_WARNING,
        }
    else:
        raise WORKLOAD.ProducerError(f"unknown FIRRTL provenance boundary {firrtl_boundary!r}")
    emit_argv0 = {row["argv"][0] for row in command_rows if row["stage"] == "emit"}
    compile_argv0 = {row["argv"][0] for row in command_rows if row["stage"] == "compile"}
    link_argv0 = {row["argv"][0] for row in command_rows if row["stage"] == "link"}
    if tool_pins["gsim_emitter"]["path"] not in emit_argv0:
        raise WORKLOAD.ProducerError("emit transcript does not invoke the pinned GSIM emitter")
    accepted_cxx = {tool_pins["cxx_wrapper"]["path"], tool_pins["cxx_compiler"]["path"]}
    if not compile_argv0.intersection(accepted_cxx) or not link_argv0.intersection(accepted_cxx):
        raise WORKLOAD.ProducerError("compile/link transcript does not invoke the pinned C++ toolchain")
    return {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "status": "complete",
        "provenance": provenance,
        "firrtl_sha256": firrtl_pin["sha256"],
        "model_manifest_sha256": manifest_pin["sha256"],
        "binary_sha256": binary_pin["sha256"],
        "artifacts": {"firrtl": firrtl_pin, "model_manifest": manifest_pin, "binary": binary_pin},
        "tools": tool_pins,
        "inputs": input_rows,
        "inputs_sha256": _document_sha(input_rows),
        "commands": command_rows,
        "commands_sha256": _document_sha(command_rows),
    }


def write_build_receipt(*, output: str | Path, **kwargs: Any) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(GATE.canonical_json(build_receipt_document(**kwargs)) + "\n", encoding="utf-8")
    return output_path


def validate_build_receipt(path: str | Path, *, pins: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Prove the sealed source manifest/FIRRTL were the inputs to the pinned GSIM binary."""
    receipt_path = Path(path)
    doc = WORKLOAD._load_mapping(receipt_path)
    schema = doc.get("schema_version")
    if schema not in {BUILD_RECEIPT_SCHEMA_V2, BUILD_RECEIPT_SCHEMA} or doc.get("status") != "complete":
        raise WORKLOAD.ProducerError("GSIM build receipt is absent, incomplete, or has the wrong schema")
    expected = {
        "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
        "model_manifest_sha256": pins["gsim_model"]["sha256"],
        "binary_sha256": pins["gsim_binary"]["sha256"],
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise WORKLOAD.ProducerError(f"GSIM build receipt {key} does not bind the pinned artifact")
    artifacts = doc.get("artifacts")
    tools, inputs, commands = doc.get("tools"), doc.get("inputs"), doc.get("commands")
    if not isinstance(artifacts, Mapping) or not isinstance(tools, Mapping):
        raise WORKLOAD.ProducerError("GSIM build receipt lacks artifact/tool pins")
    if set(tools) != {"gsim_emitter", "cxx_wrapper", "cxx_compiler"}:
        raise WORKLOAD.ProducerError("GSIM build receipt lacks an exact emitter/compiler/wrapper pin")
    if not isinstance(inputs, list) or not inputs:
        raise WORKLOAD.ProducerError("GSIM build receipt has no sealed harness/support/library inputs")
    if doc.get("inputs_sha256") != _document_sha(inputs):
        raise WORKLOAD.ProducerError("GSIM build receipt input digest is invalid")
    for where, rows in (("artifact", artifacts.values()), ("tool", tools.values()), ("input", inputs)):
        for row in rows:
            if not isinstance(row, Mapping) or not GATE._is_sha256(row.get("sha256")):
                raise WORKLOAD.ProducerError(f"GSIM build receipt has a malformed {where} pin")
            pinned_path = Path(str(row.get("path") or ""))
            if pinned_path.is_symlink() or not pinned_path.is_file() or _sha_file(pinned_path) != row["sha256"]:
                raise WORKLOAD.ProducerError(f"GSIM build receipt {where} pin is absent or changed: {pinned_path}")
    if not isinstance(commands, list) or not commands:
        raise WORKLOAD.ProducerError("GSIM build receipt has no ordered command transcript")
    stages = []
    for index, row in enumerate(commands):
        if not isinstance(row, Mapping):
            raise WORKLOAD.ProducerError(f"GSIM build receipt command {index} is malformed")
        stage, cwd, argv = row.get("stage"), row.get("cwd"), row.get("argv")
        if (
            not isinstance(stage, str)
            or not isinstance(cwd, str)
            or not Path(cwd).is_absolute()
            or not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) for arg in argv)
        ):
            raise WORKLOAD.ProducerError(f"GSIM build receipt command {index} lacks stage/cwd/exact argv")
        stages.append(stage)
    if "emit" not in stages or "compile" not in stages or stages[-1] != "link":
        raise WORKLOAD.ProducerError("GSIM build receipt command transcript is incomplete or unordered")
    if schema == BUILD_RECEIPT_SCHEMA_V2:
        if "elaborate" not in stages:
            raise WORKLOAD.ProducerError("GSIM v2 build receipt command transcript lacks elaboration")
    else:
        provenance = doc.get("provenance")
        if not isinstance(provenance, Mapping):
            raise WORKLOAD.ProducerError("GSIM v3 build receipt lacks its FIRRTL provenance boundary")
        boundary = provenance.get("firrtl_boundary")
        if boundary == FIRRTL_BOUNDARY_ELABORATED:
            if provenance.get("elaboration_performed") is not True or "elaborate" not in stages:
                raise WORKLOAD.ProducerError("GSIM elaborated-in-build receipt contradicts its command transcript")
        elif boundary == FIRRTL_BOUNDARY_ADOPTED:
            if (
                provenance.get("elaboration_performed") is not False
                or provenance.get("warning") != ADOPTED_FIRRTL_WARNING
                or "elaborate" in stages
            ):
                raise WORKLOAD.ProducerError("GSIM adopted-preexisting receipt contradicts its provenance boundary")
        else:
            raise WORKLOAD.ProducerError("GSIM v3 build receipt has an unknown FIRRTL provenance boundary")
        emit_argv0 = {row["argv"][0] for row in commands if row["stage"] == "emit"}
        compile_argv0 = {row["argv"][0] for row in commands if row["stage"] == "compile"}
        link_argv0 = {row["argv"][0] for row in commands if row["stage"] == "link"}
        if tools["gsim_emitter"]["path"] not in emit_argv0:
            raise WORKLOAD.ProducerError("GSIM v3 receipt does not invoke its pinned emitter")
        accepted_cxx = {tools["cxx_wrapper"]["path"], tools["cxx_compiler"]["path"]}
        if not compile_argv0.intersection(accepted_cxx) or not link_argv0.intersection(accepted_cxx):
            raise WORKLOAD.ProducerError("GSIM v3 receipt does not invoke its pinned C++ toolchain")
    if doc.get("commands_sha256") != _document_sha(commands):
        raise WORKLOAD.ProducerError("GSIM build receipt command transcript digest is invalid")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": _sha_file(receipt_path),
        "commands_sha256": doc["commands_sha256"],
    }


@dataclass(frozen=True)
class ArtifactPaths:
    gsim_firrtl: Path
    verilator_firrtl: Path
    gsim_model: Path
    gsim_binary: Path
    verilator_binary: Path

    def pinned(self) -> dict[str, dict[str, str]]:
        out = {}
        for name in sorted(GATE.REQUIRED_PINS):
            path = Path(getattr(self, name)).resolve(strict=True)
            if not path.is_file():
                raise WORKLOAD.ProducerError(f"{name} is not a regular file: {path}")
            if name == "gsim_model":
                validate_model_manifest(path)
            out[name] = {"path": str(path), "sha256": _sha_file(path)}
        return out


def _semantic_oracle(
    capsule_manifest: Path, command_buffer: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], Callable[[Mapping[str, Any]], bool], dict[str, Any]]:
    """Bind an ELF build and its correctness check to the grader's own numeric authority.

    Integer command-buffer programs use ``reference_outputs(command_buffer)`` as the independent
    semantic check. A complete-kernel program
    is intentionally different: either its numeric domain is not modeled by the integer command
    interpreter, or its command stream is only the accelerator projection of an explicitly declared
    ``kernel_abi.kind=whole_program`` ELF.  The formal grader therefore runs the complete ELF on the
    capsule golden's operands and compares its output to that golden.  Treating an empty or partial
    command stream as the complete computation instead prices initialized/intermediate buffers; it can
    reject a correct ELF without testing the operation that ELF implements.

    Reproduce the formal boundary here.  Golden operands are attached before the one shared ELF is
    built, positional operands are resolved by declaration order exactly as in ``capsule_runner``, and
    the capsule's declared numeric policy is used for correctness.  Verilator and GSIM must still
    produce identical declared output bytes below, so tolerance cannot conceal an engine divergence.

    This boundary is deliberately target-, operation-, and capsule-neutral.  It is selected only by
    semantic evidence already present in the capsule/command-buffer contracts.
    """
    from merlin.development.targetgen.grading import golden
    from merlin.runtime.engine.command_buffer import whole_program_entry_bindings
    from merlin.runtime.engine.reference import outputs_match, reference_outputs

    capsule = dict(WORKLOAD._load_mapping(capsule_manifest, yaml_input=True))
    capsule["__dir__"] = str(capsule_manifest.parent)
    cb = copy.deepcopy(dict(command_buffer))
    independent_float = golden.is_independent_float_golden(capsule, capsule_manifest.parent)
    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if not (independent_float or whole_program):
        expected = reference_outputs(cb)
        return (
            cb,
            expected,
            lambda observed: outputs_match(dict(observed), dict(expected)),
            {
                "kind": "command_buffer_reference",
                "golden_source": "merlin.runtime.engine.reference",
                "numeric_policy": {"compare": "exact"},
            },
        )

    canonical = golden.canonical_input_values(capsule, capsule_manifest.parent)
    operand_source = "recorded_capsule_golden"
    model_binding: dict[str, Any] = {}
    if whole_program and capsule.get("kind") == "model":
        # A model capsule's golden records only runtime inputs.  Its linalg function also has one
        # pointer argument for every externalized parameter/buffer in the sealed safetensors file.
        # Deterministic name materialization is therefore not a valid fallback: it would execute the
        # submitted program with invented weights and compare it with a golden computed from the real
        # ones.  Reuse the formal model grader's frozen-bundle adapter.  That adapter validates the
        # loader, export signature, safetensors bytes, runtime inputs, and golden against each other;
        # resolve_forward_args then reads all arguments in the exact exported function order.
        from merlin.evaluation.capsule.models.bundle import _model_runtime_bundle
        from merlin.runtime.dispatch.whole_model import resolve_forward_args

        with _model_runtime_bundle(capsule, timeout=300) as (bundle, provenance, verify_unchanged):
            arrays = resolve_forward_args(bundle)
            verify_unchanged()
        abi = cb.get("kernel_abi") or {}
        tensors = cb.get("tensors") or {}
        leaf_args = whole_program_entry_bindings(cb)
        if leaf_args is None:
            leaf_args = [
                str(arg.get("tensor"))
                for arg in (abi.get("args") or [])
                if isinstance(arg, Mapping)
                and isinstance(tensors.get(str(arg.get("tensor"))), Mapping)
                and tensors[str(arg.get("tensor"))].get("role") in ("input", "weight", "bias")
            ]
        if len(leaf_args) != len(set(leaf_args)):
            raise WORKLOAD.ProducerError("complete-model kernel ABI repeats a read-only leaf pointer")
        if len(leaf_args) != len(arrays):
            raise WORKLOAD.ProducerError(
                "complete-model exported argument count does not match its kernel ABI leaves: "
                f"bundle={len(arrays)} abi={len(leaf_args)}"
            )
        canonical = {
            name: {"shape": list(array.shape), "values": array.reshape(-1).tolist()}
            for name, array in zip(leaf_args, arrays)
        }
        operand_source = "validated_frozen_model_bundle"
        model_binding = {
            "model_source_sha256": provenance["source"]["content_sha256"],
            "model_bundle_sha256": provenance["bundle"]["content_sha256"],
            "model_bundle_construction": provenance["construction"],
            "model_bundle_validation": provenance["validation"],
        }
    if not canonical and whole_program:
        # Integer goldens are recomputed rather than shipped with recorded decoded values.  Feed the
        # ELF the exact deterministic leaves that recomputation used.  This is one shared stimulus
        # definition, not harness-side model arithmetic: the harness embeds leaves and the submitted
        # kernel alone computes every declared result.
        canonical = golden.materialized_input_values(capsule)
        operand_source = "recomputed_golden_materialization"
    tensors = cb.get("tensors")
    if not isinstance(tensors, Mapping):
        raise WORKLOAD.ProducerError("complete-kernel command buffer has no tensor declarations")
    leaves = whole_program_entry_bindings(cb)
    if leaves is None:
        leaves = [
            str(name)
            for name, spec in tensors.items()
            if isinstance(spec, Mapping) and spec.get("role") in ("input", "weight", "bias")
        ]
    overlap = set(canonical) & set(tensors)
    if set(leaves) <= set(canonical):
        bound = {name: canonical[name] for name in leaves}
        binding = "by_name"
    elif not overlap and len(leaves) == len(canonical):
        # linalg-on-tensors names ABI operands positionally (arg0, arg1, ...), while the capsule
        # records the source-level names. Both orders are the forward-function argument order.
        bound = dict(zip(leaves, canonical.values()))
        binding = "linalg_positional_declaration_order"
    else:
        raise WORKLOAD.ProducerError(
            "complete-kernel golden operands cannot be bound unambiguously to the command buffer"
        )
    if not bound or set(bound) != set(leaves):
        raise WORKLOAD.ProducerError("complete-kernel golden does not cover every command-buffer leaf")
    for name, value in bound.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("values"), list):
            raise WORKLOAD.ProducerError(f"canonical operand {name!r} has no flat value list")
        shape = (tensors[name] or {}).get("shape")
        if not isinstance(shape, list) or math.prod(shape) != len(value["values"]):
            raise WORKLOAD.ProducerError(f"canonical operand {name!r} disagrees with its command-buffer shape")
    cb["canonical_inputs"] = bound
    expected = golden.golden(capsule, capsule_manifest.parent)
    policy = dict(capsule.get("numeric_policy") or {})
    source = golden.golden_source(capsule, capsule_manifest.parent)

    def matches(observed: Mapping[str, Any]) -> bool:
        return golden.compare(dict(expected), dict(observed), policy, golden_source=source).get("status") == "pass"

    return (
        cb,
        expected,
        matches,
        {
            "kind": ("whole_program_capsule_golden" if whole_program else "independent_capsule_golden"),
            "golden_source": source,
            "numeric_policy": policy,
            "operand_binding": binding,
            "operand_source": operand_source,
            "canonical_inputs_sha256": _document_sha(bound),
            **model_binding,
        },
    )
