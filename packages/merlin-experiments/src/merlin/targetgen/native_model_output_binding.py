"""Bind original positional source results to actual compiled output names.

Only the checked current mixed-program source/return plan can justify a name
change. The independent golden and its numerical policy remain source-owned;
actual packet decoding and output hashing retain the compiler's real ABI names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


def _refuse(detail):
    from .native_model_execution import NativeModelExecutionError

    raise NativeModelExecutionError(detail)


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _refuse("source output binding requires ordinary frozen source files")
    return path.read_bytes()


def bind_source_outputs(command_buffer, capsule_dir, source_path, golden):
    """Return a fresh checked binding only when original output names differ.

    Same-name historical evidence keeps its original interpretation and checks.
    Mismatched names require the source-owned ordered declaration/provenance,
    byte-identical capsule/runtime source, full typed source plan and exact ABI.
    No candidate-supplied name-alias list is accepted.
    """
    import numpy as np
    import yaml

    compiled = (command_buffer.get("kernel_abi") or {}).get("outputs")
    outputs = golden.get("outputs") if isinstance(golden, Mapping) else None
    if isinstance(compiled, list) and isinstance(outputs, Mapping) and set(compiled) == set(outputs):
        return None
    if (
        not isinstance(compiled, list)
        or not compiled
        or any(type(name) is not str for name in compiled)
        or len(set(compiled)) != len(compiled)
        or not isinstance(outputs, Mapping)
    ):
        _refuse("independent golden outputs do not match a complete compiled output ABI")
    root = Path(capsule_dir)
    declaration_bytes = _read(root / "capsule.yaml")
    declaration = yaml.safe_load(declaration_bytes)
    if not isinstance(declaration, Mapping):
        _refuse("source output binding has no original capsule declaration")
    attributes = (declaration.get("operation") or {}).get("attributes") or {}
    names = attributes.get("outs")
    if names is None:
        names = [attributes.get("out")]
    if (
        not isinstance(names, list)
        or any(type(name) is not str or not name for name in names)
        or len(names) != len(compiled)
        or len(set(names)) != len(names)
        or set(names) != set(outputs)
    ):
        _refuse("source output binding requires the complete original ordered output declaration")
    inputs = declaration.get("inputs")
    if not isinstance(inputs, list) or any(
        not isinstance(row, Mapping) or type(row.get("name")) is not str for row in inputs
    ):
        _refuse("source output binding has no complete original input order")
    order = [row["name"] for row in inputs] + names
    provenance = golden.get("oracle_provenance") or {}
    if (
        attributes.get("arg_order") != order
        or provenance.get("arg_order") != order
        or provenance.get("interface") != "linalg_positional"
    ):
        _refuse("source output binding original declaration/provenance order differs")
    interface = declaration.get("interface_mlir") or declaration.get("linalg_mlir")
    if (
        type(interface) is not str
        or not interface
        or Path(interface).is_absolute()
        or ".." in Path(interface).parts
        or provenance.get("linalg_mlir") != interface
    ):
        _refuse("source output binding has no exact local positional interface")
    if declaration.get("linalg_mlir", interface) != interface or source_path is None:
        _refuse("source output binding interface declarations are inconsistent")
    source = _read(source_path)
    if source != _read(root / interface):
        _refuse("source output binding runtime source differs from the frozen interface")
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan

    checked = validate_mixed_program_plan(source, command_buffer)
    if checked.get("ok") is not True:
        _refuse(f"source output binding has no exact source/return plan: {checked.get('findings', [])[:2]}")
    inventory = source_operation_inventory(source)
    returned = inventory["returns"]
    if len(returned) != len(names):
        _refuse("source output binding count differs from complete source returns")
    rows = []
    for position, (name, tensor, value) in enumerate(zip(names, compiled, returned, strict=True)):
        policy = declaration.get("numeric_policy") or {}
        declared_dtype = (policy.get("outputs", {}).get(name, policy)).get("dtype")
        if (
            "shape" not in value
            or "dtype" not in value
            or value.get("source") is None
            or list(np.asarray(outputs[name]).shape) != value["shape"]
            or declared_dtype is not None
            and declared_dtype != value["dtype"]
        ):
            _refuse("source output binding original reference shape or typed return differs")
        rows.append(
            dict(
                position=position,
                source_name=name,
                compiled_name=tensor,
                shape=value["shape"],
                dtype=value["dtype"],
                source_value=value["source"],
            )
        )
    return dict(
        schema="merlin.source_output_binding.v1",
        source_sha256=hashlib.sha256(source).hexdigest(),
        capsule_declaration_sha256=hashlib.sha256(declaration_bytes).hexdigest(),
        returns=rows,
        authority="checked source return order and typed compiled ABI; names only",
    )


def source_output_values(observed, binding):
    """Rename only decoded complete actual outputs for original-policy comparison."""
    if binding is None:
        return observed
    rows = binding["returns"]
    if (
        not isinstance(observed, Mapping)
        or set(observed) != {row["compiled_name"] for row in rows}
        or len(observed) != len(rows)
    ):
        _refuse("actual compiled output roster differs from the checked source output binding")
    return {row["source_name"]: observed[row["compiled_name"]] for row in rows}


def checked_source_output_values(command_buffer, native, golden, observed):
    """Reconstruct correspondence from original pins before any saved verdict use."""
    from merlin.common.jsonio import strict_json_equal

    from .native_model_execution import _pinned_native_file

    physical = (command_buffer.get("kernel_abi") or {}).get("outputs")
    if physical is None:
        # Historical tier replay used the unchanged readback tensor roster.
        # Fresh whole-program execution still requires its explicit ABI above.
        from merlin.runtime.commandbuffer import declared_output_dtypes

        physical = list(declared_output_dtypes(command_buffer))
    names = golden.get("outputs") or {}
    expected = None
    if set(physical) != set(names):
        source = native.get("source") or {}
        capsule = _pinned_native_file(source, "capsule_declaration").parent
        interface = _pinned_native_file(source, "interface")
        expected = bind_source_outputs(command_buffer, capsule, interface, golden)
    if not strict_json_equal(native.get("source_output_binding"), expected):
        _refuse("native source output binding differs from freshly checked original evidence")
    return source_output_values(observed, expected)
