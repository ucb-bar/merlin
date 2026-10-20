"""Checked logical storage for the explicit standalone pointer-source diagnostic.

This binds ordered tensor storage only. It supplies no source/compiled arithmetic
correspondence, accelerator effects, timing or whole-program plan authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import yaml

from merlin.common import mlir_query
from merlin.targetgen.contract.harness_render import kernel_abi_from_commands, kernel_arg_order
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory

from .native_model_execution import NativeModelExecutionError

PROTOCOL = "merlin.native_pointer_source.v1"
BINDING_SCHEMA = "merlin.native_pointer_source_binding.v1"
MAX_PAYLOAD_BYTES = 16 << 20
MAX_ELEMENTS = 1_000_000


def _plain(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise NativeModelExecutionError("pointer source requires canonical ordinary paths")
    return path


def source_tree(root):
    """Bound the complete selected namespace before any array/archive loading."""
    root = _plain(root)
    if not root.is_dir():
        raise NativeModelExecutionError("pointer source directory is absent")
    rows, total, directories = {}, 0, 0
    for path in sorted(root.rglob("*")):
        _plain(path)
        if path.is_dir():
            directories += 1
            if directories > 64:
                raise NativeModelExecutionError("pointer source directory bound exceeded")
            continue
        if not path.is_file():
            raise NativeModelExecutionError("pointer source namespace contains special storage")
        size = path.stat().st_size
        total += size
        if len(rows) >= 64 or total > MAX_PAYLOAD_BYTES:
            raise NativeModelExecutionError("pointer source complete payload bound exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            read = 0
            while chunk := stream.read(1 << 20):
                read += len(chunk)
                if read > size:
                    raise NativeModelExecutionError("pointer source grew during bounded intake")
                digest.update(chunk)
        if read != size:
            raise NativeModelExecutionError("pointer source changed during bounded intake")
        rows[path.relative_to(root).as_posix()] = {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}
    if not rows:
        raise NativeModelExecutionError("pointer source namespace is empty")
    return rows


def _bounded_arrays(path):
    """Check complete NPY extents before NumPy can allocate a compressed payload."""
    if not path.exists():
        return
    total = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or len(names) > 64 or len(set(names)) != len(names):
            raise NativeModelExecutionError("pointer source archive membership is invalid")
        for name in names:
            if (
                not name.endswith(".npy")
                or not name[:-4]
                or not all(c.isascii() and (c.isalnum() or c == "_") for c in name[:-4])
            ):
                raise NativeModelExecutionError("pointer source archive member is not a local array")
            info = archive.getinfo(name)
            if info.file_size > MAX_PAYLOAD_BYTES:
                raise NativeModelExecutionError("pointer source decoded archive bound exceeded")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise NativeModelExecutionError("pointer source archive NPY version is unsupported")
                shape, fortran, dtype = readers[version](stream, max_header_size=4096)
                if (
                    not shape
                    or any(type(d) is not int or d < 1 for d in shape)
                    or math.prod(shape) > MAX_ELEMENTS
                    or fortran
                    or dtype.hasobject
                    or dtype.kind not in "iu"
                    or dtype.itemsize > 8
                ):
                    raise NativeModelExecutionError("pointer source decoded array shape/storage bound exceeded")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if total > MAX_PAYLOAD_BYTES or info.file_size != stream.tell() + size:
                    raise NativeModelExecutionError("pointer source complete decoded array payload bound exceeded")


def _shape(value):
    if not isinstance(value, list) or not value or any(type(d) is not int or d < 1 for d in value):
        raise NativeModelExecutionError("pointer source needs static positive tensor extents")
    if math.prod(value) > MAX_ELEMENTS:
        raise NativeModelExecutionError("pointer source tensor element bound exceeded")
    return value


def _names(values):
    if (
        not isinstance(values, list)
        or not values
        or any(type(v) is not str or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise NativeModelExecutionError("pointer source needs complete unique ordered tensor names")
    return values


def bind_pointer_source(command_buffer, capsule_dir, capture_bundle):
    """Return exact logical inputs and a fresh checked positional storage binding.

    The emitted command buffer is never changed or given a whole-program ABI.
    Initially only complete signed-i8 entries and raw-i32 results are supported;
    source constants, nontrivial storage encodings and carried mutable arguments
    need independent contracts and remain refused.
    """
    from merlin.targetgen.capsule_golden import canonical_input_values
    from merlin.targetgen.golden_store import load_golden

    root, capture = map(_plain, (capsule_dir, capture_bundle))
    before = (source_tree(root), source_tree(capture))
    cb = json.loads(json.dumps(command_buffer))
    if cb.get("kernel_abi") is not None:
        raise NativeModelExecutionError("standalone pointer source does not reinterpret a kernel_abi")
    pointer = cb.get("compiler_pointer_abi")
    if (
        type(pointer) is not dict
        or set(pointer) != {"version", "arguments"}
        or type(pointer.get("version")) is not int
        or pointer["version"] != 1
    ):
        raise NativeModelExecutionError("explicit standalone compiler pointer ABI v1 is required")
    arguments = _names(pointer["arguments"])
    tensors = cb.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(arguments):
        raise NativeModelExecutionError("standalone pointer tensors must exactly cover its arguments")
    if cb.get("params") or cb.get("outputs") is not None:
        raise NativeModelExecutionError("standalone pointer source has unsupported storage/output overrides")
    for spec in tensors.values():
        if type(spec) is not dict or set(spec) != {"shape", "dtype", "role"}:
            raise NativeModelExecutionError("standalone pointer source has an unsupported tensor storage declaration")
        _shape(spec["shape"])
        if (spec["role"], spec["dtype"]) not in (("input", "i8"), ("output", "i32")):
            raise NativeModelExecutionError(
                "standalone pointer source requires immutable i8 inputs and raw i32 outputs"
            )
    if kernel_arg_order(cb) != arguments:
        raise NativeModelExecutionError("compiler pointer order differs from the original logical harness")
    logical = kernel_abi_from_commands(cb)
    input_names = [arg["tensor"] for arg in logical["args"] if arg["access"] == "read"]
    output_names = _names(logical["outputs"])
    if not input_names or arguments != input_names + output_names:
        raise NativeModelExecutionError(
            "standalone pointer source needs ordered immutable inputs then complete outputs"
        )
    declaration = yaml.safe_load((root / "capsule.yaml").read_bytes())
    attrs = (declaration.get("operation") or {}).get("attributes") or {}
    original_inputs = declaration.get("inputs")
    if not isinstance(original_inputs, list) or len(original_inputs) != len(input_names):
        raise NativeModelExecutionError("pointer source original input roster is incomplete")
    source_inputs = _names([row.get("name") for row in original_inputs])
    source_outputs = _names(attrs.get("outs", [attrs.get("out")]))
    order = source_inputs + source_outputs
    if attrs.get("arg_order") != order or len(source_outputs) != len(output_names):
        raise NativeModelExecutionError("pointer source original ordered ABI is incomplete")
    name = declaration.get("interface_mlir")
    if type(name) is not str or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise NativeModelExecutionError("pointer source interface is not a local declared leaf")
    source = (root / name).read_bytes()
    if source != (capture / "model.mlir").read_bytes():
        raise NativeModelExecutionError("pointer source capture/runtime interfaces differ")
    module = mlir_query.parse(source.decode())
    module.verify()
    inventory = source_operation_inventory(source)
    typed_inputs, typed_outputs = inventory["arguments"], inventory["returns"]
    if len(typed_inputs) != len(input_names) or len(typed_outputs) != len(output_names):
        raise NativeModelExecutionError("pointer source tensor signature arity differs")
    for spec, typed, compiled in zip(original_inputs, typed_inputs, input_names, strict=True):
        if (
            spec.get("role") != "input"
            or spec.get("shape") != typed.get("shape")
            or spec.get("dtype") != typed.get("dtype")
            or tensors[compiled]["shape"] != typed.get("shape")
            or tensors[compiled]["dtype"] != typed.get("dtype")
        ):
            raise NativeModelExecutionError("pointer source input signature/storage differs")
    for typed, compiled in zip(typed_outputs, output_names, strict=True):
        if (
            typed.get("source") is None
            or tensors[compiled]["shape"] != typed.get("shape")
            or tensors[compiled]["dtype"] != typed.get("dtype")
        ):
            raise NativeModelExecutionError("pointer source result signature/storage differs")
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_bytes())
    captured_order = json.loads((capture / "input_order.json").read_bytes())
    if (
        type(manifest) is not dict
        or set(manifest) != {str(i) for i in range(len(input_names))}
        or type(captured_order) is not dict
        or any(type(v) is not int for v in captured_order.values())
        or sorted(captured_order.values()) != list(range(len(input_names)))
    ):
        raise NativeModelExecutionError("standalone pointer source constant-weight ownership is not supported")
    for i in range(len(input_names)):
        leaf = manifest[str(i)]
        if (
            type(leaf) is not dict
            or set(leaf) != {"kind", "name"}
            or leaf.get("kind") != "input"
            or type(leaf.get("name")) is not str
            or captured_order.get(leaf["name"]) != i
        ):
            raise NativeModelExecutionError("standalone pointer source constant-weight ownership is not supported")
    weights = (capture / "weights.safetensors").read_bytes()
    if len(weights) < 8 or int.from_bytes(weights[:8], "little") != len(weights) - 8 or json.loads(weights[8:]) != {}:
        raise NativeModelExecutionError("standalone pointer source has unbound constant tensor storage")
    _bounded_arrays(root / "golden.npz")
    _bounded_arrays(capture / "inputs.npz")
    golden = load_golden(root)
    provenance = golden.get("oracle_provenance") or {}
    if (
        provenance.get("arg_order") != order
        or provenance.get("interface") != "linalg_positional"
        or provenance.get("linalg_mlir") != name
        or set(golden.get("outputs") or {}) != set(source_outputs)
    ):
        raise NativeModelExecutionError("pointer source independent reference/provenance ABI differs")
    policy = declaration.get("numeric_policy") or {}
    for original, typed in zip(source_outputs, typed_outputs, strict=True):
        selected = policy.get("outputs", {}).get(original, policy)
        if (
            selected.get("compare") != "exact_int"
            or selected.get("dtype", "i32") != "i32"
            or list(np.asarray(golden["outputs"][original]).shape) != typed["shape"]
        ):
            raise NativeModelExecutionError("pointer source requires its unchanged exact i32 output policy")
    values = canonical_input_values(declaration, root)
    if set(values) != set(source_inputs):
        raise NativeModelExecutionError("pointer source independent input values are incomplete")
    bound, rows = {}, []
    with np.load(capture / "inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"in{i}" for i in range(len(input_names))}:
            raise NativeModelExecutionError("pointer source captured input roster differs")
        for position, (original, compiled, typed) in enumerate(
            zip(source_inputs, input_names, typed_inputs, strict=True)
        ):
            value = values[original]
            actual = np.asarray(archive[f"in{position}"])
            expected = np.asarray(value["values"])
            if (
                value.get("shape") != typed["shape"]
                or actual.dtype != np.int8
                or list(actual.shape) != typed["shape"]
                or expected.size != actual.size
                or not np.array_equal(actual.reshape(-1), expected.reshape(-1))
            ):
                raise NativeModelExecutionError(
                    "pointer source captured inputs differ from complete independent values"
                )
            bound[compiled] = {"shape": list(actual.shape), "values": actual.reshape(-1).astype(np.int64).tolist()}
            rows.append(
                dict(
                    argument_index=position,
                    source_name=original,
                    compiled_name=compiled,
                    dtype=typed["dtype"],
                    shape=typed["shape"],
                    raw_sha256=hashlib.sha256(actual.tobytes(order="C")).hexdigest(),
                )
            )
    returns = [
        dict(
            position=i,
            source_name=old,
            compiled_name=new,
            shape=typ["shape"],
            dtype=typ["dtype"],
            source_value=typ["source"],
        )
        for i, (old, new, typ) in enumerate(zip(source_outputs, output_names, typed_outputs, strict=True))
    ]
    if (source_tree(root), source_tree(capture)) != before:
        raise NativeModelExecutionError("pointer source namespace changed during exact input binding")
    return bound, dict(
        schema=BINDING_SCHEMA,
        protocol=PROTOCOL,
        source_sha256=hashlib.sha256(source).hexdigest(),
        inputs=rows,
        returns=returns,
        capsule_tree=before[0],
        capture_tree=before[1],
        authority="typed logical storage only; source/effect correspondence requires its original live owner",
    )
