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
LARGE_INTAKE_PROTOCOL = "merlin.native_pointer_source_intake.v2"
LARGE_MAX_ELEMENTS = 16_000_000
LARGE_MAX_PAYLOAD_BYTES = 256 << 20
LARGE_MAX_WORKING_BYTES = 512 << 20
CHUNK_ELEMENTS = 65536


def intake_limits(protocol):
    """Host-selected storage ceilings; no arithmetic or source-effect authority."""
    if protocol is None:
        return MAX_ELEMENTS, MAX_PAYLOAD_BYTES
    if type(protocol) is not str or protocol != LARGE_INTAKE_PROTOCOL:
        raise NativeModelExecutionError("unknown prospective pointer source intake protocol")
    return LARGE_MAX_ELEMENTS, LARGE_MAX_PAYLOAD_BYTES


def _plain(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise NativeModelExecutionError("pointer source requires canonical ordinary paths")
    return path


def source_tree(root, *, intake_protocol=None):
    """Bound the complete selected namespace before any array/archive loading."""
    _, maximum_bytes = intake_limits(intake_protocol)
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
        if len(rows) >= 64 or total > maximum_bytes:
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


def _bounded_arrays(path, *, intake_protocol=None):
    """Check complete NPY extents before NumPy can allocate a compressed payload."""
    maximum_elements, maximum_bytes = intake_limits(intake_protocol)
    if not path.exists():
        return 0
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
            if info.file_size > maximum_bytes:
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
                    or math.prod(shape) > maximum_elements
                    or fortran
                    or dtype.hasobject
                    or dtype.kind not in "iu"
                    or dtype.itemsize > 8
                ):
                    raise NativeModelExecutionError("pointer source decoded array shape/storage bound exceeded")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if total > maximum_bytes or info.file_size != stream.tell() + size:
                    raise NativeModelExecutionError("pointer source complete decoded array payload bound exceeded")
    return total


def _shape(value, *, intake_protocol=None):
    if not isinstance(value, list) or not value or any(type(d) is not int or d < 1 for d in value):
        raise NativeModelExecutionError("pointer source needs static positive tensor extents")
    maximum_elements, _ = intake_limits(intake_protocol)
    if math.prod(value) > maximum_elements:
        raise NativeModelExecutionError("pointer source tensor element bound exceeded")
    return value


def preflight_large_intake(command_buffer, capsule_dir, capture_bundle, *, intake_protocol):
    """Bound declarations and archive headers before any reference/array allocation.

    The 512MiB working allowance leaves half the selected 1GiB worker for Python,
    source owners and builds. Account conservatively for existing complete Python
    output decoding (64 bytes/cell), compact references and captured arrays.
    """
    intake_limits(intake_protocol)
    if intake_protocol is None:
        raise NativeModelExecutionError("large source intake requires explicit host selection")
    root, capture = map(_plain, (capsule_dir, capture_bundle))
    source_tree(root, intake_protocol=intake_protocol)
    source_tree(capture, intake_protocol=intake_protocol)
    parsed_metadata = {"capsule.yaml", "golden.yaml", "model.mlir", "input_order.json",
                       "weights.safetensors.manifest.json"}
    for directory in (root, capture):
        for path in directory.iterdir():
            if path.is_file() and path.name in parsed_metadata and path.stat().st_size > 4 << 20:
                raise NativeModelExecutionError("large source intake metadata exceeds its bounded compact format")
    tensors = command_buffer.get("tensors")
    if type(tensors) is not dict or not tensors or len(tensors) > 4096:
        raise NativeModelExecutionError("large source intake needs bounded complete tensor declarations")
    logical_bytes = output_elements = 0
    for tensor in tensors.values():
        from merlin.runtime.harness_render import _tensor

        try:
            _logical_shape, _dtype, extent = _tensor(tensor)
        except ValueError as error:
            raise NativeModelExecutionError("large source intake needs exact dense logical storage") from error
        shape = _shape(tensor.get("shape"), intake_protocol=intake_protocol)
        dtype = tensor.get("dtype")
        if dtype not in {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}:
            raise NativeModelExecutionError("large source intake requires exact integer tensor storage")
        logical_bytes += extent
        if tensor.get("role") == "output":
            output_elements += math.prod(shape)
    if logical_bytes > LARGE_MAX_PAYLOAD_BYTES:
        raise NativeModelExecutionError("large source complete logical storage bound exceeded")
    reference_bytes = _bounded_arrays(root / "golden.npz", intake_protocol=intake_protocol)
    capture_bytes = _bounded_arrays(capture / "inputs.npz", intake_protocol=intake_protocol)
    working_bytes = reference_bytes + capture_bytes + 64 * output_elements
    if working_bytes > LARGE_MAX_WORKING_BYTES:
        raise NativeModelExecutionError("large source intake exceeds its bounded complete-check working storage")
    return dict(protocol=intake_protocol, max_elements=LARGE_MAX_ELEMENTS,
                max_payload_bytes=LARGE_MAX_PAYLOAD_BYTES, max_working_bytes=LARGE_MAX_WORKING_BYTES,
                logical_bytes=logical_bytes, reference_bytes=reference_bytes,
                capture_bytes=capture_bytes, output_elements=output_elements,
                working_bytes=working_bytes, input_transport="streamed_exact_logical_bytes")


def _names(values):
    if (
        not isinstance(values, list)
        or not values
        or any(type(v) is not str or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise NativeModelExecutionError("pointer source needs complete unique ordered tensor names")
    return values


def bind_pointer_source(command_buffer, capsule_dir, capture_bundle, *, intake_protocol=None):
    """Return exact logical inputs and a fresh checked positional storage binding.

    The emitted command buffer is never changed or given a whole-program ABI.
    Initially only complete signed-i8 entries and raw-i32 results are supported;
    source constants, nontrivial storage encodings and carried mutable arguments
    need independent contracts and remain refused.
    """
    from merlin.targetgen.capsule_golden import canonical_input_values
    from merlin.targetgen.golden_store import load_golden

    intake_limits(intake_protocol)
    root, capture = map(_plain, (capsule_dir, capture_bundle))
    intake = (preflight_large_intake(command_buffer, root, capture, intake_protocol=intake_protocol)
              if intake_protocol is not None else None)
    before = (source_tree(root, intake_protocol=intake_protocol), source_tree(capture, intake_protocol=intake_protocol))
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
        _shape(spec["shape"], intake_protocol=intake_protocol)
        allowed = (("input", "i8"), ("output", "i32")) + ((("output", "i8"),) if intake is not None else ())
        if (spec["role"], spec["dtype"]) not in allowed:
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
    _bounded_arrays(root / "golden.npz", intake_protocol=intake_protocol)
    _bounded_arrays(capture / "inputs.npz", intake_protocol=intake_protocol)
    golden = load_golden(root, **({"compact_arrays": True} if intake is not None else {}))
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
            or selected.get("dtype", "i32") != (typed["dtype"] if intake is not None else "i32")
            or list(np.asarray(golden["outputs"][original]).shape) != typed["shape"]
        ):
            raise NativeModelExecutionError("pointer source requires its unchanged exact i32 output policy")
    if intake is None:
        values = canonical_input_values(declaration, root)
    else:
        recorded = provenance.get("inputs") or {}
        values = {name: {"shape": row.get("shape"), "values": row["decoded"]}
                  for name, row in recorded.items() if isinstance(row, dict) and "decoded" in row}
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
                or not (_equal_complete_arrays(actual, expected) if intake is not None
                        else np.array_equal(actual.reshape(-1), expected.reshape(-1)))
            ):
                raise NativeModelExecutionError(
                    "pointer source captured inputs differ from complete independent values"
                )
            if intake is None:
                bound[compiled] = {"shape": list(actual.shape), "values": actual.reshape(-1).astype(np.int64).tolist()}
            else:
                from merlin.llvmlower.weight_payload import WeightPayload
                from merlin.runtime.tensor_payload import TensorPayload

                actual.flags.writeable = False
                bound[compiled] = TensorPayload(tuple(actual.shape), "i8",
                    WeightPayload.array(actual.nbytes, lambda array=actual: array))
            rows.append(
                dict(
                    argument_index=position,
                    source_name=original,
                    compiled_name=compiled,
                    dtype=typed["dtype"],
                    shape=typed["shape"],
                    raw_sha256=hashlib.sha256(memoryview(actual).cast("B")).hexdigest(),
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
    after = (source_tree(root, intake_protocol=intake_protocol), source_tree(capture, intake_protocol=intake_protocol))
    if after != before:
        raise NativeModelExecutionError("pointer source namespace changed during exact input binding")
    return bound, dict(
        schema=BINDING_SCHEMA if intake is None else "merlin.native_pointer_source_binding.v2",
        protocol=PROTOCOL,
        source_sha256=hashlib.sha256(source).hexdigest(),
        inputs=rows,
        returns=returns,
        capsule_tree=before[0],
        capture_tree=before[1],
        **({"intake": intake} if intake is not None else {}),
        authority="typed logical storage only; source/effect correspondence requires its original live owner",
    )


def _equal_complete_arrays(actual, expected):
    if expected.dtype.kind not in "iu" or actual.size != expected.size:
        return False
    left, right = actual.reshape(-1), expected.reshape(-1)
    return all(np.array_equal(left[i:i + CHUNK_ELEMENTS], right[i:i + CHUNK_ELEMENTS])
               for i in range(0, actual.size, CHUNK_ELEMENTS))


def pointer_inputs_equal(left, right, *, intake_protocol=None):
    if intake_protocol is None:
        return left == right
    intake_limits(intake_protocol)
    if set(left) != set(right):
        return False
    for name in left:
        a, b = left[name], right[name]
        if (a.shape, a.dtype, a.byte_order) != (b.shape, b.dtype, b.byte_order):
            return False
        for value in (a, b):
            value.bind(shape=value.shape, dtype=value.dtype, byte_order=value.byte_order)
        if _payload_digest(a.payload) != _payload_digest(b.payload):
            return False
    return True


def _payload_digest(payload):
    digest = hashlib.sha256()
    for chunk in payload.chunks():
        digest.update(chunk)
    return digest.hexdigest()


def payload_matches_array(value, expected, *, shape):
    """Compare every streamed logical scalar with independent compact values."""
    from merlin.runtime.tensor_payload import TensorPayload

    if type(value) is not TensorPayload or tuple(shape) != value.shape:
        return False
    value.bind(shape=shape, dtype=value.dtype, byte_order="little")
    expected = np.asarray(expected)
    if expected.dtype.kind not in "iu" or expected.size != math.prod(shape):
        return False
    scalar = np.dtype(("<u" if value.dtype.startswith("u") else "<i") + str(int(value.dtype[1:]) // 8))
    flat, cursor = expected.reshape(-1), 0
    for chunk in value.payload.chunks(CHUNK_ELEMENTS * scalar.itemsize):
        actual = np.frombuffer(chunk, dtype=scalar)
        if not np.array_equal(actual, flat[cursor:cursor + actual.size]):
            return False
        cursor += actual.size
    return cursor == expected.size


def compare_complete_integer_outputs(expected, observed, policy, *, golden_source):
    """Use the existing exact comparison on bounded chunks of every output cell."""
    from itertools import islice

    from merlin.targetgen.capsule_golden import compare

    if set(expected) != set(observed):
        raise NativeModelExecutionError("large source exact comparison requires its complete output roster")
    if policy.get("compare") == "per_output":
        # Reuse the original complete policy-roster checks without allocating outputs.
        compare({name: [] for name in expected}, {name: [] for name in expected}, policy,
                golden_source=golden_source)

    def scalars(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "iu":
                raise NativeModelExecutionError("large source comparison requires exact integer outputs")
            yield from value.flat
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from scalars(child)
        elif type(value) is int or isinstance(value, np.integer):
            yield value
        else:
            raise NativeModelExecutionError("large source comparison requires exact integer outputs")

    result = dict(policy=policy.get("compare"), golden_source=golden_source, status="pass",
                  mismatch_count=0, max_abs_error=0, max_rel_error=0.0,
                  first_mismatch=None, per_output={})
    for name, expected_value in expected.items():
        selected = policy.get("outputs", {}).get(name, policy) if policy.get("compare") == "per_output" else policy
        if selected.get("compare") != "exact_int":
            raise NativeModelExecutionError("large source intake preserves exact integer output comparison only")
        left, right = scalars(expected_value), scalars(observed[name])
        offset = mismatches = nexp = nactual = 0
        maximum = relative = 0
        indices = []
        while True:
            a, b = [int(v) for v in islice(left, CHUNK_ELEMENTS)], [int(v) for v in islice(right, CHUNK_ELEMENTS)]
            if not a and not b:
                break
            nexp += len(a)
            nactual += len(b)
            checked = compare({name: a}, {name: b}, selected, golden_source=golden_source)
            mismatches += checked["mismatch_count"]
            maximum = max(maximum, checked["max_abs_error"])
            relative = max(relative, checked["max_rel_error"])
            first = checked.get("first_mismatch")
            if first is not None and result["first_mismatch"] is None:
                result["first_mismatch"] = {**first, "index": first["index"] + offset}
            row = checked["per_output"][name]
            indices.extend(i + offset for i in row.get("mismatch_indices", [])[:max(0, 64 - len(indices))])
            offset += len(a)
        row = dict(status="pass" if not mismatches else "fail", mismatch_count=mismatches,
                   max_abs_error=maximum, max_rel_error=relative, n_elements=nexp,
                   saturated=bool(mismatches and mismatches == nexp))
        if nexp != nactual:
            row.update(status="fail", failure_class="output_shape_mismatch", n_expected=nexp, n_observed=nactual)
        if mismatches:
            row.update(mismatch_indices=indices, mismatch_indices_truncated=mismatches > len(indices))
            result["status"] = "fail"
        result["per_output"][name] = row
        result["mismatch_count"] += mismatches
        result["max_abs_error"] = max(result["max_abs_error"], maximum)
        result["max_rel_error"] = max(result["max_rel_error"], relative)
    return result
