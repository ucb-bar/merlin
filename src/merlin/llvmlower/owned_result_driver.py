"""Generate typed invocation C and a separate checked input-byte blob.

Invocation values initialize runtime storage only. They never rewrite source
MLIR, predict result extents or enter the source-only compiler staging view.
"""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from merlin.common.paths import runtime_dir
from merlin.common.strict_json import loads
from merlin.frontends.execution_result_signature import validate_signature
from merlin.frontends.linalg_mlir import parse_mlir_text

from .execution_result_inputs import resolve
from .owned_result_abi import bridge, prepare
from .runtime_invocation import SCALAR_C

SCHEMA = "merlin.owned-result-driver.v1"


def load_arguments(source, signature, path, *, max_input_bytes=64 << 20):
    """Load the complete positional runtime roster after bounded NPY headers."""
    signature = validate_signature(parse_mlir_text(source), signature)
    if type(max_input_bytes) is not int or not 0 < max_input_bytes <= 64 << 20:
        raise ValueError("owned driver input budget must be in 1..64 MiB")
    count = len(signature["inputs"])
    if count > 64:
        raise ValueError("owned driver has too many input arguments")
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_input_bytes + 65536 * (count + 1):
        raise ValueError("owned driver input archive exceeds its bounded storage profile")
    total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != count or {row.filename for row in members} != {f"in{i}.npy" for i in range(count)}:
            raise ValueError("owned driver input archive changes its complete positional roster")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("owned driver input archive has an unsupported NPY header")
                shape, fortran, dtype = reader(stream)
                if (len(shape) > 16 or any(type(size) is not int or size < 0 for size in shape)
                        or fortran or dtype.hasobject or dtype.fields is not None
                        or dtype.itemsize not in (1, 2, 4, 8)):
                    raise ValueError("owned driver input archive requires dense native numeric storage")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if total > max_input_bytes or member.file_size != stream.tell() + size:
                    raise ValueError("owned driver input archive has an excessive or incomplete payload")
        with np.load(path, allow_pickle=False) as values:
            arrays = tuple(values[f"in{i}"] for i in range(count))
    resolve(source, signature, arrays)
    return arrays


def _record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _integer(number):
    if number == -(1 << 63):
        return "(-INT64_C(9223372036854775807)-1)"
    return f"INT64_C({number})" if number >= 0 else f"(-INT64_C({-number}))"


def _scalar_checks(index, row, constraint, dimensions):
    value, checks = f"s{index}", []
    dtype = row["dtype"]
    if "bounds" in constraint:
        lower, upper = constraint["bounds"]
        if dtype.startswith("i"):
            width = int(dtype[1:])
            low = max(math.ceil(lower), 0 if width == 1 else -(1 << (width - 1)))
            high = min(math.floor(upper), 1 if width == 1 else (1 << (width - 1)) - 1)
            if low > high:
                checks.append("1")
            else:
                checks.append(f"((int64_t){value} < {_integer(low)} || (int64_t){value} > {_integer(high)})")
        else:
            checks.append(f"!({value} >= {float(lower).hex()} && {value} <= {float(upper).hex()})")
    name = constraint.get("scalar")
    if name in dimensions:
        checks.append(f"((int64_t){value} != {_integer(dimensions[name])})")
    return checks


def _render(source, signature, arrays, *, max_input_bytes=64 << 20):
    """Emit runtime artifacts after independent source and invocation checks.

    A driver fixes input shapes but accepts different data/scalars satisfying
    its executable domain checks. Raw bytes retain NaNs/low-precision storage.
    Its selected transport is little-endian with the existing 64-bit C ABI.
    """
    if type(max_input_bytes) is not int or not 0 < max_input_bytes <= 64 << 20:
        raise ValueError("owned driver input budget must be in 1..64 MiB")
    signature = validate_signature(parse_mlir_text(source), signature)
    arrays = tuple(arrays)
    resolve(source, signature, arrays)
    _, inputs, outputs = prepare(source, entry=signature["entry"])
    if sum(value.nbytes for value in arrays) > max_input_bytes:
        raise ValueError("owned driver inputs exceed their byte budget")
    constraints = signature["input_constraints"]["inputs"]
    dimensions = {}
    for array, constraint in zip(arrays, constraints, strict=True):
        for name, actual in zip(constraint["shape"], array.shape, strict=True):
            if isinstance(name, str):
                dimensions[name] = actual
    blob, records = bytearray(), []
    declarations, arguments, observations, checks = [], [], [], []
    for index, (row, array, constraint) in enumerate(zip(signature["inputs"], arrays, constraints, strict=True)):
        if row["dtype"] == "i1" and np.any(array.view(np.uint8).reshape(-1) > 1):
            raise ValueError("owned driver Boolean storage must be canonical zero/one bytes")
        blob.extend(b"\x00" * (-len(blob) % 8))
        offset = len(blob)
        raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        blob.extend(raw if raw else b"\x00")
        if len(blob) > max_input_bytes:
            raise ValueError("owned driver aligned input storage exceeds its byte budget")
        records.append({"index": index, "kind": row["kind"], "dtype": row["dtype"],
                        "shape": list(array.shape), "offset": offset, "bytes": len(raw)})
        if row["dtype"] == "i1":
            checks.append(f"for (size_t i=0;i<{len(raw)};i++) if (base[{offset}+i]>1) "
                          "return MERLIN_OWNED_DRIVER_INPUT;")
        pointer = f"base+{offset}"
        shape = ",".join(map(str, array.shape))
        if row["kind"] == "scalar":
            declarations.append(f"{SCALAR_C[row['dtype']]} s{index}; memcpy(&s{index},{pointer},sizeof(s{index}));")
            arguments.append(f"&s{index}")
            checks.extend(f"if ({condition}) return MERLIN_OWNED_DRIVER_INPUT;"
                          for condition in _scalar_checks(index, row, constraint, dimensions))
            observations.append(f'{{"{row["dtype"]}",0,NULL,&s{index},{len(raw)}}}')
        else:
            rank = array.ndim
            if rank:
                strides = [math.prod(array.shape[axis + 1:]) for axis in range(rank)]
                if any(not 0 <= stride < 1 << 63 for stride in strides):
                    raise ValueError("owned driver contiguous strides exceed its selected index representation")
                declarations.append(f"const intptr_t shape{index}[]={{{shape}}};")
                tail = f",{{{shape}}},{{{','.join(map(str,strides))}}}"
            else:
                tail = ""
            declarations.append(f"D{rank} d{index}={{{pointer},{pointer},0{tail}}};")
            arguments.append(f"&d{index}")
            shape_pointer = "shape" + str(index) if rank else "NULL"
            observations.append(f'{{"{row["dtype"]}",{rank},{shape_pointer},{pointer},{len(raw)}}}')
    count = max(1, len(inputs))
    policy = signature["allocation_policy"]
    code = [bridge(inputs, outputs, entry=signature["entry"]), '#include "owned_result_driver.h"',
            '_Static_assert(sizeof(float)==4 && sizeof(double)==8 && sizeof(_Bool)==1,"selected scalar storage ABI");',
            '#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__',
            '#error "owned driver blob transport requires a selected little-endian C ABI"', '#endif',
            "int merlin_owned_driver_run(void *input,size_t bytes,MerlinOwnedObserver observe,void *context) {",
            f"if (bytes!={len(blob)} || ({len(blob)} && (!input || (uintptr_t)input%8))) "
            "return MERLIN_OWNED_DRIVER_INPUT;",
            "unsigned char *base=input;", *declarations, *checks,
            f"void *arguments[{count}]={{{','.join(arguments) if arguments else 'NULL'}}};",
            f"const MerlinOwnedInput observations[{count}]={{{','.join(observations) if observations else '{0}'}}};",
            'const char *types[]={' + ','.join(f'"{row["dtype"]}"' for row in signature["outputs"]) + '};',
            f"const MerlinOwnedPlan plan={{{len(inputs)},{len(outputs)},observations,types,"
            "merlin_owned_result_invoke_checked,"
            f"{policy['max_bytes']},{policy['max_allocations']},{policy['max_output_bytes']}}};",
            "return merlin_owned_driver_execute(&plan,arguments,observe,context);", "}", ""]
    c_source = "\n".join(code).encode()
    signature_bytes = (json.dumps(signature, sort_keys=True, indent=2) + "\n").encode()
    members = {"model_owned.c": c_source, "invocation.bin": bytes(blob), "signature.json": signature_bytes}
    implementations = {
        "llvmlower/" + name: Path(__file__).with_name(name)
        for name in ("owned_result_driver.py", "execution_result_inputs.py", "owned_result_abi.py",
                     "owned_result_c.py", "runtime_invocation.py", "owned_results.py")
    }
    implementations["frontends/execution_result_signature.py"] = (
        Path(__file__).parent.parent / "frontends/execution_result_signature.py")
    implementations.update({
        "runtime/" + name: runtime_dir() / name
        for name in ("c/owned_result_driver.c", "c/owned_result_driver.h",
                     "abi/owned_result_views.c", "abi/owned_result_views.h")
    })
    record = {"schema": SCHEMA, "source": _record(source.encode()), "entry": signature["entry"],
              "input_scope": "fixed-shapes-runtime-data; executable-original-scalar-constraints",
              "result_scope": "execution-observed-shapes; no-prediction", "byte_order": "little",
              "inputs": records, "input_bytes": len(blob), "max_input_bytes": max_input_bytes,
              "artifacts": {name: _record(raw) for name, raw in members.items()},
              "implementations": {name: _record(path.read_bytes()) for name, path in implementations.items()}}
    return record, members


def generate(source, signature, arrays, out, *, max_input_bytes=64 << 20):
    """Publish only a completely source/input-checked runtime selection."""
    record, members = _render(source, signature, arrays, max_input_bytes=max_input_bytes)
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, raw in members.items():
        (root / name).write_bytes(raw)
    (root / "driver.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


def verify(source, signature, arrays, root, *, max_input_bytes=64 << 20):
    """Reconstruct from actual source/inputs, never serialized foreign paths."""
    expected, members = _render(source, signature, arrays, max_input_bytes=max_input_bytes)
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir() or {p.name for p in root.iterdir()} != {*members, "driver.json"}:
        raise ValueError("owned driver has an incomplete or unknown member inventory")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in (*members, "driver.json")):
        raise ValueError("owned driver members must be ordinary files")
    if (root / "driver.json").stat().st_size > 1 << 20:
        raise ValueError("owned driver record exceeds its byte budget")
    if loads((root / "driver.json").read_bytes()) != expected:
        raise ValueError("owned driver record differs from source/input reconstruction")
    if any((root / name).stat().st_size != len(raw) or (root / name).read_bytes() != raw
           for name, raw in members.items()):
        raise ValueError("owned driver artifact differs from source/input reconstruction")
    return expected
