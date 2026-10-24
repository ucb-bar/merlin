"""Bind a source-identified external device catalog to whole-model call symbols.

The target package owns kernel generation and its ISA policy. This module checks the
catalog's generic dense-pointer ABI, exact source and object hashes, shape and dtype
coverage, then emits only the memref adapter needed by the host compiler.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .device_build import DeviceBuild, _flags, _nm
from .device_shim import emit_dense_translation_unit
from .toolchain import clang


def catalog_source_conversion(data: Mapping[str, object], device: str) -> tuple[str, str, str] | None:
    """Validate an explicit source-to-native dtype contract; return its source ABI.

    The target supplies the implementation and numerical proof. Core checks
    that the declaration is complete, content-bound, and points to one of the
    selected device's actual native datapaths. No target format is inferred
    from a source tensor type or from a kernel symbol.
    """
    conversion = data.get("source_conversion")
    if conversion is None:
        return None
    from merlin.system.offload import device_dtype_triples

    abi = data.get("abi") or {}
    if not isinstance(abi, Mapping):
        raise ValueError("catalog source conversion needs a dense source ABI")
    declared_source = abi.get("dtypes")
    if (
        not isinstance(conversion, Mapping)
        or conversion.get("schema") != "target_owned_dtype_conversion_v1"
        or not isinstance(declared_source, list)
        or len(declared_source) != 3
        or not all(isinstance(dtype, str) and dtype for dtype in declared_source)
        or conversion.get("source_dtypes") != declared_source
    ):
        raise ValueError("catalog source conversion disagrees with its dense ABI")
    source = tuple(declared_source)
    native = set(device_dtype_triples(device))
    allowed = conversion.get("target_dtypes")
    if (
        not native
        or not isinstance(allowed, list)
        or not allowed
        or any(
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(dtype, str) and dtype for dtype in triple)
            or tuple(triple) not in native
            for triple in allowed
        )
        or len({tuple(triple) for triple in allowed}) != len(allowed)
    ):
        raise ValueError("catalog source conversion names no distinct native target datapaths")
    policy = conversion.get("policy")
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("catalog source conversion has no explicit numeric policy")
    try:
        digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("catalog source conversion policy is not canonical JSON") from error
    if conversion.get("policy_sha256") != digest:
        raise ValueError("catalog source conversion policy hash differs from its declaration")
    bindings = data.get("bindings") or ()
    if not bindings or any(
        not isinstance(row, Mapping) or list(row.get("target_dtypes") or ()) not in allowed for row in bindings
    ):
        raise ValueError("catalog source conversion has an unbound target format")
    return source  # type: ignore[return-value]


def build_catalog_objects(
    device: str,
    signatures: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, Sequence[str]],
    *,
    manifest_path: str | Path,
    object_path: str | Path,
    source_sha256: str,
    routed: Sequence[Mapping[str, object]],
    workdir: str | Path,
    codegen_target: str = "riscv",
    cflags: Sequence[str] | None = None,
    timeout: int = 900,
) -> DeviceBuild:
    """Return a linkable catalog and shim, or raise on any incomplete binding."""
    manifest_path, object_path = Path(manifest_path), Path(object_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("external catalog manifest must be a regular file")
    if not object_path.is_file() or object_path.is_symlink():
        raise ValueError("external catalog object must be a regular file")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    conversion_source = catalog_source_conversion(data, device)
    abi = data.get("abi") or {}
    if (
        abi.get("argument_order") != ["lhs", "rhs", "out"]
        or abi.get("pointee_layout") != "dense_row_major"
        or abi.get("batch_call") != "whole_batch"
        or len(abi.get("dtypes") or ()) != 3
    ):
        raise ValueError("external catalog does not declare the dense pointer ABI")
    if data.get("source_sha256") != source_sha256:
        raise ValueError("external catalog was compiled from different prepared model bytes")
    if not data.get("coverage_complete") or data.get("covered_contractions") != data.get("matched_contractions"):
        raise ValueError("external catalog is incomplete")
    actual_sha = hashlib.sha256(object_path.read_bytes()).hexdigest()
    if data.get("compilation", {}).get("object_sha256") != actual_sha:
        raise ValueError("external catalog object hash differs from its compilation receipt")

    available: dict[str, tuple[bool, int, int, int, int]] = {}
    symbols: set[str] = set()
    for row in data.get("kernels", ()):
        dims = row["dimensions"]
        batched = row["batched"]
        key = (batched, dims["batch"], dims["m"], dims["n"], dims["k"])
        symbol = row["symbol"]
        if (
            type(batched) is not bool
            or any(type(v) is not int or v <= 0 for v in key[1:])
            or not _identifier(symbol)
            or symbol in symbols
        ):
            raise ValueError("external catalog has ambiguous or invalid kernel records")
        available[symbol] = key
        symbols.add(symbol)
    if len(symbols) != data.get("unique_kernels") or not symbols:
        raise ValueError("external catalog kernel count disagrees with its records")
    nm = _nm()
    if nm is None:
        raise ValueError("external catalog needs a symbol inspector")
    symbol_table = subprocess.run(
        [nm, "--format=posix", "--extern-only", str(object_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if symbol_table.returncode:
        raise ValueError(f"could not inspect catalog object symbols: {symbol_table.stderr[-300:]}")
    defined: dict[str, int] = {}
    unresolved: list[str] = []
    for line in symbol_table.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("unreadable catalog object symbol record")
        if fields[1].upper() == "U":
            unresolved.append(fields[0])
        elif fields[1].upper() == "T":
            defined[fields[0]] = defined.get(fields[0], 0) + 1
    if any(defined.get(symbol) != 1 for symbol in symbols) or unresolved:
        raise ValueError("catalog object is missing declared kernels or has unresolved references")

    binding_by_region: dict[str, dict] = {}
    for row in data.get("bindings", ()):
        region, symbol = row.get("region"), row.get("symbol")
        if (
            not isinstance(region, str)
            or not region
            or region in binding_by_region
            or symbol not in symbols
            or len(row.get("tensor_types") or ()) != 3
        ):
            raise ValueError("external catalog has invalid operation bindings")
        binding_by_region[region] = row
    if len(binding_by_region) != data.get("covered_contractions"):
        raise ValueError("external catalog operation count disagrees with bindings")

    expected_shapes: dict[str, tuple[bool, int, int, int, int]] = {}
    for entry, shape in signatures.items():
        if not _identifier(entry):
            raise ValueError(f"invalid offload entry symbol {entry!r}")
        dims = tuple(shape)
        if len(dims) == 3:
            m, n, k = dims
            key = (False, 1, m, n, k)
        elif len(dims) == 4:
            b, m, n, k = dims
            key = (True, b, m, n, k)
        else:
            raise ValueError(f"unsupported offload rank for {entry}")
        if tuple(dtypes.get(entry, ())) != tuple(abi["dtypes"]):
            raise ValueError(f"catalog has no proven precision for {entry}")
        expected_shapes[entry] = key
    if not expected_shapes:
        raise ValueError("external catalog has no offloaded calls to bind")
    if not routed:
        raise ValueError("external catalog needs source operation identities for routed calls")
    routed_regions: set[str] = set()
    kernels: dict[str, str] = {}
    for row in routed:
        region, entry = row.get("source_region"), row.get("symbol")
        binding = binding_by_region.get(region)
        if (
            not isinstance(region, str)
            or region in routed_regions
            or entry not in expected_shapes
            or binding is None
            or available[binding["symbol"]] != expected_shapes[entry]
            or type(binding.get("source_operation_ordinal")) is not int
            or binding["source_operation_ordinal"] != row.get("source_operation_ordinal")
            or binding.get("tensor_types") != row.get("tensor_types")
            or tuple(row.get("dtypes") or ()) != tuple(abi["dtypes"])
        ):
            raise ValueError("routed operation does not match the source catalog binding")
        if entry in kernels and kernels[entry] != binding["symbol"]:
            raise ValueError("one routed callee refers to different source-bound implementations")
        kernels[entry] = binding["symbol"]
        routed_regions.add(region)
    if set(kernels) != set(signatures) or routed_regions != set(binding_by_region):
        raise ValueError("routed operation coverage differs from catalog entries")

    unit = emit_dense_translation_unit(device, signatures, dtypes, kernel_symbol_for=kernels.__getitem__)
    if set(unit.symbols) != set(signatures) or unit.skipped:
        raise ValueError(f"external catalog shim declined routed calls: {unit.skipped}")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "device_catalog_shim.c"
    source.write_text(unit.text, encoding="utf-8")
    shim = work / "device_catalog_shim.o"
    result = subprocess.run(
        [clang(), *_flags(codegen_target, cflags), "-c", str(source), "-o", str(shim)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"external catalog shim compilation failed: {result.stderr[-600:]}")
    shim_symbols = subprocess.run(
        [nm, "--format=posix", "--extern-only", str(shim)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if shim_symbols.returncode:
        raise ValueError(f"could not inspect catalog shim symbols: {shim_symbols.stderr[-300:]}")
    shim_defined: set[str] = set()
    shim_undefined: set[str] = set()
    for line in shim_symbols.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("unreadable catalog shim symbol record")
        (shim_undefined if fields[1].upper() == "U" else shim_defined).add(fields[0])
    if not set(kernels).issubset(shim_defined) or shim_undefined != set(kernels.values()):
        raise ValueError("catalog shim entry points or kernel references disagree with routed calls")
    receipt = {
        "schema": "external_device_catalog_binding_v1",
        "source_sha256": source_sha256,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "catalog_object_sha256": actual_sha,
        "shim_object_sha256": hashlib.sha256(shim.read_bytes()).hexdigest(),
        "routed_operations": len(routed),
        "entry_to_kernel": dict(sorted(kernels.items())),
        "compiler_flags": _flags(codegen_target, cflags),
        **(
            {
                "source_conversion_policy_sha256": data["source_conversion"]["policy_sha256"],
                "source_conversion_dtypes": list(conversion_source),
            }
            if conversion_source
            else {}
        ),
    }
    (work / "catalog_binding.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return DeviceBuild(device=device, objects=(object_path, shim), shim_object=shim, kernels=kernels)


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and (value[0].isalpha() or value[0] == "_")
        and all(c.isalnum() or c == "_" for c in value)
    )
