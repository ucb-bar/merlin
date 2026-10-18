"""Everything about one captured program that does not depend on a submission, computed once.

Device grouping is Merlin's: which source groups are routed, the interface each one is stated as, and
the host call that passes it its operands are all fixed by the capture and the target, never by the
package that lowers them. A program's catalog therefore records:

- the INT8 reference (the prepared quantized program, :func:`merlin.runtime.linalg_numpy.evaluate`d)
  and the device-rewritten program, both prepared exactly as a whole-model build prepares them;
- the rewrite's sidecar (signatures, stated entries, call roles) and every routed symbol's interface;
- the offload census over the ORIGINAL captured contractions (fixed denominator);
- per input, every device call's operands and expected results, content-addressed so a constant
  weight shared by every input and step is stored once.

A consistency check rides along: the rewritten program, with each device call answered by the
interface reference (:mod:`merlin.model_check.kernel_reference`), must reproduce the INT8 reference
bit for bit. A mismatch is a Merlin grouping/rewrite defect, reported as such and never charged to a
submission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merlin.model_check import kernel_reference as KR

if TYPE_CHECKING:
    from merlin.runtime.dispatch_runtime import ForwardArgsCache
    from merlin.runtime.linalg_numpy import ConstantCache

CATALOG_SCHEMA = "merlin.model_check.catalog.v1"
CALLS_SCHEMA = "merlin.model_check.calls.v1"
CONTRACTION_OPS = ("linalg.matmul", "linalg.batch_matmul")


class CatalogError(RuntimeError):
    """A program could not be prepared or its reference is inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    """Content-addressed arrays (``<root>/<sha256>.npy``): one copy of every distinct operand."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, array: np.ndarray) -> str:
        array = np.ascontiguousarray(array)
        header = f"{array.dtype.str}|{array.shape}".encode()
        key = _sha(header + array.tobytes())
        path = self.root / f"{key}.npy"
        if not path.exists():
            # A writer's own temporary name: several processes may store the same operand at once.
            tmp = path.with_name(f"{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
            np.save(tmp, array, allow_pickle=False)
            tmp.replace(path)
        return key

    def get(self, key: str) -> np.ndarray:
        return np.load(self.root / f"{key}.npy", allow_pickle=False)


@dataclass
class ProgramCatalog:
    """The submission-independent facts of one captured program (see the module docstring)."""

    target: str
    stage: Path
    ref_prepared: Path
    dev_prepared: Path
    routing: dict[str, Any]
    sidecar: dict[str, Any]
    interfaces: dict[str, str]

    def roles(self, symbol: str) -> tuple[str, ...]:
        return tuple(b["role"] for b in self.sidecar["call_buffers"][symbol])

    def batch_shape(self, symbol: str) -> tuple[int, ...]:
        """The leading batch of ``symbol``'s host call; its interface states ONE slice (``()``: unbatched)."""
        program = (self.sidecar.get("programs") or {}).get(symbol) or {}
        return tuple(int(v) for v in program.get("batch_shape") or ())

    def record(self) -> dict[str, Any]:
        return {
            "schema": CATALOG_SCHEMA,
            "target": self.target,
            "stage": str(self.stage),
            "ref_prepared": str(self.ref_prepared),
            "dev_prepared": str(self.dev_prepared),
            "routing": self.routing,
            "interfaces": self.interfaces,
        }

    @classmethod
    def load(cls, directory: Path) -> ProgramCatalog:
        doc = json.loads((directory / "catalog.json").read_text())
        if doc.get("schema") != CATALOG_SCHEMA:
            raise CatalogError(f"{directory}: not a {CATALOG_SCHEMA} catalog")
        sidecar = json.loads((directory / "dev" / "device_signatures.json").read_text())
        return cls(
            target=doc["target"],
            stage=Path(doc["stage"]),
            ref_prepared=Path(doc["ref_prepared"]),
            dev_prepared=Path(doc["dev_prepared"]),
            routing=doc["routing"],
            sidecar=sidecar,
            interfaces=doc["interfaces"],
        )


def prepare(stage: Path, directory: Path, *, target: str) -> ProgramCatalog:
    """Prepare ``stage`` (a capture directory with ``model.mlir``) with and without device groups."""
    from merlin.compile.complete_group_routing import routing
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    stage, directory = Path(stage), Path(directory)
    (directory / "ref").mkdir(parents=True, exist_ok=True)
    (directory / "dev").mkdir(parents=True, exist_ok=True)
    ref = _prepare_model_mlir(
        stage / "model.mlir", directory / "ref", int8_compute=True, respect_captured_quantization_scope=True
    )
    try:
        route = routing(
            target,
            (stage / "linalg.mlir").read_text(),
            str(stage),
            capture=str(stage),
            model=stage.name,
            linked_elf_admission=None,
        )
    except ValueError as error:
        if "no admitted accelerator groups" not in str(error):
            raise
        # A program with nothing to route (a pure slice, say) runs on the host only: its rewritten form IS
        # its reference, and it has no device call to judge.
        sidecar = {"signatures": {}, "entries": {}, "call_buffers": {}, "routed": [], "host_only": str(error)}
        (directory / "dev" / "device_signatures.json").write_text(json.dumps(sidecar, indent=1) + "\n")
        catalog = ProgramCatalog(
            target,
            stage,
            Path(ref),
            Path(ref),
            {"operand_dtype": None, "accum_dtype": None, "numeric_policy": None},
            sidecar,
            {},
        )
        (directory / "catalog.json").write_text(json.dumps(catalog.record(), indent=1) + "\n")
        return catalog
    dev = _prepare_model_mlir(
        stage / "model.mlir",
        directory / "dev",
        int8_compute=True,
        respect_captured_quantization_scope=True,
        complete_group_device=route,
    )
    sidecar = json.loads((directory / "dev" / "device_signatures.json").read_text())
    routing_record = {
        "operand_dtype": route.operand_dtype,
        "accum_dtype": route.accum_dtype,
        "numeric_policy": route.numeric_policy,
    }
    interfaces = {
        symbol: KR.interface_for(entry, target=target, **routing_record) for symbol, entry in sidecar["entries"].items()
    }
    catalog = ProgramCatalog(target, stage, Path(ref), Path(dev), routing_record, sidecar, interfaces)
    (directory / "catalog.json").write_text(json.dumps(catalog.record(), indent=1) + "\n")
    return catalog


def _function_arity(module) -> int:
    fn = next(op for op in module.walk() if op.name == "func.func" and op.body.blocks)
    return len(fn.body.blocks[0].args)


@dataclass
class Evaluation:
    reference: list[np.ndarray]
    rewritten: list[np.ndarray]
    calls: list[dict[str, Any]]

    def consistent(self) -> list[dict[str, Any]]:
        rows = []
        for index, (a, b) in enumerate(zip(self.reference, self.rewritten, strict=True)):
            same = a.shape == b.shape and np.array_equal(a, b)
            rows.append({"output": index, "shape": list(a.shape), "exact": bool(same)})
        return rows


def _answers(
    catalog: ProgramCatalog,
    calls: list[dict[str, Any]],
    *,
    blobs: BlobStore | None,
    keep_call: Callable[[str, int], bool] | None,
) -> dict[str, Callable]:
    """Each routed symbol's host call answered by its interface reference, recording the device calls.

    A BATCHED host call is a loop over its slices, one device call per slice: the interface states one
    slice and the host shim calls the per-slice kernel once for each (``device_shim``'s logical ABI). So
    each slice is evaluated, recorded and numbered as its own device call, and the host receives the
    slices' results stacked back under the batch.
    """
    counts: dict[str, int] = {}

    def answer(symbol: str):
        roles = catalog.roles(symbol)
        batch = catalog.batch_shape(symbol)

        def device_call(values):
            results = KR.call_outputs(catalog.interfaces[symbol], values, roles)
            ordinal = counts.get(symbol, 0)
            counts[symbol] = ordinal + 1
            if blobs is not None and (keep_call is None or keep_call(symbol, ordinal)):
                calls.append(
                    {
                        "symbol": symbol,
                        "ordinal": ordinal,
                        "roles": list(roles),
                        "operands": [blobs.put(np.asarray(v)) for v in values],
                        "expected": [blobs.put(np.asarray(r)) for r in results],
                    }
                )
            return results

        def call(values):
            if not batch:
                return device_call(values)
            count = math.prod(batch)
            arrays = [np.asarray(v) for v in values]
            if any(tuple(a.shape[: len(batch)]) != batch for a in arrays):
                raise CatalogError(f"{symbol}: a host operand does not lead with the batch {list(batch)}")
            sliced = [a.reshape(count, *a.shape[len(batch) :]) for a in arrays]
            per_slice = [device_call([a[index] for a in sliced]) for index in range(count)]
            return [
                np.stack([results[j] for results in per_slice]).reshape(*batch, *per_slice[0][j].shape)
                for j in range(len(per_slice[0]))
            ]

        return call

    return {symbol: answer(symbol) for symbol in catalog.interfaces}


def evaluate(
    catalog: ProgramCatalog,
    arguments: Sequence[np.ndarray],
    *,
    blobs: BlobStore | None = None,
    keep_call: Callable[[str, int], bool] | None = None,
) -> Evaluation:
    """Both programs on ``arguments``; every (kept) device call's operands and expected results."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime import linalg_numpy as LN

    ref = parse_mlir_file(catalog.ref_prepared)
    dev = parse_mlir_file(catalog.dev_prepared)
    reference = LN.evaluate(ref, list(arguments)[: _function_arity(ref)])
    calls: list[dict[str, Any]] = []
    rewritten = LN.evaluate(
        dev,
        list(arguments)[: _function_arity(dev)],
        calls=_answers(catalog, calls, blobs=blobs, keep_call=keep_call),
    )
    return Evaluation([np.asarray(v) for v in reference], [np.asarray(v) for v in rewritten], calls)


def capture_arguments(
    stage: Path,
    inputs: Mapping[str, np.ndarray] | None,
    scratch: Path,
    *,
    weights_cache: ForwardArgsCache | None = None,
) -> list[np.ndarray]:
    """Every ``@forward`` argument of ``stage``, with its model inputs replaced by ``inputs`` (``in0``...).

    Weights, buffers and lifted constants still come from the capture: the call binds a shadow of the
    capture directory whose files are links to the capture's, except ``inputs.npz``. With
    ``weights_cache`` they are read once per cache, and every call returns the same read-only arrays.
    """
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    stage = Path(stage)
    if inputs is None:
        return resolve_forward_args(stage, cache=weights_cache)
    shadow = Path(scratch)
    shadow.mkdir(parents=True, exist_ok=True)
    for item in stage.iterdir():
        link = shadow / item.name
        if item.name == "inputs.npz" or link.exists() or link.is_symlink():
            continue
        link.symlink_to(item.resolve())
    expected = np.load(stage / "inputs.npz")
    missing = sorted(set(expected.files) - set(inputs))
    if missing:
        raise CatalogError(f"inputs for {stage.name} lack {missing}")
    for key in expected.files:
        if tuple(np.asarray(inputs[key]).shape) != tuple(expected[key].shape):
            raise CatalogError(f"input {key}: shape {np.asarray(inputs[key]).shape} != capture {expected[key].shape}")
    np.savez(shadow / "inputs.npz", **{k: np.asarray(inputs[k]) for k in expected.files})
    return resolve_forward_args(shadow, cache=weights_cache)


def record_calls(
    catalog: ProgramCatalog,
    arguments: Sequence[np.ndarray],
    *,
    blobs: BlobStore,
    keep_call: Callable[[str, int], bool] | None = None,
    module: Any = None,
    constants: ConstantCache | None = None,
) -> list[dict[str, Any]]:
    """The device calls of the rewritten program on ``arguments`` (operands and expected results).

    ``module`` is the parsed ``dev_prepared`` when the caller keeps it; ``constants`` is passed to
    :func:`linalg_numpy.evaluate` (a device call is never hoisted, so every call is still recorded)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime import linalg_numpy as LN

    dev = module if module is not None else parse_mlir_file(catalog.dev_prepared)
    calls: list[dict[str, Any]] = []
    LN.evaluate(
        dev,
        list(arguments)[: _function_arity(dev)],
        calls=_answers(catalog, calls, blobs=blobs, keep_call=keep_call),
        constants=constants,
    )
    return calls


# ------------------------------------------------------------------------------------------- census


def _node_ids(op) -> tuple[str, ...]:
    attr = op.attributes.get("prov.source_node_ids")
    return tuple(str(getattr(item, "data", item)) for item in getattr(attr, "data", ()) or ())


def _is_contraction(op) -> bool:
    if op.name in CONTRACTION_OPS or op.name.startswith("linalg.conv_"):
        return True
    if op.name != "linalg.generic":
        return False
    iterators = str(op.properties.get("iterator_types", ""))
    if "reduction" not in iterators:
        return False
    body = [inner.name for inner in op.regions[0].blocks[0].ops]
    return any(n in ("arith.mulf", "arith.muli") for n in body) and any(n in ("arith.addf", "arith.addi") for n in body)


def _macs(op) -> int:
    shapes = [list(getattr(v.type, "get_shape", lambda: [])()) for v in op.operands]
    out = list(getattr(op.results[0].type, "get_shape", lambda: [])()) if op.results else []
    if op.name == "linalg.matmul" and len(shapes) >= 2 and len(shapes[0]) == 2:
        return int(shapes[0][0] * shapes[0][1] * shapes[1][-1])
    if op.name == "linalg.batch_matmul" and len(shapes[0]) == 3:
        return int(shapes[0][0] * shapes[0][1] * shapes[0][2] * shapes[1][-1])
    if op.name.startswith("linalg.conv_") and len(shapes) >= 2 and out:
        # Output elements times the filter's reduced extent (every filter dim but its output channels).
        filt = shapes[1]
        return int(math.prod(out)) * int(math.prod(filt)) // max(1, int(out[-1] if "hwcf" in op.name else filt[0]))
    # A generic: the size of its iteration domain, each loop dim's extent read off the operand that
    # indexes it directly (one multiply-accumulate per point of the domain).
    maps = getattr(op.properties.get("indexing_maps"), "data", ())
    extent: dict[int, int] = {}
    for amap, shape in zip(maps, shapes, strict=False):
        for axis, expr in enumerate(getattr(amap.data, "results", ())):
            position = getattr(expr, "position", None)
            if position is not None and axis < len(shape):
                extent[position] = int(shape[axis])
    return int(math.prod(extent.values())) if extent else 0


def _dtype(value) -> str:
    element = (
        getattr(getattr(value, "type", None), "element_type", None)
        or getattr(value.type, "get_element_type", lambda: None)()
    )
    return str(element)


def census(catalog: ProgramCatalog, *, refused: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every contraction of the original capture, classified ``device`` or ``host:<reason>``."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.quant_scope import captured_quantized_contraction

    module = parse_mlir_file(catalog.stage / "model.mlir")
    routed: dict[str, dict[str, Any]] = {}
    for row in catalog.sidecar.get("routed") or ():
        for node in row.get("source_node_ids") or ():
            routed[str(node)] = row
    refused = dict(refused or {})
    rows: list[dict[str, Any]] = []
    for op in module.walk():
        if not _is_contraction(op):
            continue
        nodes = _node_ids(op)
        dtypes = [_dtype(v) for v in list(op.operands)[:2]]
        row = next((routed[n] for n in nodes if n in routed), None)
        # A contraction the capture already integerized (int x int operands) was quantized by the recipe
        # even though no dequantize feeds it; left on the host, that is a policy decision, not the recipe's.
        integer = all(d[:1] in ("i", "u") and d[1:].isdigit() for d in dtypes)
        if row is not None and str(row.get("symbol")) not in refused:
            placement = "device"
        elif row is not None:
            placement = "host:compiler_refused"
        elif not (integer or captured_quantized_contraction(op)):
            placement = "host:not_quantized_by_recipe"
        else:
            placement = "host:policy_refused"
        rows.append(
            {
                "op": op.name,
                "nodes": list(nodes),
                "dtypes": dtypes,
                "macs": _macs(op),
                "placement": placement,
                "symbol": row.get("symbol") if row else None,
                "source_region_id": row.get("source_region_id") if row else None,
            }
        )
    total_macs = sum(r["macs"] for r in rows) or 1
    counts: dict[str, int] = {}
    macs: dict[str, int] = {}
    for r in rows:
        counts[r["placement"]] = counts.get(r["placement"], 0) + 1
        macs[r["placement"]] = macs.get(r["placement"], 0) + r["macs"]
    return {
        "denominator": len(rows),
        "device": counts.get("device", 0),
        "host": {k: v for k, v in sorted(counts.items()) if k != "device"},
        "device_mac_fraction": round(macs.get("device", 0) / total_macs, 6),
        "host_mac_fraction": {k: round(v / total_macs, 6) for k, v in sorted(macs.items()) if k != "device"},
        "contractions": rows,
    }


def write_calls(path: Path, calls: Sequence[Mapping[str, Any]], **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": CALLS_SCHEMA, **meta, "calls": list(calls)}, indent=1) + "\n")


def read_calls(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != CALLS_SCHEMA:
        raise CatalogError(f"{path}: not a {CALLS_SCHEMA} document")
    return doc
