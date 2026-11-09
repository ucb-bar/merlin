"""Type-only scale cases from original public Phase 0 sources.

No tensor values, reference outputs, captured weights or candidate declarations
are read.  Geometric siblings are restricted to the existing generic source
builder's integer semantics. Replayed shape-homogeneous compositions retain
their full dataflow, storage epochs and outputs; arbitrary source stays exact.
Resource-frontier cases already present in the public corpus retain their full
original types.  Source construction establishes neither device residence nor
index, output-store, numerical or physical legality.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from merlin.targetgen import component_program
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

SCHEMA = "merlin.public_compile_sources.v2"
SCALES = (1, 8, 64)
MAX_SOURCE_BYTES = 256 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CASES = 256
SCALABLE_COMPONENT_OPS = frozenset({"matmul", "add", "copy", "transpose", "alias", "update", "cast"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file(path, *, limit=MAX_SOURCE_BYTES):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("public compile source must be an original ordinary file")
    if path.stat().st_size > limit:
        raise ValueError("public compile input exceeds its byte budget")
    return path.read_bytes()


@dataclass(frozen=True)
class PublicCompileCase:
    member: str
    scale: int
    source_ir: str
    original_abi: CompileOnlySourceAbi
    provenance: tuple[tuple[Path, str], ...]
    derivation_json: str

    def verify(self):
        self.original_abi.record()
        if any(hashlib.sha256(_file(path, limit=MAX_METADATA_BYTES)).hexdigest() != digest
               for path, digest in self.provenance):
            raise ValueError("original public compile source changed")
        if len(self.source_ir.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("public compile source exceeds its source-text budget")

    def record(self):
        self.verify()
        return {
            "member": self.member,
            "scale": self.scale,
            "source_sha256": hashlib.sha256(self.source_ir.encode()).hexdigest(),
            "source_bytes": len(self.source_ir.encode()),
            "original_abi": self.original_abi.record(),
            "provenance": [{"sha256": digest} for _, digest in self.provenance],
            "derivation": json.loads(self.derivation_json),
        }


def _abi(parsed, typed=None):
    if parsed.get("returns") is None or not parsed["results"]:
        raise ValueError("public compile source requires a complete ordered static tensor return")
    result = []
    for role, observed in (("inputs", parsed["args"]), ("outputs", parsed["results"])):
        slots = []
        for index, row in enumerate(observed):
            if not row["shape"] or any(type(size) is not int or size < 0 for size in row["shape"]):
                raise ValueError("public compile source has an unresolved tensor shape")
            name = typed[role][index]["name"] if typed is not None else f"{role}_{index}"
            slots.append(CompileOnlyTensor(name, tuple(row["shape"]), row["dtype"]))
        result.append(tuple(slots))
    return CompileOnlySourceAbi(*result)


def _siblings(parsed):
    """Only reconstruct a single ordinary integer contraction/copy with no captures."""
    nodes = parsed["ops"]
    if len(nodes) != 1 or len(parsed["results"]) != 1:
        return None
    node = nodes[0]
    operation = node["kind"]
    if operation not in {"linalg.matmul", "linalg.copy"}:
        return None
    inputs = parsed["args"]
    arity = 2 if operation == "linalg.matmul" else 1
    if len(inputs) != arity or len(node["ins"]) != arity or any(len(row["shape"]) != 2 for row in inputs):
        return None
    if [row["source"] for row in node["ins"]] != [("arg", i) for i in range(arity)]:
        return None
    if parsed["returns"][0]["source"] != ("op", node["id"]):
        return None
    operand, accumulator = inputs[0]["dtype"], parsed["results"][0]["dtype"]
    if operand not in {"i8", "i16", "i32", "i64"} or any(row["dtype"] != operand for row in inputs):
        return None
    if accumulator not in {"i8", "i16", "i32", "i64"}:
        return None
    m, inner = inputs[0]["shape"]
    n = inputs[1]["shape"][1] if arity == 2 else inner
    if arity == 2 and inputs[1]["shape"][0] != inner or parsed["results"][0]["shape"] != [m, n]:
        return None
    return (
        {
            "inputs": [{"name": "A", "role": "input", "shape": [m, inner], "dtype": "operand"}]
            + ([{"name": "W", "role": "weight", "shape": [inner, n], "dtype": "operand"}] if arity == 2 else []),
            "nodes": [
                {"name": "P", "op": "matmul" if arity == 2 else "copy", "inputs": ["A", "W"] if arity == 2 else ["A"]}
            ],
            "outputs": [{"name": "Y", "value": "P"}],
        },
        operand,
        accumulator,
    )


def derive(capsules, *, selected="all", scales=SCALES, index_bits=None):
    """Discover solely selected public members; inputs come from the original host.

    ``scales`` is a host preparation choice, never accepted from a broker request.
    The default fixed metadata ladder stresses code growth across source extents;
    it is unrelated to a validation model or desired schedule.  Static source
    extents are bounded by the reviewed host index width where selected.
    """
    if type(scales) is not tuple or not scales or scales[0] != 1 or any(type(s) is not int or s < 1 for s in scales):
        raise ValueError("public compile scale ladder must be an original bounded positive tuple")
    if len(scales) > 8 or len(set(scales)) != len(scales):
        raise ValueError("public compile scale ladder exceeds its metadata budget")
    if index_bits is not None and (type(index_bits) is not int or not 2 <= index_bits <= 128):
        raise ValueError("public compile cases need the original reviewed index width")
    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    rows, cases, unavailable = {}, [], []
    for capsule in capsules:
        if capsule.get("label") not in {"public", "dev"}:
            continue
        name = str(capsule.get("name") or Path(capsule["__dir__"]).name)
        if name in rows:
            raise ValueError("public compile member identity is ambiguous")
        rows[name] = capsule
    if wanted != {"all"} and (not wanted or wanted - set(rows)):
        raise ValueError("compile-only requested an unknown or non-public member")
    for name, capsule in sorted(rows.items()):
        if wanted != {"all"} and name not in wanted:
            continue
        issued = len(cases)
        try:
            root = Path(capsule["__dir__"]).absolute()
            metadata = _file(root / "capsule.yaml", limit=MAX_METADATA_BYTES)
            relative = Path(capsule.get("linalg_mlir") or "")
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("public member has no retained tensor source")
            raw = _file(root / relative)
            source = raw.decode()
            if "prov.weights_file" in source:
                raise ValueError("compile-only cannot read captured tensor values")
            parsed = parse_linalg_mlir(source)
            typed = capsule.get("component_program")
            if typed is not None:
                storage = typed["selected_storage"]
                actual, replay = component_program.render(
                    capsule["operation"]["attributes"]["program"],
                    operand_dtype=storage["operand"],
                    accumulator_dtype=storage["accumulator"],
                )
                if actual != typed or replay != source:
                    raise ValueError("public component source differs from its original semantics")
            provenance = (
                (root / "capsule.yaml", hashlib.sha256(metadata).hexdigest()),
                (root / relative, hashlib.sha256(raw).hexdigest()),
            )
            siblings = _siblings(parsed)
            if typed is not None:
                program = capsule["operation"]["attributes"]["program"]
                if program["nodes"] and all(node["op"] in SCALABLE_COMPONENT_OPS for node in program["nodes"]):
                    # The original producer replay above establishes these
                    # bodies. These operations have homogeneous shape rules;
                    # render rederives every internal shape. Fixed-size
                    # reduction axes require a separate rule.
                    siblings = program, storage["operand"], storage["accumulator"]
            for scale in scales if siblings is not None else (1,):
                if len(cases) >= MAX_CASES:
                    raise ValueError("public compile case roster exceeds its member budget")
                emitted, abi, original = source, _abi(parsed, typed), typed
                if scale != 1:
                    import copy

                    program, operand, accumulator = siblings
                    program = copy.deepcopy(program)
                    for tensor in program["inputs"]:
                        tensor["shape"] = [size * scale for size in tensor["shape"]]
                    if index_bits is not None and any(
                        size >= 1 << (index_bits - 1) for tensor in program["inputs"] for size in tensor["shape"]
                    ):
                        raise ValueError("public compile source extent exceeds reviewed host index range")
                    original, emitted = component_program.render(
                        program, operand_dtype=operand, accumulator_dtype=accumulator
                    )
                    abi = _abi(parse_linalg_mlir(emitted), original)
                if original is not None and index_bits is not None:
                    limit = 1 << (index_bits - 1)
                    tensors = original["inputs"] + original["nodes"] + original["outputs"]
                    if any(math.prod(row["shape"]) * ((int(row["dtype"][1:]) + 7) // 8) >= limit
                           for row in tensors):
                        raise ValueError("public compile tensor byte extent exceeds reviewed host index range")
                derivation = {
                    "basis": "retained_public_source" if scale == 1 else "geometric_tensor_types",
                    "extent_ratio": scale,
                    "host_index_bits": index_bits,
                    "tensor_values_allocated": False,
                    "goldens_allocated": False,
                    "resource_frontier": {
                        "sha256": _digest((capsule.get("component_coverage") or {}).get("resource_boundaries", {}))
                    }
                    if (capsule.get("component_coverage") or {}).get("resource_boundaries")
                    else None,
                    "scope": "source types only; indexing, physical resources and numerics unknown",
                }
                cases.append(
                    PublicCompileCase(name, scale, emitted, abi, provenance, json.dumps(derivation, sort_keys=True))
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            del cases[issued:]
            unavailable.append({"member": name, "status": "unavailable", "reason": type(error).__name__})
    records = [case.record() for case in cases]
    manifest = {
        "schema": SCHEMA,
        "scales": list(scales),
        "index_bits": index_bits,
        "cases": records,
        "unavailable": unavailable,
        "numeric": "not_attempted",
        "rtl": "not_attempted",
    }
    return tuple(cases), {**manifest, "sha256": _digest(manifest)}
