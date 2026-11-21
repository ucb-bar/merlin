"""Frozen certificate workload identities and declared logical tensor byte encoding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import sha256_file
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.search import corpus as P2_CORPUS
from merlin.development.targetgen.execution.runner import compile as OOT
from merlin.development.targetgen.execution.runner import oracle_run_compile as _merlin_oracle_run_compile_1
from merlin.target.contract.harness import readback_policy as RB

# ---- gsim_workload -----------------------------------------------------------------------------

OUTPUT_ENCODING = GATE.OUTPUT_ENCODING


class ProducerError(RuntimeError):
    """The available bytes cannot support a strict v1 certificate."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_mapping(path: Path, *, yaml_input: bool = False) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_input else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ProducerError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProducerError(f"{path} does not contain a mapping")
    return value


def derive_workload(capsule_manifest: str | Path) -> dict[str, Any]:
    """Derive an exact workload identity only from the frozen capsule descriptor.

    Operand symbol names and source annotations are excluded: they do not alter the operation.  Shapes,
    dtypes, output semantics, epilogue, and numeric comparison policy remain load-bearing.
    """
    path = Path(capsule_manifest)
    doc = _load_mapping(path, yaml_input=True)
    operation = doc.get("operation")
    inputs = doc.get("inputs")
    numeric = doc.get("numeric_policy")
    if not isinstance(operation, Mapping) or not isinstance(inputs, list) or not inputs:
        raise ProducerError(f"{path}: operation/inputs are absent")
    op = operation.get("op")
    attrs = operation.get("attributes")
    if not isinstance(op, str) or not op or not isinstance(attrs, Mapping):
        raise ProducerError(f"{path}: operation is malformed")
    if not isinstance(numeric, Mapping) or not numeric:
        raise ProducerError(f"{path}: numeric policy is absent")

    tensors: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise ProducerError(f"{path}: input {index} is malformed")
        name, shape, dtype = item.get("name"), item.get("shape"), item.get("dtype")
        if not isinstance(name, str) or not isinstance(shape, list) or not shape or not isinstance(dtype, str):
            raise ProducerError(f"{path}: input {index} lacks name/shape/dtype")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ProducerError(f"{path}: input {name} has a non-positive/non-integer shape")
        tensors[name] = item

    semantic_attrs = {
        str(key): value for key, value in attrs.items() if key not in ("lhs", "weight", "src", "out", "semantic")
    }
    if op == "matmul":
        lhs = tensors.get(str(attrs.get("lhs") or ""))
        weight = tensors.get(str(attrs.get("weight") or ""))
        if lhs is None or weight is None:
            raise ProducerError(f"{path}: matmul operands do not resolve to declared inputs")
        lhs_shape, weight_shape = lhs["shape"], weight["shape"]
        if len(lhs_shape) != 2 or len(weight_shape) != 2 or lhs_shape[1] != weight_shape[0]:
            raise ProducerError(f"{path}: matmul shapes are not MxK and KxN")
        shape = {"m": lhs_shape[0], "n": weight_shape[1], "k": lhs_shape[1]}
        operand_dtypes = {"lhs": lhs["dtype"], "weight": weight["dtype"]}
    elif op == "movement":
        src = tensors.get(str(attrs.get("src") or ""))
        if src is None:
            raise ProducerError(f"{path}: movement source does not resolve to a declared input")
        shape = {"dimensions": list(src["shape"])}
        operand_dtypes = {"src": src["dtype"]}
    else:
        # No guessed shape algebra for an unknown operation.  Exact named input roles/shapes are still a
        # valid envelope and are derived directly from the descriptor.
        shape = {"inputs": [{"role": item.get("role"), "shape": item["shape"]} for item in inputs]}
        operand_dtypes = {str(item.get("role") or item["name"]): item["dtype"] for item in inputs}
    semantics = {
        "operand_dtypes": operand_dtypes,
        "operation_attributes": semantic_attrs,
        "numeric_policy": dict(numeric),
    }
    return GATE.canonical_workload({"operation": op, "shape": shape, "semantics": semantics})


def derive_frozen_corpus_workloads(
    root: str | Path, *, manifest_sha256: str, capsules_sha256: str, expected_target: str
) -> dict[str, dict[str, Any]]:
    """Derive every workload after the existing frozen-corpus verifier re-hashes its bytes."""
    corpus = P2_CORPUS.load_frozen_performance_corpus(
        Path(root), manifest_sha256=manifest_sha256, capsules_sha256=capsules_sha256, expected_target=expected_target
    )
    workloads = {}
    for member in corpus.capsules:
        if member.capsule in workloads:
            raise ProducerError(f"frozen corpus has duplicate capsule {member.capsule!r}")
        workloads[member.capsule] = derive_workload(member.source_dir / "capsule.yaml")
    if not workloads:
        raise ProducerError("frozen corpus contains no workload")
    return workloads


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flat_values(item))
        return out
    return [value]


def _encode_scalar(value: Any, dtype: str) -> bytes:
    if len(dtype) > 1 and dtype[0] in ("i", "u") and dtype[1:].isdecimal():
        signed, bits = dtype[0] == "i", int(dtype[1:])
        if bits == 1:
            bits = 8
        if bits <= 0 or bits % 8:
            raise ProducerError(f"output dtype {dtype!r} has no byte-exact scalar encoding")
        try:
            return int(value).to_bytes(bits // 8, byteorder="little", signed=signed)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ProducerError(f"output value {value!r} is not representable as {dtype}") from exc
    if dtype in ("f16", "float16"):
        return struct.pack("<e", float(value))
    if dtype in ("f32", "float32"):
        return struct.pack("<f", float(value))
    if dtype in ("f64", "float64"):
        return struct.pack("<d", float(value))
    if dtype in ("bf16", "bfloat16"):
        # Round a parsed real value to IEEE bfloat16, ties-to-even, then emit its little-endian bits.
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
        return struct.pack("<H", rounded & 0xFFFF)
    raise ProducerError(f"output dtype {dtype!r} has no declared byte encoding")


def encode_declared_outputs(outputs: Any, command_buffer: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Encode parsed tensor values as the command buffer's exact logical little-endian bytes.

    This deliberately does not hash JSON text.  Shape and dtype come from the frozen command buffer,
    the flattened element count must match exactly, and each scalar is range-checked before encoding.
    The claim is logical tensor bytes; it is not a claim about padding or an engine's private memory.
    """
    if not isinstance(outputs, Mapping):
        raise ProducerError("simulator output is not a tensor mapping")
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ProducerError("command buffer has no tensor declarations")
    declarations = []
    for name, spec in tensors.items():
        if isinstance(spec, Mapping) and spec.get("role") == "output":
            declarations.append((str(name), spec))
    if not declarations:
        raise ProducerError("command buffer declares no output tensors")
    rows, aggregate = [], hashlib.sha256()
    for name, spec in sorted(declarations):
        shape, dtype = spec.get("shape"), spec.get("dtype")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            raise ProducerError(f"output tensor {name!r} has an invalid shape")
        if not isinstance(dtype, str) or not dtype:
            raise ProducerError(f"output tensor {name!r} has no dtype")
        if name not in outputs:
            raise ProducerError(f"simulator omitted declared output tensor {name!r}")
        flat = _flat_values(outputs[name])
        count = math.prod(shape)
        if len(flat) != count:
            raise ProducerError(f"output tensor {name!r} has {len(flat)} values; declaration requires {count}")
        raw = b"".join(_encode_scalar(value, dtype) for value in flat)
        identity = GATE.canonical_json({"name": name, "shape": shape, "dtype": dtype}).encode("utf-8")
        aggregate.update(len(identity).to_bytes(8, "little"))
        aggregate.update(identity)
        aggregate.update(len(raw).to_bytes(8, "little"))
        aggregate.update(raw)
        rows.append(
            {"name": name, "shape": list(shape), "dtype": dtype, "n_bytes": len(raw), "sha256": _sha_bytes(raw)}
        )
    return aggregate.hexdigest(), rows


# ---- gsim_digest_readback ----------------------------------------------------------------------
# A Phase 2 gSIM measurement that reads its outputs back as one digest, verified by Spike.
#
# Reading a large output back over gSIM's simulated serial link costs hours: a 3136x64 i32 output
# was still printing after 2 h as text, as a binary frame, and as a coherent memory dump. The cycles of
# a measurement cell come from the ``METRIC`` window and are unaffected by how outputs come back, but
# the cell must still show that the gSIM run produced the right outputs.
#
# So the grader-rendered harness of the gSIM run prints one XXH64 per output (``out_digest_v1``, never
# candidate code), and that digest is tied to full values by these checks:
#
# 1. the program's full-value build runs on Spike, where reading values back is cheap: values ``V``;
# 2. the digest build -- the SAME kernel object behind the digest harness -- runs on Spike, and its digest
#    must be the digest of ``V``'s container bytes (the digest program holds exactly ``V`` on Spike);
# 3. the SAME ELF bytes (sha256 checked) run on gSIM, and its digest must equal the Spike digest.
#
# Then gSIM held ``V`` (up to an accidental 64-bit collision), and ``V`` is what the cell reports --
# labelled as values verified by digest rather than read from gSIM. Any disagreement falls back to the
# historical full gSIM readback, so a numerical verdict never rests on the shortcut being right.
# ``MERLIN_PHASE2_GSIM_READBACK=full`` keeps the full readback for every cell.

MODE_ENV = "MERLIN_PHASE2_GSIM_READBACK"

SCHEMA = "merlin_gsim_digest_readback_v1"


def mode() -> str:
    """``digest`` (default) or ``full``: how Phase 2 gSIM runs read their outputs back."""
    selected = os.environ.get(MODE_ENV, "digest").strip().lower()
    if selected not in ("digest", "full"):
        raise ValueError(f"{MODE_ENV} must be 'digest' or 'full', got {selected!r}")
    return selected


def enabled() -> bool:
    return mode() == "digest"


def run_gsim(
    cb: dict[str, Any],
    llvm_text: str,
    *,
    target: str,
    workdir: str | Path,
    timeout: int,
    oracle: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The gSIM oracle result for ``cb``: digest-verified when the checks hold, else a full readback."""
    oracle = _merlin_oracle_run_compile_1.run_on_oracle if oracle is None else oracle
    workdir = Path(workdir)

    def full_gsim(reason: str) -> dict[str, Any]:
        result = oracle(cb, llvm_text, simulator="gsim", target=target, workdir=workdir, timeout=timeout)
        result["readback"] = {"schema": SCHEMA, "mode": "full", "reason": reason}
        return result

    if not enabled():
        return full_gsim(f"{MODE_ENV}=full")
    policy = RB.ReadbackPolicy(RB.OUT_DIGEST_V1)
    digest_dir = workdir / "digest_readback"
    values_dir = workdir / "spike_full_values"
    digest_dir.mkdir(parents=True, exist_ok=True)
    values_dir.mkdir(parents=True, exist_ok=True)
    try:
        values = oracle(cb, llvm_text, simulator="spike", target=target, workdir=values_dir, timeout=timeout)
        on_spike = oracle(
            cb,
            llvm_text,
            simulator="spike",
            target=target,
            workdir=digest_dir,
            timeout=timeout,
            readback_policy=policy,
        )
    except Exception as exc:  # noqa: BLE001 -- the shortcut is optional; the full readback decides
        return full_gsim(f"spike digest verification unavailable: {type(exc).__name__}: {exc}"[:400])
    roster = on_spike.get("output_digests") or {}
    if not roster:
        return full_gsim("the digest program reported no output digests on Spike")
    spike_elf = sha256_file(Path(str(on_spike["elf"])))
    try:
        mismatched = RB.digest_mismatches(roster, values["outputs"])
    except (KeyError, ValueError) as exc:
        return full_gsim(f"spike values cannot be packed for their digest: {exc}"[:400])
    if mismatched:
        return full_gsim(f"the digest program's Spike output differs from the full-value build: {mismatched}")
    on_gsim = oracle(
        cb, llvm_text, simulator="gsim", target=target, workdir=digest_dir, timeout=timeout, readback_policy=policy
    )
    gsim_elf = sha256_file(Path(str(on_gsim["elf"])))
    if gsim_elf != spike_elf:
        return full_gsim("the gSIM digest build is not the ELF Spike verified")
    observed = {name: row["digest"] for name, row in (on_gsim.get("output_digests") or {}).items()}
    expected = {name: row["digest"] for name, row in roster.items()}
    if observed != expected:
        differing = sorted(n for n in set(observed) | set(expected) if observed.get(n) != expected.get(n))
        return full_gsim(f"gSIM digests differ from the Spike-verified values: {differing}")
    result = dict(on_gsim)
    result["outputs"] = values["outputs"]
    result["readback"] = {
        "schema": SCHEMA,
        "mode": "digest",
        "policy": policy.record(),
        "elf_sha256": gsim_elf,
        "spike_full_values_elf_sha256": sha256_file(Path(str(values["elf"]))),
        "digests": expected,
        "basis": (
            "outputs are the Spike full-value readback; the same digest ELF produced identical XXH64 "
            "digests of every output on Spike and on gSIM, and those equal the digest of these values"
        ),
    }
    return result
