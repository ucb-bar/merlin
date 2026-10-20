"""Narrow source semantics for archive exploration, never target admission.

This recognizes one direct, zero-initialized signed rank-two dot.  A source
profile establishes neither accelerator execution nor timing equivalence. The
ordinary owner still requires original complete numerics, actual event counts,
reviewed target semantics and the unchanged timing-domain checks.
"""

from __future__ import annotations

import hashlib
import json

from merlin.common import mlir_query as Q
from merlin.perf.candidate_source import MAX_TEXT
from merlin.runtime import _native_files as F
from merlin.transforms.dense_ir import recognize_signed_integer_matmul

from .contracts import document_sha256

SCHEMA = "merlin.archive_single_dot_profile.v1"


def source_modules():
    """Original host recognizers to pin with the queue owner, not candidate code."""
    from merlin.perf import candidate_source, compiler_plan_evidence
    from merlin.targetgen.oot_starterkit import llvm_context, plan
    from merlin.transforms import dense_ir

    return (Q, candidate_source, compiler_plan_evidence, llvm_context, plan, dense_ir)


def _profile(m, k, n):
    if any(type(value) is not int or not 0 < value < 2**31 for value in (m, k, n)):
        raise ValueError("source contraction extents are not bounded positive integers")
    return {"schema": SCHEMA, "family": "matmul", "m": m, "k": k, "n": n,
            "operand_dtypes": ["i8", "i8"], "result_dtype": "i32",
            "initialization": "zero", "layout": "unencoded_rank_two_matmul_maps",
            "result": "complete_raw_i32_dot", "other_source_arithmetic": False}


def interface_profile(text):
    """Read the exact original interface contract, including raw readout."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    cb = parse_interface_mlir(text)
    commands = cb["commands"]
    if [row.get("opcode") for row in commands] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]:
        raise ValueError("archive source is not one closed resident dot and raw readout")
    pack, dot, commit, evict = commands
    p, d, c, e = [row.get("operands", {}) for row in commands]
    if (pack.get("attributes") != {"layout": "packed_rhs"} or dot.get("attributes", {})
            or commit.get("attributes") != {"epilogue": [], "output_dtype": "i32"}
            or evict.get("attributes", {}) or set(p) != {"src", "dst"}
            or set(d) != {"lhs", "rhs", "dst"} or set(c) != {"src", "dst"}
            or set(e) != {"handle"} or d["rhs"] != p["dst"]
            or c["src"] != d["dst"] or e["handle"] != p["dst"]):
        raise ValueError("archive dot has additional arithmetic, layout or readout effects")
    tensors = cb["tensors"]
    if set(tensors) != {p["src"], d["lhs"]}:
        raise ValueError("archive dot has additional external tensors")
    lhs, rhs = tensors[d["lhs"]], tensors[p["src"]]
    if (lhs.get("dtype") != "i8" or rhs.get("dtype") != "i8"
            or len(lhs.get("shape", [])) != 2 or len(rhs.get("shape", [])) != 2):
        raise ValueError("archive dot is not signed rank-two integer input")
    (m, k), (rhs_k, n) = lhs["shape"], rhs["shape"]
    if k != rhs_k:
        raise ValueError("archive dot reduction extents differ")
    return _profile(m, k, n)


def typed_profile(text):
    """Independently recognize the complete current source; ignore only provenance.

    No graph labels, model names, qualification JSON, or declared operation
    family contribute to this finite structural proof.
    """
    from xdsl.dialects.builtin import IntegerAttr

    if type(text) is not str or not 0 < len(text.encode()) <= MAX_TEXT:
        raise ValueError("current source exceeds bounded profile intake")
    module = Q.parse(text)
    module.verify()
    for op in module.walk():
        if any(not key.startswith("prov.") for key in op.attributes):
            raise ValueError("current source has unrecognized executable attributes")
    detached = module.clone()
    for op in detached.walk():
        op.attributes.clear()
    top = list(detached.body.block.ops)
    if len(top) != 1 or top[0].name != "func.func":
        raise ValueError("current source needs one closed function")
    function = top[0]
    if (len(function.body.blocks) != 1
            or set(function.properties) != {"sym_name", "function_type"}):
        raise ValueError("current source function has unsupported effects or blocks")
    block = function.body.block
    ops = list(block.ops)
    if len(block.args) != 2 or [op.name for op in ops] != [
            "arith.constant", "tensor.splat", "linalg.generic", "func.return"]:
        raise ValueError("current source is not one direct dot with only zero initialization")
    zero, splat, dot, returned = ops
    value = zero.properties.get("value")
    if (set(zero.properties) != {"value"} or zero.operands or len(zero.results) != 1
            or not isinstance(value, IntegerAttr) or str(value.type) != "i32" or value.value.data != 0
            or splat.properties or list(splat.operands) != [zero.results[0]] or len(splat.results) != 1
            or returned.properties or list(returned.operands) != list(dot.results)):
        raise ValueError("current source initialization or complete raw result is not exact")
    lhs, rhs, initial = recognize_signed_integer_matmul(dot)
    if (list(block.args) != [lhs, rhs] or initial is not splat.results[0]
            or initial.type != dot.results[0].type):
        raise ValueError("current source dot inputs/result do not have direct external ownership")
    shapes = [tuple(value.type.get_shape()) for value in (lhs, rhs, initial)]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("current source needs a separately checked batched/layout domain")
    (m, k), (rhs_k, n), output = shapes
    if rhs_k != k or output != (m, n):
        raise ValueError("current source dot shape or complete output differs")
    return _profile(m, k, n)


def train_support(catalog, profile):
    """Exact joint semantics/geometry from TRAIN only; not independent min/max."""
    sources = catalog.get("sources") or {}
    return sorted({entry["case"]["source_group_sha256"] for entry in catalog.get("cases", [])
                   if entry["case"]["split"] == "train"
                   and entry["case"]["mechanism_family"] == profile["family"]
                   and entry["case"]["source"].get("visibility") == "public"
                   and any(row.get("native_observation_available") is True
                           and row.get("scope_binding") is not None
                           and row.get("originally_measured") is True
                           and row.get("original_numerical_status") == "pass"
                           and type(row.get("cycles")) is int and row["cycles"] > 0
                           for row in entry.get("records", []))
                   and sources.get(entry["case"]["source"]["source_sha256"], {}).get("semantic_profile") == profile})


def current(member, compiled, workspace, catalog):
    """Reopen current source + actual compiled plan, preserving precise refusal."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan

    unknown = {"status": "UNKNOWN", "family": None}
    if (member.descriptor.get("operation") or {}).get("op") != "model":
        return {**unknown, "reason": "typed model source profile is not applicable"}
    if member.descriptor.get("performance", {}).get("capacity"):
        return {**unknown, "reason": "capacity extrapolation has no paired public qualification"}
    try:
        name = member.descriptor["interface_mlir"]
        if type(name) is not str or name != "capsule.interface.mlir":
            raise ValueError("current model source has no exact selected interface role")
        path = member.source_dir / name
        source_pin = F.pin(path, max_bytes=MAX_TEXT)
        raw = path.read_bytes()
        if F.pin(path, max_bytes=MAX_TEXT) != source_pin:
            raise ValueError("current typed model source changed during reopening")
        digest = hashlib.sha256(raw).hexdigest()
        diagnostics = compiled.get("candidate_diagnostics") or {}
        artifacts = diagnostics.get("artifacts") or {}
        join = diagnostics.get("source_emission_join") or {}
        if (compiled.get("status") != "observed" or join.get("status") != "verified"
                or digest != (artifacts.get("source") or {}).get("sha256")
                or digest != join.get("source_sha256")):
            raise ValueError("current typed source lacks original complete source/emission binding")
        wanted = (artifacts.get("command_buffer") or {}).get("sha256")
        buffers = {}
        for file in F.members(workspace, max_files=10000):
            if file.name == "command_buffer.json" and file.is_file() and not file.is_symlink():
                pin = F.pin(file, max_bytes=MAX_TEXT)
                if pin["sha256"] == wanted:
                    buffers.setdefault(pin["sha256"], file.read_bytes())
        if len(buffers) != 1:
            raise ValueError("current typed source lacks actual selected compiled plan bytes")
        cb_raw = next(iter(buffers.values()))
        cb = json.loads(cb_raw)
        if validate_mixed_program_plan(raw, cb).get("ok") is not True:
            raise ValueError("current typed source/compiled tensor and task plan differ")
        profile = typed_profile(raw.decode())
        supported = train_support(catalog, profile)
        if not supported:
            return {**unknown, "reason": "exact joint source semantics/geometry absent from public TRAIN evidence",
                    "profile": profile, "source_sha256": digest}
        result = {"status": "source_domain_supported", "family": profile["family"], "profile": profile,
                  "source_sha256": digest, "command_buffer_sha256": wanted,
                  "training_source_groups": supported,
                  "authority": "source timing-family/domain selection only; no target or timing qualification"}
        result["sha256"] = document_sha256(result)
        return result
    except (ValueError, KeyError, TypeError, AttributeError, NotImplementedError) as error:
        return {**unknown, "reason": "current typed model source profile refused: " + str(error)[:180]}
    except (OSError, ParseError, VerifyException) as error:
        return {**unknown, "reason": "current typed model source profile unavailable: " + type(error).__name__}
