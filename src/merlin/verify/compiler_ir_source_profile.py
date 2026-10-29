"""Source-only model admission for tensor, Q/DQ and draft MACC boundaries.

The installed Lean source recognizers decide eligibility before any proposed
target is inspected. Their diagnostic result grants no boundary proof credit.
Tools, malformed responses, timeouts and input/source custody failures block.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from merlin.common.owned_process import run_owned

from .compiler_profile import SourceOutsideProfile
from .configuration_elision_bridge import digest, file_ref
from .gate_errors import GateUnavailable

TENSOR = "typed-tensor-rulebook-index64-v1"
QDQ = "proposed-qdq-typed-source-index64-v1"
MACC = "draft-uniform-half-positive-zero-macc-index64-v1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def _outside(profile, code, reason):
    return SourceOutsideProfile(profile, code, reason)


def admit_source(raw, *, pass_name, validation):
    from xdsl.dialects import builtin, func

    from merlin.common.ir_lock import IR_LOCK
    from merlin.frontends.linalg_mlir import make_context

    from .lean import check, export
    from .memory_scalar_nan_admission import _parse_bounded

    profile = MACC if pass_name == "bf16_f32acc_draft_contract" else QDQ if pass_name == "lower_quant_ext" else TENSOR
    if type(raw) is not str or len(raw.encode()) > MAX_SOURCE_BYTES:
        raise _outside(profile, "source-byte-bound", "complete original IR exceeds the declared 4MiB model bound")
    with IR_LOCK:
        module = _parse_bounded(raw, maximum_bytes=MAX_SOURCE_BYTES, memory=False, context=make_context())
    operations = tuple(module.walk())
    if len(operations) > 5000:
        raise _outside(profile, "source-syntax-bound", "source profile permits at most 5000 operations")
    if any(op.successors or any(len(region.blocks) > 1 for region in op.regions) for op in operations):
        raise _outside(profile, "single-block-regions", "successors and multi-block regions are outside the IR model")
    if pass_name == "bf16_f32acc_draft_contract":
        for operation in operations:
            if operation.name not in ("linalg.matmul", "linalg.generic"):
                continue
            half = any(
                isinstance(result.type, builtin.TensorType)
                and isinstance(result.type.element_type, (builtin.Float16Type, builtin.BFloat16Type))
                for result in operation.results
            )
            if half and type(operation.parent_op()) is not func.FuncOp:
                raise _outside(profile, "top-level-sites", "draft MACC admission covers function-entry contractions")
    directory = Path(validation.gate.directory) / "source-profiles" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    source, exported = directory / "source.mlir", directory / "source.json"
    source.write_bytes(raw.encode())
    from merlin.common.paths import module_source_path, repo_root

    tracked = [
        check.lean_exe(),
        export.MLIR_EXPORT,
        Path(validation.tools.compiler_python),
        repo_root() / "merlin/lean/Main.lean",
        repo_root() / "merlin/lean/MerlinVerify/Driver/SourceProfile.lean",
        module_source_path("merlin.verify.compiler_ir_source_profile"),
    ]
    try:
        refs = [file_ref(path) for path in tracked]
        result = run_owned(
            [str(validation.tools.compiler_python), str(export.MLIR_EXPORT), str(source), str(exported)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise GateUnavailable("source-only authoritative export could not complete: " + result.stderr[-500:])
        exported_ref = file_ref(exported)
        exported_bytes = exported.read_bytes()
        if digest(exported_bytes) != exported_ref["sha256"] or len(exported_bytes) != exported_ref["bytes"]:
            raise GateUnavailable("source-only export changed during independent read")
        document = json.loads(exported_bytes)
        if document.get("source") != {"sha256": digest(raw), "bytes": len(raw.encode())}:
            raise GateUnavailable("source-only authoritative export does not bind the original bytes")
        kind = "macc" if profile == MACC else "tensor"
        result = run_owned(
            [str(check.lean_exe()), "source-profile", kind, str(exported), "64"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, "PATH": f"{validation.tools.sdk_bin}{os.pathsep}{os.environ.get('PATH', '')}"},
        )
        detail = json.loads(result.stdout)
        if (
            result.returncode not in (0, 1)
            or detail.get("schema") != "merlin.compiler-source-profile.v1"
            or detail.get("profile") != kind
            or detail.get("index_bits") != 64
            or type(detail.get("supported")) is not bool
            or result.returncode != (0 if detail["supported"] else 1)
            or detail.get("boundary_proof") is not False
            or detail.get("policy_approval") is not False
        ):
            raise GateUnavailable("source-only recognizer returned a different or incomplete obligation")
        if (
            source.read_bytes() != raw.encode()
            or file_ref(exported) != exported_ref
            or [file_ref(path) for path in tracked] != refs
        ):
            raise GateUnavailable("source-only admission sources, tools or original bytes changed")
        (directory / "admission.json").write_text(
            json.dumps({**detail, "sources": refs, "export": exported_ref}, indent=2) + "\n"
        )
        if not detail["supported"]:
            if type(detail.get("reason")) is not str or not detail["reason"]:
                raise GateUnavailable("source-only recognizer omitted its model exclusion")
            raise _outside(profile, "source-model", detail["reason"])
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        if isinstance(error, SourceOutsideProfile):
            raise
        raise GateUnavailable("source-only admission could not complete") from error
    return profile


def admit_memory_dce(raw):
    from .buffer_reuse_bridge import decode_pair
    from .compiler_memory_stage_intake import decode
    from .lean_graph_input_admission import InputAdmissionError

    profile = "ordinary-memory-context-dead-index-constants-index64-v1"
    if len(raw.encode()) > 256 * 1024:
        raise _outside(profile, "source-byte-bound", "memory source model permits at most 256KiB of original IR")
    try:
        projected = decode_pair(raw, raw)  # Source-only identity; never a producer or a proof.
        decode(projected["source_export"], index_bits=64, max_nodes=5000)
    except (ValueError, InputAdmissionError) as error:
        raise _outside(profile, "source-model", str(error)) from error
    return profile


def admit_host_source(raw, *, fusion_groups, entry):
    """Original graph and original fusion selections, never emitted helpers."""
    from xdsl.dialects import func

    from merlin.common.ir_lock import IR_LOCK
    from merlin.frontends.linalg_mlir import make_context

    from .gate_errors import GateUnsupported
    from .host_graph_fusion_intake import DOMAIN
    from .host_graph_fusion_projection import project
    from .memory_scalar_nan_admission import _parse_bounded

    if len(raw.encode()) > 1 << 20:
        raise _outside(DOMAIN, "source-byte-bound", "host fusion source exceeds its 1MiB model bound")
    with IR_LOCK:
        module = _parse_bounded(raw, maximum_bytes=1 << 20, memory=False, context=make_context())
    try:
        projected = project(raw, strict_source=True)
        if any(call["count"] > 1 for call in projected["calls"]):
            raise GateUnsupported("source already has a multi-stage host dispatch")
    except GateUnsupported as error:
        raise _outside(DOMAIN, "source-model", str(error)) from error
    if entry != projected["entry"] or len(fusion_groups) != 1:
        raise _outside(DOMAIN, "fusion-selection", "host model checks one original selected group in its public entry")
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == entry)
    operations = tuple(function.body.block.ops)
    call_counts = iter(call["count"] for call in projected["calls"])
    counts = [next(call_counts) if op.name == "func.call" else int(op.name == "linalg.generic") for op in operations]
    group = fusion_groups[0]
    if any(type(index) is not int or not 0 <= index < len(counts) for index in group):
        raise ValueError("host fusion requires its actual original operation indices")
    count = sum(counts[index] for index in group)
    if not 2 <= count <= 128:
        raise _outside(DOMAIN, "fusion-stage-count", "selected source group requires 2 through 128 pointwise stages")
    return DOMAIN
