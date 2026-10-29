"""Pure preparation protocol for later canonical integration; no eligibility."""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.common.paths import module_source_path, repo_root

from .compiler_loop_generic_profile import source_profile
from .gate_errors import GateUnavailable
from .lean_graph_generic_prepare import prepare
from .lean_graph_generic_unroll_emit import emit

ROOT = Path(__file__).resolve().parent


def ref(path):
    path = path.resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def prepare_query(before, after, *, factor=2, index_bits=64):
    profile = source_profile(before, kind="ordered_loop_unrolling", factor=factor, index_bits=index_bits)
    value = prepare(before, after, factor=factor, index_bits=index_bits)
    value["source_profile"] = profile
    query = emit(value)
    if len(query.encode()) > 4 * 1024 * 1024:
        raise GateUnavailable("generic compiler query4MiB resource policy exceeded")
    value.pop("raw_buffers")
    prefix = "Merlin.GraphStatePipelineInstance."
    names = [
        "controlsChecked",
        "ordinaryChecked",
        "ordinary_control",
        "literalsChecked",
        "boundsChecked",
        "reachable_source",
        "m_default",
        "m_index",
        "m_iterations",
        *("control" + str(row["identity"]) for row in value["index_operations"]),
        "tick_effect",
        "tailScopeChecked",
        "tail_same",
        "body_group",
        "grouped",
        "source_form",
        "target_form",
        "minimal",
        "sourceRolesChecked",
        "targetRolesChecked",
        "sourceRoleTypes",
        "targetRoleTypes",
        "sourceScopeChecked",
        "targetScopeChecked",
        "rest",
    ]
    for i, function in enumerate(value["capture"]["functions"][0]):
        if function["declaration"]:
            continue
        local = ["originalTyped", "finalTyped"]
        local += (
            ["rawSourceJoin", "rawTargetJoin", "sourceJoin", "targetJoin", "growth", "actual"]
            if i == value["function"]
            else ["same", "actual"]
        )
        names += [f"F{i}.{name}" for name in local]
    names += ["complete", "complete2", "completeData"]
    sources = [
        ROOT / name
        for name in (
            "compiler_loop_generic_intake.py",
            "compiler_loop_generic_profile.py",
            "lean_graph_generic_prepare.py",
            "lean_graph_generic_parser.py",
            "memory_scalar_nan_admission.py",
            "lean_graph_generic_unroll_emit.py",
            "lean_graph_generic_roles.py",
            "lean_graph_generic_affine.py",
            "lean_graph_generic_admission.py",
            "lean_graph_generic_exact.py",
            "lean_graph_generic_regions.py",
            "lean_graph_generic_projection.py",
            "lean_graph_generic_render.py",
        )
    ]
    sources += [
        ROOT / "lean" / "Merlin" / name
        for name in (
            "GraphStatePipelineValueCandidate.lean",
            "GraphStatePipelineGenericContextCandidate.lean",
            "GraphStatePipelineReachableCandidate.lean",
            "GraphStatePipelineControlRolesCandidate.lean",
        )
    ]
    sources += [
        repo_root() / "src/merlin" / name
        for name in (
            "frontends/linalg_mlir.py",
            "common/dense_float_ir.py",
            "common/ir_lock.py",
            "common/paths.py",
            "verify/lean_graph_input_admission.py",
            "verify/gate_errors.py",
            "xdsl_dialects/fp8.py",
            "transforms/unrolling.py",
            "transforms/certificates.py",
            "verify/lean_graph_state_pipeline_capture.py",
            "verify/lean_graph_state_scalar.py",
            "verify/lean_graph_state_pipeline.py",
        )
    ]
    # This finite parser/runtime source roster is custody, not a proof of Python
    # or parser correspondence. Root additionally binds actual gate/worker/tools.
    sources += [
        module_source_path(name)
        for name in (
            "xdsl.dialects.builtin",
            "xdsl.parser",
            "xdsl.parser.core",
            "xdsl.parser.attribute_parser",
            "xdsl.printer",
            "xdsl.context",
            "xdsl.ir.core",
            "xdsl.ir.affine",
            "xdsl.ir.affine.affine_expr",
            "xdsl.ir.affine.affine_map",
            "xdsl.ir.affine.affine_set",
            "xdsl.utils.lexer",
            "xdsl.utils.mlir_lexer",
        )
    ]
    return {
        "authority": False,
        "projection_correspondence": (
            "T7 occurrence-aware actual control projection; checked role syntax/types/Fits; "
            "not arbitrary raw External equivalence"
        ),
        "raw_buffers": (before, after),
        "query": query,
        "roots": tuple(prefix + name for name in names),
        "sources": [ref(path) for path in sorted(sources)],
        "capture": value,
        "final_root": prefix + "complete",
        "domain": "compiler-static-ordered-scf-generic-value-index64-v1",
    }
