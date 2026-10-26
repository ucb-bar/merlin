"""Fresh complete-IR shape normalization validation under declared view semantics.

The typed descriptor/SSA/world theorem is kernel checked. The full parser and
projection remain trusted Python; this does not gate ordinary compilation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .lean_graph_loops import _reference
from .lean_graph_loops_memory_scf import audit
from .lean_graph_loops_shape_model import SHAPE_ROOTS, obligation
from .lean_graph_loops_shape_projection import project_exports


def verify_shape_files(before, after, *, selection, output, index_bits, max_nodes, descriptor_policy):
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            before,
            after,
            selection=selection,
            output=output,
            index_bits=index_bits,
            max_nodes=max_nodes,
            descriptor_policy=descriptor_policy,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    type=type(error).__name__,
                    lean_complete_shape_checked=False,
                    pipeline_receipt_issued=False,
                    python_projection_trusted=True,
                ),
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(before, after, *, selection, output, index_bits, max_nodes, descriptor_policy):
    if descriptor_policy != "typed_static_identity_layout":
        raise ValueError(
            "explicit typed static descriptor policy required; metadata hashes cannot establish conformance"
        )
    first, second = before.read_bytes(), after.read_bytes()
    originals = [_reference(before, first), _reference(after, second)]
    dependencies = selection.dependencies()
    saved_before, saved_after = output / "before.mlir", output / "after.mlir"
    saved_before.write_bytes(first)
    saved_after.write_bytes(second)
    from . import lean_graph_loops_shape_capture as capture

    exported = output / "actual-pair.json"
    env = dict(os.environ, PYTHONPATH=str(selection.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(Path(capture.__file__).resolve()),
        "--source",
        str(selection.source),
        "--revision",
        selection.revision,
        "--before",
        str(saved_before.resolve()),
        "--after",
        str(saved_after.resolve()),
        "--output",
        str(exported.resolve()),
    ]
    parsed = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=120)
    (output / "projection.stdout").write_text(parsed.stdout)
    (output / "projection.stderr").write_text(parsed.stderr)
    if parsed.returncode:
        raise ValueError("fresh actual parsing/SSA export refused; malformed IR is separate")
    data = json.loads(exported.read_bytes())
    export_ref = _reference(exported)
    if data["schema"] != "merlin.graph.shape.actual_pair.v1" or data["producer_revision"] != selection.revision:
        raise ValueError("fresh shape parser export schema/source identity changed")
    if data["before"] != _reference(saved_before, first) or data["after"] != _reference(saved_after, second):
        raise ValueError("actual shape trees not bound to the same input buffers")
    manifest = json.loads(selection.manifest.read_bytes())
    for field in ("producer_source", "xdsl_loaded_source"):
        if not data[field] or data[field] != manifest[field]:
            raise ValueError("fresh loaded shape provider/parser differs from original selection")
    projection = project_exports(
        data["before_export"], data["after_export"], index_bits=index_bits, max_nodes=max_nodes
    )
    model = output / "model.json"
    model.write_text(json.dumps(projection, sort_keys=True, indent=2) + "\n")
    model_ref = _reference(model)
    query = output / "Obligation.lean"
    query.write_text(obligation(projection))
    query_ref = _reference(query)
    proof = selection.proof
    env.update(LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2")
    lean_command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
    checked = subprocess.run(lean_command, env=env, capture_output=True, text=True, check=False, timeout=180)
    (output / "stdout.log").write_text(checked.stdout)
    (output / "stderr.log").write_text(checked.stderr)
    result = dict(
        schema="merlin.graph.shape.actual_whole_module.v1",
        originals=originals,
        dependencies=dependencies,
        fresh_exports=export_ref,
        model=model_ref,
        query=query_ref,
        command=lean_command,
        projection_command=command,
        returncode=checked.returncode,
        lean_complete_shape_checked=False,
        python_projection_trusted=True,
        descriptor_policy=descriptor_policy,
        typed_runtime_descriptor_fidelity_is_premise=True,
        retained_operations_may_be_partial_and_observable=True,
        returned_values_and_entire_world=True,
        private_ssa_stacks_related_not_equated=True,
        categorized_error_or_divergence_equivalence=False,
        primitive_totality_inferred=False,
        ordinary_compiler_gating=False,
        pipeline_receipt_issued=False,
        old_epoch_model_adapter_proved=False,
        independent_review_complete=False,
        numerical_gates_discharged=False,
        native_allocator_abi_hardware_conformance_proved=False,
        LeanMLIR_selected=False,
    )
    if (
        dependencies != selection.dependencies()
        or originals != [_reference(before), _reference(after)]
        or saved_before.read_bytes() != first
        or saved_after.read_bytes() != second
        or export_ref != _reference(exported)
        or model_ref != _reference(model)
        or query_ref != _reference(query)
    ):
        result["refusal"] = "actual inputs/tree/model/query or original proof/source/tool/import/host changed"
    elif checked.returncode:
        result["refusal"] = "Lean refused whole-module descriptor/SSA/scope/dimension/call obligations"
        if checked.returncode < 0:
            result["context_disqualified_unknown_native_failure"] = True
    else:
        result["transitive_axioms"] = audit(checked.stdout, SHAPE_ROOTS)
        result["lean_complete_shape_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
