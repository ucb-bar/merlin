"""Kernel-check exact actual reuse→nested segmentation in one memory meaning.

Fresh complete IR projection is trusted Python. This leaf validator does not
integrate receipts into ordinary compilation or prove a native SCF interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .lean_graph_loops import _reference
from .lean_graph_loops_memory_scf_model import MEMORY_SCF_ROOTS, obligation
from .lean_graph_loops_memory_scf_projection import project_exports
from .lean_graph_loops_reuse import ALLOCATOR_MODE
from .lean_graph_loops_reuse import _audit as _prior_audit


def audit(stdout, roots=MEMORY_SCF_ROOTS):
    names = [
        line.split("'", 2)[1]
        for line in stdout.splitlines()
        if line.startswith("'") and (" depends on axioms:" in line or " does not depend on any axioms" in line)
    ]
    if names != list(roots):
        raise ValueError("exact ordered root audit roster changed, missing or duplicated")
    return _prior_audit(stdout, roots)


def verify_memory_scf_files(before, middle, after, *, selection, output, index_bits, max_nodes, allocator_model):
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            before,
            middle,
            after,
            selection=selection,
            output=output,
            index_bits=index_bits,
            max_nodes=max_nodes,
            allocator_model=allocator_model,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    type=type(error).__name__,
                    lean_complete_composition_checked=False,
                    pipeline_receipt_issued=False,
                    python_projection_trusted=True,
                ),
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(before, middle, after, *, selection, output, index_bits, max_nodes, allocator_model):
    if allocator_model != ALLOCATOR_MODE:
        raise ValueError("explicit total/private/unobservable allocator required; actual8→16B counterexample retained")
    inputs = [path.read_bytes() for path in (before, middle, after)]
    originals = [_reference(path, data) for path, data in zip((before, middle, after), inputs, strict=True)]
    dependencies = selection.dependencies()
    from . import lean_graph_loops_memory_scf_capture as capture

    buffers = [output / name for name in ("before.mlir", "middle.mlir", "after.mlir")]
    for path, data in zip(buffers, inputs, strict=True):
        path.write_bytes(data)
    exported = output / "actual-triple.json"
    env = dict(os.environ, PYTHONPATH=str(selection.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(Path(capture.__file__).resolve()),
        "--source",
        str(selection.source),
        "--revision",
        selection.revision,
        "--index-bits",
        str(index_bits),
        "--before",
        str(buffers[0].resolve()),
        "--middle",
        str(buffers[1].resolve()),
        "--after",
        str(buffers[2].resolve()),
        "--output",
        str(exported.resolve()),
    ]
    parsed = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=120)
    (output / "projection.stdout").write_text(parsed.stdout)
    (output / "projection.stderr").write_text(parsed.stderr)
    if parsed.returncode:
        raise ValueError("fresh verified actual triple parsing/SSA export refused; malformed IR is separate")
    tree = json.loads(exported.read_bytes())
    exported_ref = _reference(exported)
    for name, path, data in zip(("before", "middle", "after"), buffers, inputs, strict=True):
        if tree[name] != _reference(path, data):
            raise ValueError("actual triple trees not bound to the SAME input buffers")
    manifest = json.loads(selection.manifest.read_bytes())
    for name in ("producer_source", "xdsl_loaded_source"):
        known = {row["path"]: row for row in manifest[name]}
        if not tree[name] or any(known.get(row["path"]) != row for row in tree[name]):
            raise ValueError("fresh loaded provider/parser differs from originally selected source")
    projection = project_exports(
        tree["before_export"], tree["middle_export"], tree["after_export"], index_bits=index_bits, max_nodes=max_nodes
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
    checked = subprocess.run(lean_command, env=env, capture_output=True, text=True, check=False, timeout=120)
    (output / "stdout.log").write_text(checked.stdout)
    (output / "stderr.log").write_text(checked.stderr)
    result = dict(
        schema="merlin.graph.memory_scf.actual_composition.v1",
        originals=originals,
        dependencies=dependencies,
        fresh_exports=exported_ref,
        model=model_ref,
        query=query_ref,
        projection_command=command,
        command=lean_command,
        returncode=checked.returncode,
        lean_complete_composition_checked=False,
        python_projection_trusted=True,
        operation_level_initializer_bridge=True,
        same_semantics_exact_middle=True,
        allocator_total_private_unobservable_is_premise=True,
        categorized_failure_or_divergence_equivalence=False,
        primitive_totality_inferred=False,
        independent_review_complete=False,
        pipeline_receipt_issued=False,
        LeanMLIR_selected=False,
        numerical_gates_discharged=False,
        native_allocator_or_hardware_proved=False,
    )
    if (
        dependencies != selection.dependencies()
        or originals != [_reference(path) for path in (before, middle, after)]
        or inputs != [path.read_bytes() for path in buffers]
        or exported_ref != _reference(exported)
        or model_ref != _reference(model)
        or query_ref != _reference(query)
    ):
        result["refusal"] = "actual triple/model/query or originally selected proof/source/tool/host binding changed"
    elif checked.returncode:
        result["refusal"] = "Lean refused complete operational memory/SSA/control/multiple-boundary composition"
        if checked.returncode < 0:
            result["context_disqualified_unknown_native_failure"] = True
    else:
        result["transitive_axioms"] = audit(checked.stdout, MEMORY_SCF_ROOTS)
        result["lean_complete_composition_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
