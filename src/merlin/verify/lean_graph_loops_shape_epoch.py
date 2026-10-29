"""Fresh four-buffer validation of shape normalization, reuse and segmentation.

The checked semantic adapter relates typed function entry and complete caller
observations to the epoch/slot model. Raw parsing and projection remain trusted;
the receipt neither gates ordinary compilation nor proves native model fidelity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .lean_graph_loops import _reference
from .lean_graph_loops_memory_scf import audit
from .lean_graph_loops_shape_epoch_model import obligation
from .lean_graph_loops_shape_epoch_observer import bind_observers
from .lean_graph_loops_shape_epoch_projection import bind_memory, project_adapter
from .lean_graph_loops_shape_epoch_selection import ALLOCATOR_POLICY, DESCRIPTOR_POLICY, ENTRY_POLICY
from .lean_graph_loops_shape_epoch_steps import bind_steps
from .lean_graph_loops_shape_model import SHAPE_ROOTS


def project_chain(shape, memory, *, index_bits, max_nodes):
    adapter = project_adapter(shape["before_export"], shape["after_export"], index_bits=index_bits, max_nodes=max_nodes)
    adapter = bind_memory(adapter, shape["after_export"], memory, index_bits=index_bits, max_nodes=max_nodes)
    return bind_steps(bind_observers(adapter), shape["after_export"])


def _parse(selection, output, buffers, index_bits):
    from . import lean_graph_loops_memory_scf_capture as memory_capture
    from . import lean_graph_loops_shape_capture as shape_capture

    env = dict(os.environ, PYTHONPATH=str(selection.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    captures, commands = [], []
    for name, capture, paths in (
        ("shape", shape_capture, (buffers[0], buffers[1])),
        ("memory", memory_capture, (buffers[1], buffers[2], buffers[3])),
    ):
        destination = output / ("actual-" + name + ".json")
        command = [
            sys.executable,
            str(Path(capture.__file__).resolve()),
            "--source",
            str(selection.source),
            "--revision",
            selection.revision,
            "--before",
            str(paths[0]),
            "--after",
            str(paths[-1]),
            "--output",
            str(destination),
        ]
        if name == "memory":
            command += ["--middle", str(paths[1]), "--index-bits", str(index_bits)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=120)
        (output / (name + ".stdout")).write_text(result.stdout)
        (output / (name + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise ValueError("fresh actual " + name + " parsing/SSA verification failed; malformed IR is distinct")
        captures.append((destination, json.loads(destination.read_bytes())))
        commands.append(command)
    return captures, commands


def _check_exports(captures, buffers, selection):
    shape, memory = [data for _, data in captures]
    if (
        shape["schema"] != "merlin.graph.shape.actual_pair.v1"
        or memory["schema"] != "merlin.graph.memory_scf.actual_triple.v1"
    ):
        raise ValueError("fresh parser export schema changed")
    if any(data["producer_revision"] != selection.revision for data in (shape, memory)):
        raise ValueError("fresh actual provider revision changed")
    for data, fields, paths in (
        (shape, ("before", "after"), buffers[:2]),
        (memory, ("before", "middle", "after"), buffers[1:]),
    ):
        if any(data[field] != _reference(path) for field, path in zip(fields, paths, strict=True)):
            raise ValueError("fresh trees differ from the SAME four submitted buffers")
    contract = json.loads(selection.contract.read_bytes())
    for field in ("producer_source", "xdsl_loaded_source"):
        known = {row["path"]: row for row in contract[field]}
        for data in (shape, memory):
            if not data[field] or any(known.get(row["path"]) != row for row in data[field]):
                raise ValueError("fresh parser/provider source differs from original adapter selection")


def verify_epoch_files(
    source,
    normalized,
    reused,
    segmented,
    *,
    selection,
    output,
    index_bits,
    max_nodes,
    allocator_policy,
    descriptor_policy,
    entry_policy,
):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            (source, normalized, reused, segmented),
            selection,
            output,
            index_bits,
            max_nodes,
            allocator_policy,
            descriptor_policy,
            entry_policy,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    type=type(error).__name__,
                    lean_complete_chain_checked=False,
                    python_projection_trusted=True,
                    pipeline_receipt_issued=False,
                ),
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(paths, selection, output, index_bits, max_nodes, allocator, descriptor, entry):
    if (allocator, descriptor, entry) != (ALLOCATOR_POLICY, DESCRIPTOR_POLICY, ENTRY_POLICY):
        raise ValueError("explicit allocator, descriptor and private/caller entry policies required")
    if type(index_bits) is not int or index_bits not in (32, 64) or type(max_nodes) is not int or max_nodes <= 0:
        raise ValueError("explicit supported index width and positive syntax budget required")
    originals = [Path(path).resolve(strict=True) for path in paths]
    payloads = [path.read_bytes() for path in originals]
    original_refs = [_reference(path, data) for path, data in zip(originals, payloads, strict=True)]
    dependencies = selection.dependencies()
    buffers = [output / name for name in ("source.mlir", "normalized.mlir", "reused.mlir", "segmented.mlir")]
    for path, data in zip(buffers, payloads, strict=True):
        path.write_bytes(data)
    captures, parse_commands = _parse(selection, output, buffers, index_bits)
    _check_exports(captures, buffers, selection)
    capture_refs = [_reference(path) for path, _ in captures]
    model_data = project_chain(captures[0][1], captures[1][1], index_bits=index_bits, max_nodes=max_nodes)
    model = output / "model.json"
    model.write_text(json.dumps(model_data, sort_keys=True, indent=2) + "\n")
    model_ref = _reference(model)
    text, adapter_roots = obligation(model_data)
    roots = SHAPE_ROOTS + adapter_roots
    query = output / "Obligation.lean"
    query.write_text(text)
    query_ref = _reference(query)
    proof = selection.proof
    env = dict(
        os.environ, LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=600)
    (output / "stdout.log").write_text(result.stdout)
    (output / "stderr.log").write_text(result.stderr)
    receipt = dict(
        schema="merlin.graph.shape_epoch.actual_chain.v1",
        originals=original_refs,
        dependencies=dependencies,
        captures=capture_refs,
        model=model_ref,
        query=query_ref,
        command=command,
        projection_commands=parse_commands,
        returncode=result.returncode,
        lean_complete_chain_checked=False,
        complete_return_and_caller_partition_observations=True,
        scoped_ssa_and_full_heap_adapter_derived=True,
        same_interpretation_exact_middle=True,
        all_finite_call_depths=True,
        policies=dict(allocator=allocator, descriptor=descriptor, entry=entry),
        python_projection_trusted=True,
        raw_parser_formally_verified=False,
        allocator_total_private_unobservable_is_premise=True,
        descriptor_and_primitive_fidelity_is_premise=True,
        categorized_error_or_divergence_equivalence=False,
        primitive_totality_inferred=False,
        pipeline_receipt_issued=False,
        ordinary_compiler_gating=False,
        independent_review_complete=False,
        numerical_gates_discharged=False,
        native_abi_hardware_fidelity_proved=False,
        LeanMLIR_selected=False,
    )
    if (
        dependencies != selection.dependencies()
        or original_refs != [_reference(path) for path in originals]
        or payloads != [path.read_bytes() for path in buffers]
        or capture_refs != [_reference(path) for path, _ in captures]
        or model_ref != _reference(model)
        or query_ref != _reference(query)
    ):
        receipt["refusal"] = "same-buffer/projection/query or original source/tool/import/host binding changed"
    elif result.returncode:
        receipt["refusal"] = "Lean refused the complete typed function/memory/initializer/SSA composition"
        if result.returncode < 0:
            receipt["context_disqualified_unknown_native_failure"] = True
    else:
        receipt["transitive_axioms"] = audit(result.stdout, roots)
        receipt["lean_complete_chain_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt
