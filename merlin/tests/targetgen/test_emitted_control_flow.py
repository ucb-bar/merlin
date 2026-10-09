"""Actual CFG/phi inventory remains distinct from executed output coverage."""

import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from merlin.targetgen.contract.emitted_control_flow import observe_emitted_control_flow
from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable, observe_emitted_dataflow

_LOOP = """module { llvm.func @entry(%input: !llvm.ptr, %output: !llvm.ptr) {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %limit = llvm.mlir.constant(15 : i64) : i64
  llvm.br ^loop(%zero : i64)
^loop(%index: i64):
  %condition = llvm.icmp "ult" %index, %limit : i64
  llvm.cond_br %condition, ^body, ^done
^body:
  %src = llvm.getelementptr %input[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %value = llvm.load %src : !llvm.ptr -> i8
  %dst = llvm.getelementptr %output[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %value, %dst : i8, !llvm.ptr
  %next = llvm.add %index, %one : i64
  llvm.br ^loop(%next : i64)
^done:
  llvm.return
} }"""


def observe(text=_LOOP, **kwargs):
    return observe_emitted_control_flow(text, entry_symbol="entry", pointer_bits=64, **kwargs)


def test_complete_runtime_loop_retains_cyclic_phi_edges_and_static_memory_operations():
    result = observe()
    assert len(result.blocks) == 4 and len(result.arguments) == 2
    loop_argument = result.blocks[1].arguments[0]
    edges = [edge for op in result.operations for edge in op.edges if edge.successor == 1]
    assert len(edges) == 2 and all(edge.parameters == (loop_argument,) for edge in edges)
    assert edges[0].arguments != edges[1].arguments
    assert [row.name for row in result.operations].count("llvm.load") == 1
    assert [row.name for row in result.operations].count("llvm.store") == 1
    assert "loop_trip_counts" in result.unknown and "complete_output_stores" in result.unknown
    with pytest.raises(DataflowUnavailable, match="control-flow join"):
        observe_emitted_dataflow(_LOOP, entry_symbol="entry", pointer_bits=64)


def test_ssa_and_block_names_do_not_select_observation():
    renamed = _LOOP.replace("%", "%renamed_").replace("^", "^renamed_")
    original, actual = asdict(observe()), asdict(observe(renamed))
    assert original.pop("source_sha256") != actual.pop("source_sha256")
    assert original == actual


def test_different_pointer_incoming_values_remain_distinct_not_a_claimed_unique_origin():
    source = """module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {
      %cond = llvm.mlir.constant(1 : i1) : i1
      llvm.cond_br %cond, ^join(%a : !llvm.ptr), ^join(%b : !llvm.ptr)
    ^join(%chosen: !llvm.ptr):
      %value = llvm.load %chosen : !llvm.ptr -> i8
      llvm.store %value, %b : i8, !llvm.ptr
      llvm.return } }"""
    result = observe(source)
    edges = next(row.edges for row in result.operations if row.name == "llvm.cond_br")
    assert edges[0].parameters == edges[1].parameters
    assert edges[0].arguments == (result.arguments[0],)
    assert edges[1].arguments == (result.arguments[1],)
    assert "pointer_ranges" in result.unknown and "alias" in result.unknown


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("llvm.br ^loop(%zero : i64)", "llvm.br ^loop"),
        lambda text: text.replace(
            "%zero = llvm.mlir.constant(0 : i64) : i64", "%zero = llvm.mlir.constant(0 : i32) : i32"
        ).replace("^loop(%zero : i64)", "^loop(%zero : i32)"),
        lambda text: text.replace(
            "^done:\n  llvm.return", "^done:\n  llvm.store %value, %output : i8, !llvm.ptr\n  llvm.return"
        ),
        lambda text: text.replace(
            "  %value = llvm.load %src : !llvm.ptr -> i8",
            "  llvm.store %value, %output : i8, !llvm.ptr\n  %value = llvm.load %src : !llvm.ptr -> i8",
        ),
        lambda text: text.replace("module {", "module { llvm.func @opaque(!llvm.ptr)"),
        lambda text: text.replace(
            "%next = llvm.add %index, %one : i64", "%next = llvm.select %condition, %index, %one : i1, i64"
        ),
        lambda text: text.replace(
            "llvm.store %value, %dst : i8, !llvm.ptr",
            "llvm.store %value, %dst {candidate_complete = true} : i8, !llvm.ptr",
        ),
    ],
)
def test_missing_join_nondominating_definition_dispatch_or_opaque_claim_refuses(change):
    with pytest.raises(DataflowUnavailable):
        observe(change(_LOOP))


def test_dead_block_is_retained_in_original_complete_static_denominator():
    result = observe(_LOOP.replace("^done:\n  llvm.return", "^done:\n  llvm.return\n^dead:\n  llvm.return"))
    assert len(result.blocks) == 5
    assert result.operations[result.blocks[-1].operations[-1]].name == "llvm.return"
    assert "dynamic_execution_paths" in result.unknown


@pytest.mark.parametrize(
    "kwargs", [{"max_blocks": 3}, {"max_operations": 3}, {"max_blocks": True}, {"max_operations": 0}]
)
def test_explicit_reader_bounds_are_not_compiled_resource_permissions(kwargs):
    with pytest.raises(DataflowUnavailable, match="bound"):
        observe(**kwargs)


@pytest.mark.parametrize("shape", [(2, 3), (3, 5), (1, 17)])
def test_real_selected_stock_upstream_copy_observes_multiple_dimensions_and_tail(tmp_path, shape):
    selected = os.environ.get("MERLIN_TEST_MLIR_OPT")
    if not selected:
        pytest.skip("requires an explicitly selected stock upstream MLIR optimizer")
    tensor = "memref<" + "x".join(map(str, shape)) + "xi8>"
    source = tmp_path / "copy.mlir"
    source.write_text(
        f"module {{ func.func @entry(%in: {tensor}, %out: {tensor}) {{ "
        f"linalg.copy ins(%in : {tensor}) outs(%out : {tensor}) func.return }} }}"
    )
    command = (
        str(Path(selected).resolve(strict=True)),
        str(source),
        "--convert-linalg-to-loops",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm=use-bare-ptr-memref-call-conv",
        "--reconcile-unrealized-casts",
        "--canonicalize",
    )
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    )
    assert result.returncode == 0, result.stderr
    observed = observe(result.stdout)
    assert len(observed.blocks) > 1 and len(observed.arguments) == 2
    assert sum(row.name == "llvm.store" for row in observed.operations) == 1
    geps = [row for row in observed.operations if row.name == "llvm.getelementptr"]
    assert len(geps) == 2 and all(len(row.operands) == 2 for row in geps)
    assert all(dict(row.properties)["elem_type"] == "i8" for row in geps)
    assert "complete_output_stores" in observed.unknown
    with pytest.raises(DataflowUnavailable, match="control-flow join"):
        observe_emitted_dataflow(result.stdout, entry_symbol="entry", pointer_bits=64)
