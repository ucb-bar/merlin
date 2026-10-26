"""Nonzero actual common-model witnesses; finite xDSL evidence, not native proof."""

from __future__ import annotations

import json

import pytest
from xdsl.dialects import builtin
from xdsl.interpreters.memref import xtype_for_el_type
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr

from merlin.common.paths import repo_root
from merlin.tests.ir.test_lean_graph_loops_memory_scf import parse
from merlin.verify.lean_graph_loops_memory_scf_cases import CASES, CompositionCase
from merlin.verify.lean_graph_loops_memory_scf_runtime import execute
from merlin.verify.lean_graph_loops_reuse_cases import Case


def signed(value, bits):
    return (value + 2 ** (bits - 1)) % (2**bits) - 2 ** (bits - 1)


def reference(seeds, caller, case):
    observed = []
    multiplier = 4 if case.prefixes else 2
    for epoch, seed in enumerate(seeds):
        value = signed(multiplier * seed, case.epoch.bits)
        caller = signed(caller + value, case.epoch.bits)
        observed.append(caller)
        if case.epoch.overlap and epoch == 1:
            caller = signed(multiplier * seeds[0], case.epoch.bits)
    return (signed(observed[0] + observed[-1], case.epoch.bits), case.epoch.shape[0]), [caller]


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.epoch.name != "compact_huge"]
    + [CompositionCase(Case("two_epochs", (2, 3), 32, 16, 2), 2)],
    ids=lambda case: case.epoch.name,
)
@pytest.mark.parametrize("negative", [False, True])
def test_nonzero_actual_complete_triples(tmp_path, case, negative):
    root = repo_root() / "out/artifacts/verification/graph-memory-scf/independent-cases-v2"
    if case.epoch.name == "two_epochs":
        old = repo_root() / "out/artifacts/verification/graph-buffer-reuse"
        paths = [
            old / "actual-cases-v3/two_epochs/before.mlir",
            old / "actual-cases-v3/two_epochs/after.mlir",
            old / "actual-neighbor-segment-v1/after.mlir",
        ]
    else:
        paths = [
            root / (case.epoch.name + ".mlir"),
            root / case.epoch.name / "middle.mlir",
            root / case.epoch.name / "after.mlir",
        ]
    seeds = [(-1 if negative else 1) * (17 + 12 * index) for index in range(case.epoch.epochs)]
    initial = -7 if negative else 11
    records = []
    kind = builtin.IntegerType(case.epoch.bits)
    for path in paths:
        caller = ShapedArray(TypedPtr.new([initial], xtype=xtype_for_el_type(kind, case.epoch.index_bits)), [1])
        records.append(execute(parse(path.read_text()), (*seeds, caller), index_bits=case.epoch.index_bits))
    expected, caller = reference(seeds, initial, case)
    assert all(record["result"] == expected and record["caller"] == caller for record in records)
    assert records[0]["trace"] == records[1]["trace"] == records[2]["trace"]
    assert records[0]["recognized_control_operations"] == records[1]["recognized_control_operations"] == 0
    assert records[2]["recognized_control_operations"] == case.epoch.epochs * 2
    assert records[0]["allocation_count"] > records[1]["allocation_count"] == records[2]["allocation_count"]
    assert all(record["all_private_reads_initialized"] and record["all_private_frees_valid"] for record in records)
    (tmp_path / "RUNTIME.json").write_text(
        json.dumps(
            dict(
                case=case.epoch.name,
                seeds=seeds,
                initial_caller=initial,
                records=records,
                expected=list(expected),
                native_conformance_proved=False,
            ),
            indent=2,
        )
        + "\n"
    )
