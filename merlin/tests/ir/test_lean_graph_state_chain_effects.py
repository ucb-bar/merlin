"""Actual composed schedules retain caller aliases and ordered observers."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import test_lean_graph_state as base

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state_chain as direct
from merlin.verify import lean_graph_state_chain_transport as transport


def actual_effect_chain(root, kind):
    contract = base.contract()
    if kind == "memory":
        text = """module {
func.func @kernel(%a:i8,%b:i8,%left:memref<7xi8>,%right:memref<7xi8>)->i8 {
  %first=arith.addi %a,%b:i8
  "state.config"(%a):(i8)->()
  %work=arith.muli %a,%b:i8
  %seen="state.read"():()->i8
  %result=arith.addi %first,%seen:i8
  %zero=arith.constant 0:index
  memref.store %a,%left[%zero]:memref<7xi8>
  memref.store %b,%right[%zero]:memref<7xi8>
  func.return %result:i8
}
func.func @untouched(%arg:i8)->i8 {func.return %arg:i8}
}
"""
        contract.update(standard_memory_sequential=True, index_bits=64)
        length = 9
    else:
        text = """module {
func.func @kernel(%a:i8,%b:i8)->i8 {
  %first=arith.addi %a,%b:i8
  "state.first"(%a):(i8)->()
  %work=arith.muli %a,%b:i8
  "state.second"(%b):(i8)->()
  %result=arith.addi %first,%work:i8
  func.return %result:i8
}
func.func @untouched(%arg:i8)->i8 {func.return %arg:i8}
}
"""
        contract = {
            "operations": [
                base.operation("state.first", ["i8"], writes=["first"], observable=True),
                base.operation("state.second", ["i8"], writes=["second"], observable=True),
            ],
            "disjoint_resources": [["first", "second"]],
        }
        length = 6
    selected = root / "contract.json"
    selected.write_text(json.dumps(contract, indent=2) + "\n")
    source = root / "source.mlir"
    source.write_text(text)
    first_order = [0, 2, 1, *range(3, length)]
    second_order = [1, 0, *range(2, length)]
    stages = []
    captures = []
    for index, order in enumerate((first_order, second_order)):
        actual_source = source if index == 0 else root / "middle.mlir"
        if index:
            actual_source.write_bytes(stages[0][1].read_bytes())
        destination = root / f"actual-{index}"
        process = subprocess.run(
            [
                sys.executable,
                str(repo_root() / "src/merlin/verify/lean_graph_state_capture.py"),
                "--reference",
                str(base.REFERENCE),
                "--source",
                str(actual_source),
                "--contract",
                str(selected),
                "--destination",
                str(destination),
                "--order",
                *map(str, order),
            ],
            cwd=base.REFERENCE,
            env={**os.environ, "PYTHONPATH": str(base.REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
        )
        (root / f"producer-{index}.stdout").write_text(process.stdout)
        (root / f"producer-{index}.stderr").write_text(process.stderr)
        assert process.returncode == 0, process.stderr
        capture = json.loads((destination / "capture.json").read_text())
        assert capture["projection_checked"]
        captures.append(capture)
        stages.append((actual_source, destination / "after.mlir"))
    return stages, selected, captures[0]


@pytest.mark.parametrize("kind", ["memory", "observers"])
def test_actual_effectful_chain_preserves_complete_direct_and_common_models(kind, tmp_path):
    edges, contract, capture = actual_effect_chain(tmp_path, kind)
    checked = direct.check_chain(
        edges,
        tmp_path / "direct",
        reference_root=base.REFERENCE,
        contract=contract,
        original_contract_sha256=capture["contract_sha256"],
    )
    assert checked["status"] == "graph_model_checked"
    composed = transport.check_composition(
        edges, tmp_path / "independent", reference_root=base.REFERENCE, contract=contract
    )
    assert composed["status"] == "kernel_model_checked"
    projection = json.loads((tmp_path / "independent/transport-1/projection/transport.json").read_text())
    identities = [row["identity"] for row in projection["source_cells"] if row["kind"] == "physical"]
    assert ("memory:caller" if kind == "memory" else "observation:ordered") in identities
    assert all(projection["renaming"]["forward"][index] == index for index in projection["renaming"]["physical"])


@pytest.mark.parametrize("kind", ["caller-alias", "memory-binding", "observer-order", "observer-binding", "untouched"])
def test_valid_actual_effectful_chain_mutation_refused(kind, tmp_path):
    edges, contract, capture = actual_effect_chain(
        tmp_path, "memory" if kind.startswith(("caller", "memory")) else "observers"
    )
    text = edges[-1][1].read_text()
    lines = text.splitlines()
    if kind in ("caller-alias", "observer-order"):
        needle = '"memref.store"' if kind == "caller-alias" else '"state.'
        positions = [index for index, line in enumerate(lines) if needle in line]
        assert len(positions) == 2
        lines[positions[0]], lines[positions[1]] = lines[positions[1]], lines[positions[0]]
        changed = "\n".join(lines) + "\n"
    elif kind == "memory-binding":
        changed = text.replace('"memref.store"(%a, %left,', '"memref.store"(%a, %right,')
    elif kind == "observer-binding":
        changed = text.replace('"state.first"(%a)', '"state.first"(%b)')
    else:
        changed = text.replace('sym_name = "untouched"', 'sym_name = "changed"')
    assert changed != text
    mutation = tmp_path / "mutated-final.mlir"
    mutation.write_text(changed)
    edges[-1] = (edges[-1][0], mutation)
    result = direct.check_chain(
        edges,
        tmp_path / "refused",
        reference_root=base.REFERENCE,
        contract=contract,
        original_contract_sha256=capture["contract_sha256"],
    )
    assert result["status"] == "unsupported" and result["failed_stage"] == 1
