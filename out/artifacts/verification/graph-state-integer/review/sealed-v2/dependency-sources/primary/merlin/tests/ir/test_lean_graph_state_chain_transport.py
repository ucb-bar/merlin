"""Checked actual reindexing: ordered slots, physical identity and composition."""

from __future__ import annotations

import copy
import json
import os
import subprocess

import pytest
import test_lean_graph_state_chain as fixture

from merlin.verify import lean_graph_state as stable
from merlin.verify import lean_graph_state_chain_transport as transport
from merlin.verify.lean_graph_state_chain_kernel import prepare


@pytest.fixture(scope="module")
def actual_projection(tmp_path_factory):
    root = tmp_path_factory.mktemp("transport-actual")
    edges, contract, _ = fixture.actual_chain(root)
    result = transport.check_composition(
        edges, root / "checked", reference_root=fixture.fixture.REFERENCE, contract=contract
    )
    assert result["status"] == "kernel_model_checked"
    projection = json.loads((root / "checked/transport-1/projection/transport.json").read_text())
    return root, projection


def test_actual_independent_composition_has_checked_nonidentity_mapping(actual_projection):
    root, projection = actual_projection
    assert any(index != actual for index, actual in enumerate(projection["renaming"]["forward"]))
    for key in projection["renaming"]["physical"]:
        assert projection["renaming"]["forward"][key] == key
    report = json.loads((root / "checked/kernel/report.json").read_text())
    assert report["status"] == "kernel_model_checked"
    assert not report["audit"]["raw"]["violations"]
    assert set(report["audit"]["raw"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}


@pytest.mark.parametrize(
    "kind",
    [
        "wrong-same-type-map",
        "missing-map",
        "missing-physical",
        "field-identity",
        "operand-slots",
        "result-slots",
        "result-type",
        "cell-type",
        "missing-operation",
        "order",
    ],
)
def test_kernel_proves_mutated_correspondence_refused(kind, actual_projection, tmp_path):
    _, original = actual_projection
    value = copy.deepcopy(original)
    if kind == "wrong-same-type-map":
        value["renaming"]["forward"][1], value["renaming"]["forward"][2] = (
            value["renaming"]["forward"][2],
            value["renaming"]["forward"][1],
        )
        value["renaming"]["backward"] = [
            value["renaming"]["forward"].index(key) for key in range(len(value["renaming"]["forward"]))
        ]
    elif kind == "missing-map":
        value["renaming"]["forward"].pop()
    elif kind == "missing-physical":
        value["renaming"]["physical"] = []
    elif kind == "field-identity":
        value["target_cells"][0]["identity"] = "declared:substituted-field"
    elif kind == "operand-slots":
        primitive = next(row for row in value["target_primitives"] if row["core"]["operation"] == "state.config")
        primitive["inputs"].reverse()
    elif kind == "result-slots":
        value["target_primitives"][0]["outputs"] = value["target_primitives"][1]["outputs"]
    elif kind == "result-type":
        value["target_primitives"][0]["core"]["resultTypes"] = ["i16"]
    elif kind == "cell-type":
        value["target_cells"][1]["type"] = "i16"
    elif kind == "missing-operation":
        value["target_primitives"].pop()
    else:
        value["order"][0], value["order"][1] = value["order"][1], value["order"][0]
    declarations = transport.definitions(value, "Actual").split("theorem accepted", 1)[0]
    query = tmp_path / "GraphStateChainRefusal.lean"
    query.write_text(
        "import Merlin.GraphStateChain\nnamespace Merlin.GraphStateChainRefusal\n"
        "open Merlin.StateEffects Merlin.GraphState.Chain\n"
        + declarations
        + "theorem refused : checkTransport mapping source target order sourceCells targetCells = false := by decide\n"
        "#print axioms refused\nend Actual\nend Merlin.GraphStateChainRefusal\n"
    )
    frozen = prepare()
    process = subprocess.run(
        [str(stable.LEAN), str(query)],
        cwd=tmp_path,
        env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
        capture_output=True,
        text=True,
    )
    (tmp_path / "kernel.stdout").write_text(process.stdout)
    (tmp_path / "kernel.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "sorryAx" not in process.stdout and "refused" in process.stdout


@pytest.mark.parametrize("kind", ["boolean-index", "float-index", "wrong-type-list"])
def test_malformed_transport_field_refused(kind, actual_projection):
    _, original = actual_projection
    value = copy.deepcopy(original)
    if kind == "boolean-index":
        value["renaming"]["forward"][0] = True
    elif kind == "float-index":
        value["renaming"]["forward"][0] = 0.0
    else:
        value["source_primitives"][0]["core"]["operandTypes"] = "i8"
    with pytest.raises(ValueError):
        transport.emit_query(value)
