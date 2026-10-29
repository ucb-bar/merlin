"""Live original/middle/proof binding; stale and coherent forgeries grant no authority."""

import json
from pathlib import Path

import pytest
from test_lean_graph_state_scalar import CASES, REFERENCE, produce

from merlin.verify import lean_graph_state as stable
from merlin.verify import lean_graph_state_scalar as checker
from merlin.verify import lean_graph_state_scalar_receipt as receipt


def setup(root, chain=False):
    mode, text = CASES["chain"] if chain else CASES["cse-i8"]
    before, after, projected = produce(root, text, mode)
    assert projected["projection_checked"], projected
    if chain:
        middle = root / "actual/middle.mlir"
        selected = [stable.file_ref(path) for path in (before, middle, after)]
        result = checker.check_chain(
            before, middle, after, root / "checked", reference_root=REFERENCE, expected_originals=selected
        )
    else:
        selected = [stable.file_ref(before), stable.file_ref(after)]
        result = checker.check_boundary(
            before, after, root / "checked", reference_root=REFERENCE, expected_originals=selected
        )
    assert result["status"] == "conditional_graph_checked", result
    path = root / "receipt.json"
    receipt.create(root / "checked", path)
    return path, selected


@pytest.mark.parametrize("chain", [False, True])
def test_fresh_live_scalar_receipt_and_actual_chain_replay(chain, tmp_path):
    path, selected = setup(tmp_path, chain)
    result = receipt.replay(path, tmp_path / "replay", originals=selected, reference_root=REFERENCE)
    assert result["status"] == "conditional_graph_checked" and result["retained_object_reproduced"]
    assert not result["host_transport_proved_in_Lean"]


def refresh(value):
    if type(value) is dict and set(value) == {"path", "sha256", "bytes"}:
        return stable.file_ref(value["path"])
    if type(value) is dict:
        return {key: refresh(item) for key, item in value.items()}
    if type(value) is list:
        return [refresh(item) for item in value]
    return value


@pytest.mark.parametrize(
    "kind",
    [
        "version-bool",
        "extra-field",
        "mode",
        "approval",
        "tool",
        "profile",
        "stale-original",
        "stale-proof",
        "duplicate-json",
        "coherent-originals",
        "coherent-snapshot",
        "coherent-model",
        "coherent-query",
        "coherent-object",
        "coherent-audit",
        "coherent-statements",
        "coherent-imports",
        "coherent-dependencies",
        "coherent-changed-label",
        "wrong-middle-selection",
    ],
)
def test_scalar_receipt_forgeries_refused(kind, tmp_path):
    path, selected = setup(tmp_path, chain=kind == "wrong-middle-selection")
    data = json.loads(path.read_text())
    root = tmp_path / "checked"
    if kind == "version-bool":
        data["version"] = True
    elif kind == "extra-field":
        data["invented_approval"] = False
    elif kind == "mode":
        data["mode"] = "actual-two-stage-chain"
    elif kind == "approval":
        data["claims"]["source_success_proved"] = True
    elif kind == "tool":
        data["tool"]["sha256"] = "0" * 64
    elif kind == "profile":
        data["profile"]["sha256"] = "0" * 64
    elif kind in ("stale-original", "coherent-originals"):
        for row in selected:
            p = Path(row["path"])
            p.write_text(p.read_text().replace("arith.addi", "arith.subi"))
        if kind.startswith("coherent"):
            selected = [stable.file_ref(row["path"]) for row in selected]
    elif kind in ("stale-proof", "coherent-query"):
        p = root / "kernel/GraphStateScalarInstance.lean"
        p.write_text(p.read_text() + "\n-- coherently rehashed substituted query buffer\n")
    elif kind == "coherent-snapshot":
        p = root / "before.mlir"
        p.write_text(p.read_text().replace("arith.addi", "arith.subi"))
    elif kind == "coherent-model":
        p = root / "projection/projection.json"
        model = json.loads(p.read_text())
        model["functions"][0]["after"][0]["pure"]["kind"] = "sub"
        p.write_text(json.dumps(model))
        (root / "kernel/GraphStateScalarInstance.lean").write_text(checker.emit_query(model)[0])
    elif kind == "coherent-object":
        (root / "kernel/GraphStateScalarInstance.olean").write_bytes(b"coherently rehashed invalid object")
    elif kind in ("coherent-audit", "coherent-statements", "coherent-imports"):
        p = root / "kernel/report.json"
        kernel = json.loads(p.read_text())
        audit = kernel["audit"]
        if kind == "coherent-audit":
            audit["raw"]["theorems_audited"] += 1
        elif kind == "coherent-statements":
            audit["statements"]["Merlin.GraphStateScalarInstance.F0.actual"] = "True"
            statements = root / "kernel/statements.stdout"
            value = json.loads(statements.read_text())
            value["statements"] = audit["statements"]
            statements.write_text(json.dumps(value))
        else:
            removed = audit["raw"]["imports"].pop()
            audit["imported_objects"] = [row for row in audit["imported_objects"] if row["path"] != removed["olean"]]
        (root / "kernel/audit.stdout").write_text(json.dumps(audit["raw"]))
        p.write_text(json.dumps(kernel))
    elif kind == "coherent-dependencies":
        removed = data["dependencies"].pop()
        p = root / "report.json"
        report = json.loads(p.read_text())
        report["dependencies"] = [row for row in report["dependencies"] if row["path"] != removed["path"]]
        p.write_text(json.dumps(report))
    elif kind == "coherent-changed-label":
        p = root / "report.json"
        report = json.loads(p.read_text())
        report["genuine_changed_boundary"] = False
        p.write_text(json.dumps(report))
        p = root / "projection/projection.json"
        model = json.loads(p.read_text())
        model["genuine_changed_boundary"] = False
        p.write_text(json.dumps(model))
    elif kind == "wrong-middle-selection":
        selected[1] = selected[0]
    if kind.startswith("coherent-"):
        for p in (root / "kernel/report.json", root / "report.json"):
            p.write_text(json.dumps(refresh(json.loads(p.read_text()))))
        data = refresh(data)
    path.write_text(json.dumps(data))
    if kind == "duplicate-json":
        path.write_text(path.read_text()[:-1] + ',"version":1}')
    with pytest.raises(ValueError):
        receipt.replay(path, tmp_path / "refused", originals=selected, reference_root=REFERENCE)
