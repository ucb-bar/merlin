"""Independent original selection and live proof reproduction; JSON is not authority."""

import json
from pathlib import Path

import pytest
from test_lean_graph_state_elision import REFERENCE, checked, produce

from merlin.verify import lean_graph_state as state
from merlin.verify import lean_graph_state_elision_receipt as receipt


def setup(root):
    before, after, selected, _ = produce(root)
    report = checked(root, before, after, selected)
    assert report["status"] == "conditional_model_checked"
    path = root / "receipt.json"
    receipt.create(root / "checked", path)
    return path, dict(
        originals=[state.file_ref(before), state.file_ref(after)],
        contract=selected,
        original_contract_sha256=state.file_ref(selected)["sha256"],
        reference_root=REFERENCE,
    )


def test_receipt_fresh_actual_replay(tmp_path):
    path, selection = setup(tmp_path)
    result = receipt.replay(path, tmp_path / "replayed", **selection)
    assert result["status"] == "conditional_model_checked" and result["exact_old_object_reproduced"]
    assert result["host_protocol_proved_in_Lean"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "version-bool",
        "unknown-field",
        "approval",
        "stale-original",
        "stale-proof",
        "coherent-query",
        "coherent-object",
        "coherent-audit",
        "coherent-statement",
        "missing-dependency",
        "snapshot",
        "wrong-original-selection",
        "coherent-originals",
        "coherent-snapshot",
        "coherent-missing-dependency",
    ],
)
def test_receipt_stale_and_coherent_forgeries_refused(kind, tmp_path):
    path, selection = setup(tmp_path)
    data = json.loads(path.read_text())
    root = tmp_path / "checked"
    if kind == "version-bool":
        data["version"] = True
    elif kind == "unknown-field":
        data["invented_approval"] = False
    elif kind == "approval":
        data["claims"]["admission_approved"] = True
    elif kind == "stale-original":
        Path(selection["originals"][0]["path"]).write_text("kernel changed()\n")
    elif kind in ("stale-proof", "coherent-query"):
        q = root / "kernel/GraphStateElisionInstance.lean"
        q.write_text(q.read_text() + "\n-- substituted proof buffer\n")
    elif kind == "coherent-object":
        (root / "kernel/GraphStateElisionInstance.olean").write_bytes(b"coherently rehashed invalid object")
    elif kind in ("coherent-audit", "coherent-statement"):
        q = root / "kernel/report.json"
        report = json.loads(q.read_text())
        if kind == "coherent-audit":
            report["audit"]["raw"]["theorems_audited"] += 1
            (root / "kernel/audit.stdout").write_text(json.dumps(report["audit"]["raw"]))
        else:
            report["audit"]["statements"]["Merlin.GraphStateElisionInstance.actual_elision"] = "True"
            statements = root / "kernel/statements.stdout"
            raw = json.loads(statements.read_text())
            raw["statements"] = report["audit"]["statements"]
            statements.write_text(json.dumps(raw))
        q.write_text(json.dumps(report))
    elif kind == "missing-dependency":
        data["dependencies"].pop()
    elif kind == "coherent-missing-dependency":
        report = root / "report.json"
        changed = json.loads(report.read_text())
        removed = data["dependencies"].pop()
        changed["dependencies"] = [row for row in changed["dependencies"] if row["path"] != removed["path"]]
        report.write_text(json.dumps(changed))
    elif kind == "coherent-originals":
        for row in selection["originals"]:
            q = Path(row["path"])
            q.write_text(q.read_text() + "  observe(value=91)\n")
        selection["originals"] = [state.file_ref(row["path"]) for row in selection["originals"]]
    elif kind == "coherent-snapshot":
        q = root / "before.mk"
        q.write_text(q.read_text() + "  observe(value=91)\n")
    elif kind == "snapshot":
        (root / "before.mk").write_text("kernel commands()\n")
    else:
        selection["originals"] = list(reversed(selection["originals"]))
    if kind.startswith("coherent-"):
        # Rehash EVERY associated mutable record, including the copied report.
        def refresh(value):
            if type(value) is dict and set(value) == {"path", "sha256", "bytes"}:
                return state.file_ref(value["path"])
            if type(value) is dict:
                return {key: refresh(item) for key, item in value.items()}
            if type(value) is list:
                return [refresh(item) for item in value]
            return value

        kernel = root / "kernel/report.json"
        kernel.write_text(json.dumps(refresh(json.loads(kernel.read_text()))))
        report = root / "report.json"
        report.write_text(json.dumps(refresh(json.loads(report.read_text()))))
        data = refresh(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        receipt.replay(path, tmp_path / "refused", **selection)
