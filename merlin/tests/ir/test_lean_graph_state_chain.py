"""Actual consecutive production schedules and checked complete handoffs."""

from __future__ import annotations

import json

import pytest
import test_lean_graph_state as fixture

from merlin.verify import lean_graph_state_chain as chain


def actual_chain(root, width=8, tail=False, arrangement="cross-config"):
    first = root / "first"
    first.mkdir()
    first_order = [0, 2, 1, 3, 4, 5] if arrangement == "cross-config" else [2, 0, 1, 3, 4, 5]
    if tail:
        first_order.append(6)
    before, middle, contract, produced = fixture.actual(first, width, tail, first_order)
    second = root / "second"
    second.mkdir()
    source = second / "source.mlir"
    source.write_bytes(middle.read_bytes())
    selected = second / "contract.json"
    selected.write_bytes(contract.read_bytes())
    # Actual second production transformation consumes EXACT first output.
    command = [
        fixture.sys.executable,
        str(fixture.repo_root() / "src/merlin/verify/lean_graph_state_capture.py"),
        "--reference",
        str(fixture.REFERENCE),
        "--source",
        str(source),
        "--contract",
        str(selected),
        "--destination",
        str(second / "actual"),
        "--order",
        *map(str, [1, 0, 2, 3, 4, 5, 6] if tail else [1, 0, 2, 3, 4, 5]),
    ]
    process = fixture.subprocess.run(
        command,
        cwd=fixture.REFERENCE,
        env={**fixture.os.environ, "PYTHONPATH": str(fixture.REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    (second / "producer.stdout").write_text(process.stdout)
    (second / "producer.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stderr
    capture = json.loads((second / "actual/capture.json").read_text())
    assert capture["projection_checked"] and capture["producer"].endswith("planning.apply_order")
    return [(before, middle), (source, second / "actual/after.mlir")], contract, produced


@pytest.mark.parametrize(
    "width,tail,arrangement", [(8, False, "cross-config"), (16, False, "cross-config"), (32, True, "two-producers")]
)
def test_actual_two_boundary_chain_direct_complete_proof(width, tail, arrangement, tmp_path):
    edges, contract, produced = actual_chain(tmp_path, width, tail, arrangement)
    result = chain.check_chain(
        edges,
        tmp_path / "checked",
        reference_root=fixture.REFERENCE,
        contract=contract,
        original_contract_sha256=produced["contract_sha256"],
    )
    assert result["status"] == "graph_model_checked" and result["complete_chain_checked"]
    assert not result["stage_state_numberings_identified"] and not result["public_acceptance_issued"]
    direct = json.loads((tmp_path / "checked/complete/report.json").read_text())
    assert direct["actual_exchanges"] == (2 if arrangement == "cross-config" else 1)
    assert direct["transitive_audit"]["status"] == "checked"


def test_actual_chain_rejects_changed_middle_handoff(tmp_path):
    edges, contract, produced = actual_chain(tmp_path)
    middle = edges[1][0]
    middle.write_text(middle.read_text().replace('sym_name = "untouched"', 'sym_name = "changed"'))
    with pytest.raises(ValueError, match="identical actual intermediate"):
        chain.check_chain(
            edges,
            tmp_path / "checked",
            reference_root=fixture.REFERENCE,
            contract=contract,
            original_contract_sha256=produced["contract_sha256"],
        )


def test_actual_chain_rejects_rebound_original_semantics(tmp_path):
    edges, contract, produced = actual_chain(tmp_path)
    data = json.loads(contract.read_text())
    data["operations"][0]["observable"] = True
    contract.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contract differs"):
        chain.check_chain(
            edges,
            tmp_path / "checked",
            reference_root=fixture.REFERENCE,
            contract=contract,
            original_contract_sha256=produced["contract_sha256"],
        )


def test_actual_chain_rejects_changed_final_binding(tmp_path):
    edges, contract, produced = actual_chain(tmp_path)
    final = edges[-1][1]
    text = final.read_text()
    changed = text.replace('"arith.addi"(%first, %seen)', '"arith.addi"(%work, %seen)')
    assert changed != text
    final.write_text(changed)
    result = chain.check_chain(
        edges,
        tmp_path / "checked",
        reference_root=fixture.REFERENCE,
        contract=contract,
        original_contract_sha256=produced["contract_sha256"],
    )
    assert result["status"] == "unsupported" and result["failed_stage"] == 1
    assert not result["complete_chain_checked"] and not (tmp_path / "checked/complete").exists()
