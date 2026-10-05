"""Machine-dialect input scope must belong to the selected elaborated RTL."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from merlin.targetgen.cli import main as targetgen_main
from merlin.targetgen.dialect_source_scope import audit_dialect_source_scope
from merlin.targetgen.isa_census import derive_source_census
from merlin.targetgen.rtl import elaboration, source_selection


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "selected-source"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "configs.py").write_text("class SelectedConfig:\n    pass\n")
    (root / "emit.py").write_text(
        "import pathlib, sys\n"
        "assert sys.argv[1] == 'SelectedConfig'\n"
        "pathlib.Path(sys.argv[2]).write_text('FIRRTL version 3.3.0\\ncircuit Top :%[[]]\\n  module Top :\\n')\n"
    )
    (root / "Instructions.scala").write_text(
        'def ADD = BitPat("b00000000000000000000000000110011")\n'
    )
    controls = ["N"] * 17
    (root / "IDecode.scala").write_text("val table: Table = Array(\n  ADD -> List(" + ", ".join(controls) + ")\n)\n")
    (root / "isa_definition.py").write_text("class ADD(RType, opcode=0x33, funct3=0, funct7=0):\n    pass\n")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "source")
    revision = _git(root, "rev-parse", "HEAD")
    receipt = elaboration.issue(
        source_root=root, revision=revision, config_file="configs.py", config="SelectedConfig",
        command=[sys.executable, str(root / "emit.py"), "SelectedConfig", "{firrtl}"],
        output=tmp_path / "elaboration",
    )
    firtool = tmp_path / "firtool"
    firtool.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[-1]).write_text('module {\\n  hw.module @Top() {\\n    hw.output\\n  }\\n}\\n')\n"
    )
    firtool.chmod(0o755)
    selection = source_selection.produce_selection(
        target="fixture", firrtl=receipt.parent / "run1" / "selected.fir",
        generator="fixture", config="SelectedConfig", core_root="Top", firtool=firtool,
        output=tmp_path / "selection", elaboration_receipt=receipt,
    )
    census = derive_source_census(
        pattern_file=root / "Instructions.scala", decoder_file=root / "IDecode.scala",
        model_isa_file=root / "isa_definition.py", rtl_revision=revision,
        model_revision=revision, verify_revisions=True,
    )
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    inventory = {
        "target": "fixture",
        "selected_sources": {"rtl_revision": revision, "model_revision": revision},
        "parameter_domains": {"registers": {
            "kind": "integer", "unit": "register_index", "reviewed": True,
            "evidence_sources": ["Instructions.scala"],
            "intervals": [{"min": 0, "max": 7, "step": 1}],
        }},
        "variants": [{
            "id": "ADD", "rtl_bitpat": census["rows"][0]["pattern_bits"],
            "rtl_decode_controls": controls, "family": "scalar_control", "required": True,
            "parameter_domains": ["registers"], "model_classes": ["ADD"],
            "architecture_sources": ["Instructions.scala", "IDecode.scala"],
        }],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    return selection, census_path, inventory_path, root


def test_source_scope_replays_pinned_elaboration_and_census(tmp_path: Path) -> None:
    selection, census, inventory, _ = _inputs(tmp_path)
    report = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory,
        expected_config="SelectedConfig",
    )
    assert report["status"] == "ready", report["blockers"]
    assert report["selected_rtl_checkout"] == "."
    assert report["mode_counts"]["required_modes"] == 1
    assert {row["path_at_revision"] for row in report["architecture_sources"]} == {
        "Instructions.scala", "IDecode.scala",
    }
    assert report["blockers"] == []


def test_copied_decoder_checkout_does_not_inherit_selected_elaboration(tmp_path: Path) -> None:
    selection, census_path, inventory, root = _inputs(tmp_path)
    copied = tmp_path / "other-checkout"
    subprocess.run(["git", "clone", "-q", str(root), str(copied)], check=True)
    census = derive_source_census(
        pattern_file=copied / "Instructions.scala", decoder_file=copied / "IDecode.scala",
        model_isa_file=copied / "isa_definition.py", rtl_revision=_git(root, "rev-parse", "HEAD"),
        model_revision=_git(root, "rev-parse", "HEAD"), verify_revisions=True,
    )
    census_path.write_text(json.dumps(census))
    report = audit_dialect_source_scope(
        selection_path=selection, census_path=census_path, inventory_path=inventory,
        expected_config="SelectedConfig",
    )
    assert report["status"] == "blocked"
    assert "census RTL sources do not belong to exactly one pinned elaboration checkout" in report["blockers"]


def test_forged_census_and_open_mode_scope_fail(tmp_path: Path) -> None:
    selection, census_path, inventory_path, _ = _inputs(tmp_path)
    census = json.loads(census_path.read_text())
    changed = copy.deepcopy(census)
    changed["rows"][0]["family"] = "forged"
    census_path.write_text(json.dumps(changed))
    report = audit_dialect_source_scope(
        selection_path=selection, census_path=census_path, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert "selected ISA census differs from a fresh pinned-source replay" in report["blockers"]
    census_path.write_text(json.dumps(census))
    inventory = json.loads(inventory_path.read_text())
    inventory["variants"] = []
    inventory_path.write_text(json.dumps(inventory))
    report = audit_dialect_source_scope(
        selection_path=selection, census_path=census_path, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert not report["mode_scope_ready"]
    assert "selected decoder mode population or source discrepancy review is incomplete" in report["blockers"]


def test_config_label_without_elaboration_receipt_is_blocked(tmp_path: Path) -> None:
    selection_path, census, inventory, _ = _inputs(tmp_path)
    selection = json.loads(selection_path.read_text())
    del selection["production"]["elaboration"]
    unbound = tmp_path / "unbound-selection.json"
    unbound.write_text(json.dumps(selection))
    report = audit_dialect_source_scope(
        selection_path=unbound, census_path=census, inventory_path=inventory,
        expected_config="SelectedConfig",
    )
    assert report["source_consistency_status"] == "verified"
    assert report["elaboration_status"] == "not_selected"
    assert report["status"] == "blocked"
    assert "selected configuration-to-FIRRTL elaboration is not reproduced" in report["blockers"]


def test_wrong_campaign_configuration_or_ledger_target_is_blocked(tmp_path: Path) -> None:
    selection, census, inventory_path, _ = _inputs(tmp_path)
    wrong_config = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="OtherConfig",
    )
    assert "selected source configuration differs from requested campaign" in wrong_config["blockers"]
    inventory = json.loads(inventory_path.read_text())
    inventory["target"] = "other_target"
    inventory_path.write_text(json.dumps(inventory))
    wrong_target = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert "mode ledger target differs from selected source target" in wrong_target["blockers"]


def test_required_mode_architecture_sources_must_be_pinned(tmp_path: Path) -> None:
    selection, census, inventory_path, root = _inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text())
    inventory["variants"][0]["architecture_sources"] = []
    inventory_path.write_text(json.dumps(inventory))
    missing = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert "ADD: architecture source references are absent or invalid" in missing["blockers"]

    inventory["variants"][0]["architecture_sources"] = ["../other-checkout/Instructions.scala"]
    inventory_path.write_text(json.dumps(inventory))
    escaping = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert "ADD: architecture source references are absent or invalid" in escaping["blockers"]

    inventory["variants"][0]["architecture_sources"] = ["not-present.scala"]
    inventory_path.write_text(json.dumps(inventory))
    absent = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert any("architecture: source not-present.scala is not pinned" in item for item in absent["blockers"])

    inventory["variants"][0]["architecture_sources"] = ["Instructions.scala"]
    inventory_path.write_text(json.dumps(inventory))
    (root / "Instructions.scala").write_text('def ADD = BitPat("b00000000000000000000000000000000")\n')
    changed = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert changed["status"] == "blocked"
    assert changed["elaboration_status"] == "unverified"


def test_parameter_domain_evidence_must_be_in_selected_checkout(tmp_path: Path) -> None:
    selection, census, inventory_path, _ = _inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text())
    inventory["parameter_domains"]["registers"]["evidence_sources"] = ["missing_geometry.scala"]
    inventory_path.write_text(json.dumps(inventory))
    report = audit_dialect_source_scope(
        selection_path=selection, census_path=census, inventory_path=inventory_path,
        expected_config="SelectedConfig",
    )
    assert report["mode_scope_ready"]
    assert report["parameter_domains_ready"]
    assert report["status"] == "blocked"
    assert any(
        "parameter domain registers: source missing_geometry.scala is not pinned" in item
        for item in report["blockers"]
    )


def test_cli_status_and_failed_invocation_cannot_leave_stale_success(tmp_path: Path, capsys) -> None:
    selection, census, inventory, _ = _inputs(tmp_path)
    output = tmp_path / "scope.json"
    args = [
        "audit-dialect-source-scope", "--source-selection", str(selection),
        "--census", str(census), "--inventory", str(inventory),
        "--expected-config", "SelectedConfig", "--out", str(output),
    ]
    assert targetgen_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY"
    assert json.loads(output.read_text())["status"] == "ready"
    census.unlink()
    assert targetgen_main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert not output.exists()
