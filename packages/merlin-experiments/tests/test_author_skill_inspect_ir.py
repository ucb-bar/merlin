"""The inspect-ir skill runs a submission's commands exactly as the grader does and checks the contracts."""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import pytest
import yaml
from inspect_skill_support import check_skill_layout, load_script, skill_dir

from merlin.common.paths import repo_root
from merlin.targetgen import package_runtime as PR
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

SKILL = "inspect-ir"
INTERFACE = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<16x16xi8>
  %R = merlin_iface.resident_pack %W {layout = "row_major"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %C = merlin_iface.matmul %A, %R : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %C {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}
"""
TOOL = '''#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
source = open(args[-1]).read()
assert "merlin_iface" in source
if "--parse" in args:
    sys.exit(0)
for a in args:
    if a.startswith("--cb="):
        open(a[5:], "w").write(json.dumps(DECLINE or {"commands": [{"opcode": "MATMUL"}]}))
if "--target" in args:
    print('builtin.module {\\n  "mine.op"() {mark = MARK} : () -> ()\\n}')
if "--llvm" in args:
    print("builtin.module {\\n  llvm.func @entry() {\\n    llvm.return\\n  }\\n}")
'''


def _package(root: Path, *, bundle: bool, mark: int = 1, decline=None) -> Path:
    root.mkdir(parents=True)
    tool = root / "opt-tool"
    tool.write_text(TOOL.replace("MARK", str(mark)).replace("DECLINE", repr(decline)))
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    commands = {"parse": {"argv": ["{tool}", "--parse", "{input_mlir}"]},
                "lower_interface_to_target": {"argv": ["{tool}", "--target", "{input_mlir}"]},
                "emit_command_buffer": {"argv": ["{tool}", "--cb={output_json}", "{input_mlir}"]},
                "lower_target_to_llvm": {"argv": ["{tool}", "--llvm", "{input_mlir}"]}}
    if bundle:
        commands["emit_analysis_bundle"] = {"argv": ["{tool}", "--cb={output_json}", "--llvm", "{input_mlir}"]}
    (root / "manifest.yaml").write_text(yaml.safe_dump({"language": "python", "entrypoints": {"tool": "opt-tool"},
                                                        "commands": commands}))
    return root


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "layers" / "CASE").mkdir(parents=True)
    (ws / "layers" / "CASE" / "capsule.interface.mlir").write_text(INTERFACE)
    for name in ("interface_dialect_contract.yaml", "target_dialect_contract.yaml"):
        shutil.copyfile(repo_root() / "merlin" / "contract" / name, ws / name)
    return ws


def test_layout_rules(tmp_path):
    check_skill_layout(SKILL, tmp_path)


def test_fixture_interface_is_valid_grammar():
    parsed = parse_interface_mlir(INTERFACE)
    assert [c["opcode"] for c in parsed["commands"]]


def test_argv_substitution_and_order_match_the_package_runtime(tmp_path):
    script = load_script(SKILL, "dump_stages")
    for bundle in (False, True):
        root = _package(tmp_path / f"pkg{bundle}", bundle=bundle)
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        pkg = PR.Package(directory=root, manifest=manifest, tool=(root / "opt-tool").resolve())
        inp, out = tmp_path / "in.mlir", tmp_path / "cb.json"
        for name in manifest["commands"]:
            assert script.resolve_argv(manifest["commands"][name]["argv"], tool=pkg.tool, input_mlir=inp.resolve(),
                                       output_json=out.resolve()) == PR._resolve_argv(pkg, name, inp, out)
        planned = [name for name, _ in script.plan(manifest)]
        assert planned[:2] == ["parse", "lower_interface_to_target"]
        assert tuple(planned[2:]) == PR.analysis_emission_entrypoints(pkg)
    bad = {"argv": ["{tool}", "{input_json}"]}
    with pytest.raises(SystemExit, match="only"):
        script.resolve_argv(bad["argv"], tool=Path("/t"), input_mlir=Path("/i"), output_json=None)
    pkg.manifest["commands"]["bad"] = bad
    with pytest.raises(PR.CertFailure):
        PR._resolve_argv(pkg, "bad", tmp_path / "in.mlir", None)
    assert set(script.PLACEHOLDERS) <= set(PR._resolve_argv.__code__.co_consts)


def test_run_compare_and_check_on_a_fixture_submission(tmp_path, capsys):
    script = load_script(SKILL, "dump_stages")
    ws = _workspace(tmp_path)
    _package(ws / "submission", bundle=True)
    _package(ws / "variants" / "v", bundle=False, mark=2)
    a, b = tmp_path / "a", tmp_path / "b"
    assert script.main(["run", "CASE", "--ws", str(ws), "--out", str(a)]) == 0
    assert script.main(["run", "CASE", "--ws", str(ws), "--submission", "variants/v", "--out", str(b)]) == 0
    record = json.loads((a / "stages.json").read_text())
    assert [s["stage"] for s in record["stages"]] == ["parse", "lower_interface_to_target", "emit_analysis_bundle"]
    assert set(record["outputs"]) == set(script.OUTPUTS)
    assert [s["stage"] for s in json.loads((b / "stages.json").read_text())["stages"]][2:] == [
        "emit_command_buffer", "lower_target_to_llvm"]
    capsys.readouterr()
    assert script.main(["compare", str(a), str(b)]) == 1
    assert "first differing stage output: lowered.target.mlir" in capsys.readouterr().out
    assert script.main(["compare", str(a), str(a)]) == 0
    assert "byte-identical" in capsys.readouterr().out
    assert script.main(["check", str(a), "--ws", str(ws)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["findings"] == [] and any("merlin_iface.matmul" in note for note in report["notes"])
    with pytest.raises(SystemExit, match="outside"):
        script.main(["run", "CASE", "--ws", str(ws), "--out", str(ws / "submission" / "x")])


def test_check_reports_contract_breaks(tmp_path, capsys):
    script = load_script(SKILL, "dump_stages")
    ws = _workspace(tmp_path)
    _package(ws / "submission", bundle=True, decline={"declined": {"reason": ""}})
    out = tmp_path / "d"
    assert script.main(["run", "CASE", "--ws", str(ws), "--out", str(out)]) == 0
    (out / "input.interface.mlir").write_text(INTERFACE.replace('version = "0.1"', 'version = "9"')
                                              .replace("merlin_iface.evict", "merlin_iface.unknown_op"))
    (out / "lowered.target.mlir").write_text("")
    capsys.readouterr()
    assert script.main(["check", str(out), "--ws", str(ws)]) == 1
    findings = "\n".join(json.loads(capsys.readouterr().out)["findings"])
    assert "merlin_iface.version" in findings and "merlin_iface.unknown_op" in findings
    assert "target IR is empty" in findings and "`declined` without a non-empty `reason`" in findings


def test_contract_fields_the_check_reads_exist():
    contract = yaml.safe_load((repo_root() / "merlin/contract/interface_dialect_contract.yaml").read_text())["dialect"]
    assert contract["namespace"] and any("must_equal" in a for a in contract["module_attributes"])
    assert all("name" in op for op in contract["required_ops"])
    target = yaml.safe_load((repo_root() / "merlin/contract/target_dialect_contract.yaml").read_text())
    script = load_script(SKILL, "dump_stages")
    planned = {name for bundle in (False, True) for name, _ in script.plan(
        {"commands": {"emit_analysis_bundle": {}} if bundle else {"lower_target_to_llvm": {}}})}
    assert {row["produced_by"] for row in target["required_outputs"]} <= planned
    text = (skill_dir(SKILL) / "SKILL.md").read_text()
    assert "entrypoints.tool" in text and "build.tool_output" in text and "commands.NAME.argv" in text
