"""The inspect-kernel skill applies the grader's stack-frame rule and reads the grader's loop and task records."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
import yaml
from inspect_skill_support import check_skill_layout, load_script, skill_dir

from merlin.targetgen.contract import stack_usage as SU
from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy

SKILL = "inspect-kernel"
#: Two nested counted loops (4 x 3) around one site each, plus a site outside any loop.
LOWERED = """builtin.module {
  llvm.func @entry(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> {merlin.global_task = -1 : i64} : () -> ()
    llvm.br ^bb1(%0 : i64) {merlin.global_task = 0 : i64}
  ^bb1(%4: i64):
    %5 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %5, ^bb2, ^bb6
  ^bb2:
    "llvm.inline_asm"(%4, %4) <{asm_string = "site $0, $1", constraints = "r,r,~{memory}", has_side_effects}> {merlin.global_task = 0 : i64} : (i64, i64) -> ()
    llvm.br ^bb3(%0 : i64)
  ^bb3(%6: i64):
    %7 = llvm.icmp "slt" %6, %3 : i64
    llvm.cond_br %7, ^bb4, ^bb5
  ^bb4:
    "llvm.inline_asm"(%6, %6) <{asm_string = "site $0, $1", constraints = "r,r,~{memory}", has_side_effects}> {merlin.global_task = 1 : i64, prov.region_id = "r1"} : (i64, i64) -> ()
    %8 = llvm.add %6, %1 : i64
    llvm.br ^bb3(%8 : i64)
  ^bb5:
    %9 = llvm.add %4, %1 : i64
    llvm.br ^bb1(%9 : i64)
  ^bb6:
    llvm.return
  }
}
"""


def test_layout_rules(tmp_path):
    check_skill_layout(SKILL, tmp_path)


@pytest.mark.parametrize("report,entry,budget", [
    ("kernel.ll:entry\t96\tstatic\nkernel.ll:helper\t0\tstatic\n", "entry", 128),
    ("kernel.ll:entry\t200\tstatic\n", "entry", 128),
    ("kernel.ll:entry\t16\tstatic\nkernel.ll:helper\t32\tdynamic\n", "entry", 128),
    ("kernel.ll:entry\t16\tstatic\nkernel.ll:entry\t16\tstatic\n", "entry", 128),
    ("kernel.ll:other\t16\tstatic\n", "entry", 128),
    ("elsewhere.ll:entry\t16\tstatic\n", "entry", 128),
])
def test_stack_rule_agrees_with_the_grader(tmp_path, report, entry, budget):
    script = load_script(SKILL, "build_kernel")
    su = tmp_path / "kernel.su"
    su.write_text(report)
    try:
        SU.measure_entrypoint(su, llvm_path=tmp_path / "kernel.ll", entry_symbol=entry, max_static_bytes=budget)
        refused = False
    except SU.StackFramePreflightError:
        refused = True
    verdict = script.stack_verdict(script.parse_stack_usage(report), entry=entry, max_stack=budget)
    assert bool(verdict["refusals"]) is refused, verdict
    assert [(r.function, r.frame_bytes, r.allocation) for r in SU.parse_stack_usage(report)] == [
        (r["function"], r["bytes"], r["allocation"]) for r in script.parse_stack_usage(report)]


def test_recipe_is_read_from_the_toolchain_block(tmp_path):
    script = load_script(SKILL, "build_kernel")
    assert {f.name for f in dataclasses.fields(KernelStackFramePolicy)} == {"entry_symbol", "max_static_bytes"}
    contract = {"runner": {"backend": {"toolchain": {
        "cflags": ["-march=rvX", "-mabi=abiY", "-mcmodel=medany", "-O2", "-std=gnu11", "-nostdlib", "-ffreestanding"],
        "compiler": {"path": "/opt/bin/arch-vendor-elf-gcc"},
        "kernel_stack_frame": {"entry_symbol": "entry", "max_static_bytes": 4096}}}}}
    (tmp_path / "target_contract.yaml").write_text(yaml.safe_dump(contract))
    recipe = script.recipe_from_contract(tmp_path / "target_contract.yaml")
    assert recipe == {"cflags": ["-march=rvX", "-mabi=abiY", "-mcmodel=medany", "-O2", "-ffreestanding"],
                      "triple": "arch-vendor-elf", "entry": "entry", "max_stack": 4096}
    assert script.recipe_from_contract(tmp_path / "absent.yaml") == {}


def test_nm_and_objdump_parsing():
    script = load_script(SKILL, "build_kernel")
    nm = "0000000000000000 T entry\n0000000000000040 t helper\n                 U memcpy\n0000000000000080 T memset\n"
    assert script.parse_nm(nm) == {"defined": {"entry": "T", "helper": "t", "memset": "T"}, "undefined": ["memcpy"],
                                   "external": ["entry", "memset"]}
    dis = ("0000000000000000 <entry>:\n       0: 13 01 01 fe   addi sp, sp, -32\n"
           "0000000000000004 <.Lpcrel_hi0>:\n       4: 97 05 00 00   auipc a1, 0\n"
           "\t\t0000000000000004:  R_X_PCREL_HI20 sym\n0000000000000008 <helper>:\n       8: 67 80 00 00   ret\n")
    assert script.function_sizes(dis) == {"entry": 2, "helper": 1}


def test_build_kernel_end_to_end_when_the_toolchain_is_present(tmp_path):
    from merlin.llvmlower import codegen
    from merlin.targetgen.contract import toolchain

    triple = next(f.partition("=")[2] for f in codegen.RISCV_FLAGS if f.startswith("--target="))
    script = load_script(SKILL, "build_kernel")
    names = ("mlir-translate", "clang", "llvm-nm", "llvm-objdump")
    found = shutil.which("mlir-translate")
    candidates = [toolchain.mlir_install() / "bin", *([Path(found).parent] if found else [])]
    bindir = next((d for d in candidates if all((d / name).is_file() for name in names)), None)
    if bindir is None:
        pytest.skip("LLVM toolchain not available")
    case = tmp_path / "case"
    case.mkdir()
    (case / "lowered.llvm.mlir").write_text(LOWERED.replace('"site $0, $1"', '"nop"').replace(
        'constraints = "r,r,~{memory}"', 'constraints = "r,r"'))
    code = script.main([str(case), "--ws", str(tmp_path), "--out", str(tmp_path / "out"), "--toolchain", str(bindir),
                        "--triple",
                        triple, "--entry", "entry", "--max-stack", "4096", "--cflag=-O2"])
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert code == 0 and report["stack"]["refusals"] == [] and report["symbols"]["external_defined"] == ["entry"]


def test_loop_map_uses_the_graders_loop_records(tmp_path):
    from merlin.targetgen.rocc.decode import _parse_module
    from merlin.targetgen.rocc.static_operands import counted_loop_observations

    script = load_script(SKILL, "loop_map")
    observation, _ = counted_loop_observations(_parse_module(LOWERED))
    loops = {loop["header_block"]: loop["trip_count"] for loop in observation["functions"][0]["loops"]}
    assert loops == {1: 4, 3: 3}
    (tmp_path / "lowered.llvm.mlir").write_text(LOWERED)
    (tmp_path / "instruction_trace.json").write_text(json.dumps({
        "instructions": [{"index": 0, "class": "FENCE"}, {"index": 1, "class": "A"}, {"index": 2, "class": "B"}],
        "summary": {"counted_loops": observation}}))
    (tmp_path / "command_buffer.json").write_text(json.dumps({"commands": [{"opcode": "P"}, {"opcode": "Q"}]}))
    report = script.loop_map(tmp_path)
    by_header = {row["header"]: row for row in report["loops"]}
    assert by_header[3]["issued_estimate"] == 12 and by_header[3]["depth"] == 1
    assert by_header[3]["tasks"] == {"1": "commands[1] Q"} and by_header[3]["regions"] == ["r1"]
    assert by_header[1]["issued_estimate"] == 16 and by_header[1]["issued_by_class"] == {"B": 12, "A": 4}
    assert report["issued_by_class"] == {"B": 12, "A": 4, "FENCE": 1} and report["notes"] == []
    plan = {"params": {"global_program_plan": {"tasks": [{"task_index": 1, "kind": "device",
                                                          "source_op_indices": [0]}]}}}
    (tmp_path / "command_buffer.json").write_text(json.dumps(plan))
    assert script.loop_map(tmp_path)["loops"][1]["tasks"]["1"].startswith("plan task 1: device")


def test_documented_rules_match_the_grader_messages():
    text = (skill_dir(SKILL) / "reference" / "products.md").read_text()
    source = (SU.__file__ and open(SU.__file__, encoding="utf-8").read())
    for fragment in ("exactly one is required", "static stack bytes", "contains dynamic allocation for"):
        assert fragment in source and fragment in text, fragment
    assert "kernel_stack_frame" in (skill_dir(SKILL) / "SKILL.md").read_text()
