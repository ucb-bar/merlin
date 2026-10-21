"""Structural region checks; arithmetic and opaque-call meanings stay external."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.llvm_region_boundary import LLVMRegionBoundary


def source(bits=64, count=7):
    return f"""declare void @touch(ptr, i{bits})
define void @work(ptr %input, i{bits} %limit) {{
entry:
  call void @touch(ptr %input, i{bits} 0)
  br label %header
header:
  %row = phi i{bits} [ 0, %entry ], [ %next, %latch ]
  %more = icmp slt i{bits} %row, {count}
  br i1 %more, label %guard, label %exit
guard:
  %valid = icmp slt i{bits} %row, %limit
  br i1 %valid, label %copy, label %pad
copy:
  call void @touch(ptr %input, i{bits} %row)
  br label %join
pad:
  call void @touch(ptr %input, i{bits} 0)
  br label %join
join:
  br label %latch
latch:
  %next = add i{bits} %row, 1
  br label %header
exit:
  ret void
}}
"""


def module(text=None, bits=64, **kwargs):
    return LLVMRegionBoundary(
        source(bits) if text is None else text, function_name="work", argument_types=("ptr", f"i{bits}"), **kwargs
    )


def index(m, text):
    return next(i for i in range(len(m.lines)) if m.code(i) == text)


@pytest.mark.parametrize("bits,count", [(16, 3), (32, 19), (64, 7)])
def test_actual_structural_union_and_exact_replacement(bits, count):
    m = module(source(bits, count), bits)
    nodes = m.closed_region("header", "exit")
    m.require_diamond_loop("header", "copy", nodes)
    m.require_no_liveouts(nodes)
    config = index(m, f"call void @touch(ptr %input, i{bits} 0)")
    copy = index(m, f"call void @touch(ptr %input, i{bits} %row)")
    m.require_precedes(config, [copy])
    union = m.dependency_union(nodes, include_lines=(config,))
    assert index(m, f"declare void @touch(ptr, i{bits})") in union
    assert m.definition_lines["%valid"] in union
    assert m.proof_budget(union)["complete_cfg_blocks"] == 8
    changed = m.lines.copy()
    changed[copy] = f"  call void @touch(ptr %input, i{bits} {count})\n"
    m.verify_outside("".join(changed), {copy: changed[copy]})


@pytest.mark.parametrize(
    "replace",
    [
        ("ret void", "unreachable"),
        ("br label %join", "indirectbr ptr %input, [label %join]"),
        ("br label %join", "switch i64 %row, label %join [i64 0, label %pad]"),
        ("br label %join", "invoke void @touch(ptr %input, i64 0) to label %join unwind label %pad"),
        ("br label %join", 'callbr void asm sideeffect "", ""() to label %join [label %pad]'),
        ("br i1 %more, label %guard, label %exit", "br i1 %more, label %guard, label %exit, !prof !0"),
        ("%next = add i64 %row, 1", "%row = add i64 %row, 1"),
        ("pad:", "copy:"),
        ("  ret void", "  ret void\ndead:\n  ret void"),
        ("  br label %latch", "  ret void\n  br label %latch"),
        ("@touch(ptr, i64)", "@touch(ptr, i64)\n@address = global ptr blockaddress(@work, %header)"),
    ],
)
def test_unknown_control_and_duplicate_namespace_refuse(replace):
    with pytest.raises(ValueError):
        module(source().replace(*replace))


def test_mandatory_body_bypass_and_edge_orientation_refuse():
    text = source().replace("  br label %join", "  br i1 %valid, label %join, label %latch", 1)
    m = module(text)
    with pytest.raises(ValueError, match="bypass"):
        m.require_diamond_loop("header", "copy", m.closed_region("header", "exit"))
    m = module(source().replace("label %copy, label %pad", "label %pad, label %copy"))
    with pytest.raises(ValueError, match="mandatory"):
        m.require_diamond_loop("header", "copy", m.closed_region("header", "exit"))


def test_region_side_entry_and_early_return_refuse():
    text = source().replace("  br label %header", "  br i1 %external, label %header, label %copy", 1)
    text = text.replace("entry:\n", "entry:\n  %external = icmp slt i64 %limit, 2\n")
    with pytest.raises(ValueError, match="side entry"):
        module(text).closed_region("header", "exit")
    m = module(source().replace("  br label %join", "  ret void", 1))
    with pytest.raises(ValueError, match="terminates"):
        m.closed_region("header", "exit")


@pytest.mark.parametrize(
    "unsafe",
    [
        "freeze i64 %row",
        "add nsw i64 %row, 1",
        "add nuw i64 %row, 1",
        "udiv exact i64 %row, 2",
        "select i1 %valid, i64 poison, i64 0",
        "select i1 %valid, i64 undef, i64 0",
        "ptrtoint ptr getelementptr inbounds (i8, ptr %input, i64 1) to i64",
    ],
)
def test_dependency_poison_assumptions_refuse(unsafe):
    m = module(source().replace("add i64 %row, 1", unsafe))
    with pytest.raises(ValueError, match="poison"):
        m.dependency_union(m.closed_region("header", "exit"))


def test_liveout_and_dominance_including_same_block_order():
    m = module(source().replace("exit:\n", "exit:\n  call void @touch(ptr %input, i64 %row)\n"))
    with pytest.raises(ValueError, match="outside-used"):
        m.require_no_liveouts(m.closed_region("header", "exit"))
    m = module()
    copy = index(m, "call void @touch(ptr %input, i64 %row)")
    with pytest.raises(ValueError, match="precede"):
        m.require_precedes(copy, [copy])
    with pytest.raises(ValueError, match="dominate"):
        m.require_precedes(m.definition_lines["%next"], [copy])


def test_quoted_spans_do_not_invent_uses_or_split_comments():
    text = source().replace(
        "declare void @touch",
        '; ignored %row = poison\n@message = private constant [4 x i8] c"a;b\\00"\ndeclare void @touch',
    )
    text = text.replace(
        "call void @touch(ptr %input, i64 %row)", 'call void asm sideeffect "literal %outside ; still quoted", ""()'
    )
    m = module(text)
    union = m.dependency_union(m.closed_region("header", "exit"))
    assert any("still quoted" in m.code(i) for i in union)
    assert all("%outside" not in m.references(i) for i in union)


def test_exact_ssa_tokens_remain_distinct():
    text = source().replace("  %more =", "  %a.b = add i64 %row, 1\n  %a_b = add i64 %row, 2\n  %more =")
    m = module(text)
    assert m.definition_lines["%a.b"] != m.definition_lines["%a_b"]
    with pytest.raises(ValueError, match="namespace"):
        module(source().replace("  %more =", '  %"row" = add i64 1, 2\n  %more ='))


def test_outside_control_and_frontier_changes_refuse():
    m = module()
    copy = index(m, "call void @touch(ptr %input, i64 %row)")
    candidate = source().replace("i64 %row)\n", "i64 7)\n", 1)
    with pytest.raises(ValueError, match="outside"):
        m.verify_outside(candidate.replace("ret void", "ret void ; changed"), {copy: candidate.splitlines(True)[copy]})
    line = m.definition_lines["%row"]
    candidate = m.lines.copy()
    candidate[line] = candidate[line].replace("%row =", "%foreign =")
    with pytest.raises(ValueError, match="frontier"):
        m.verify_outside("".join(candidate), {line: candidate[line]})


def test_aggregate_budget_charges_all_regions_and_complete_cfg():
    m = module()
    nodes = m.closed_region("header", "exit")
    union = m.dependency_union(nodes)
    budget = m.proof_budget(union)
    with pytest.raises(ValueError, match="aggregate"):
        m.proof_budget(
            union, changed_lines=["add i64 1, 2"] * budget["components"], max_components=budget["components"]
        )
    with pytest.raises(ValueError, match="aggregate"):
        m.proof_budget(union, max_bytes=budget["bytes"] - 1)
    with pytest.raises(ValueError, match="scan"):
        module(max_cfg_blocks=2)
    with pytest.raises(ValueError, match="scan"):
        module(max_scan_lines=2)
    with pytest.raises(ValueError, match="bound"):
        module(max_scan_bytes=2)


def test_candidate_preserves_original_stricter_scan_budget():
    text = source()
    m = module(text, max_scan_bytes=len(text.encode()))
    use = index(m, "call void @touch(ptr %input, i64 %row)")
    lines = m.lines.copy()
    lines[use] = lines[use].rstrip() + " ;" + "padding" * 100 + "\n"
    with pytest.raises(ValueError, match="bound"):
        m.verify_outside("".join(lines), {use: lines[use]})


def test_budget_refuses_before_consuming_another_proposal_fragment():
    m = module()

    def changed():
        yield "add i64 1, 2" + " " * 1000 + "0"
        raise AssertionError("consumed beyond over-budget first fragment")

    with pytest.raises(ValueError, match="aggregate"):
        m.proof_budget(set(), changed_lines=changed(), max_bytes=len(m.cfg_encoded) + 20)
    with pytest.raises(ValueError, match="line set"):
        m.proof_budget(iter([m.first]))


def test_assigned_exception_edge_cannot_hide_before_direct_branch():
    text = source().replace(
        "  %next = add i64 %row, 1",
        "  %bad = invoke i64 @hidden() to label %join unwind label %pad\n  %next = add i64 %row, 1",
    )
    with pytest.raises(ValueError, match="terminator"):
        module(text)


def test_globals_follow_actual_alias_chain_and_reject_poison():
    prefix = "@root = global i64 0\n@alias = alias i64, ptr @root\n"
    text = prefix + source().replace("ptr %input, i64 %row)", "ptr @alias, i64 %row)")
    m = module(text)
    union = m.dependency_union(m.closed_region("header", "exit"))
    assert index(m, "@root = global i64 0") in union
    with pytest.raises(ValueError, match="poison"):
        m = module(text.replace("@root = global i64 0", "@root = global i64 poison"))
        m.dependency_union(m.closed_region("header", "exit"))


def test_actual_llvm_assembler_and_disassembler(tmp_path):
    install = Path(os.environ.get("MERLIN_MLIR_INSTALL", "")) / "bin"
    assembler = shutil.which("llvm-as") or str(install / "llvm-as")
    disassembler = shutil.which("llvm-dis") or str(install / "llvm-dis")
    if not Path(assembler).is_file() or not Path(disassembler).is_file():
        pytest.skip("explicit LLVM assembler/disassembler unavailable")
    original = tmp_path / "original.ll"
    bitcode = tmp_path / "module.bc"
    parsed = tmp_path / "parsed.ll"
    original.write_text(source())
    subprocess.run([assembler, str(original), "-o", str(bitcode)], check=True, capture_output=True)
    subprocess.run([disassembler, str(bitcode), "-o", str(parsed)], check=True, capture_output=True)
    m = module(parsed.read_text())
    m.require_diamond_loop("header", "copy", m.closed_region("header", "exit"))
    m.require_no_liveouts(m.closed_region("header", "exit"))
