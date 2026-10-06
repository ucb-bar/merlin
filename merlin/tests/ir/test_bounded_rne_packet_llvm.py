"""Pure multi-return proof, CPU compilation and refusal of memory/effects."""

import runpy
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.bounded_rne_packet_llvm import rewrite_packet_helpers
from merlin.llvmlower.late_quant_rne import _tokens
from merlin.llvmlower.toolchain import clang


def source(width):
    scalar = runpy.run_path(str(Path(__file__).with_name("test_late_quant_rne.py")))["SOURCE"]
    body = scalar[scalar.index("{") + 1 : scalar.index("  ret ")]
    pieces = []
    for lane in range(width):
        part = body
        for token in reversed(_tokens(body)):
            if token.text.startswith("%"):
                part = part[: token.start] + token.text + f".lane{lane}" + part[token.end :]
        pieces.append(part)
    aggregate = "{ " + ", ".join(["i8"] * width) + " }"
    previous = "poison"
    for lane in range(width):
        name = f"%ret{lane}"
        pieces.append(f"  {name} = insertvalue {aggregate} {previous}, i8 %v17.lane{lane}, {lane}\n")
        previous = name
    return (
        f"define {aggregate} @ordinary_name("
        + ", ".join(f"float %x.lane{i}" for i in range(width))
        + ") {\n"
        + "".join(pieces)
        + f"  ret {aggregate} {previous}\n"
        + "}\ndeclare float @llvm.maximum.f32(float,float)\ndeclare float @llvm.minimum.f32(float,float)\n"
    )


@pytest.mark.parametrize("width", [2, 3, 4])
def test_proved_packet_compiles_and_default_is_unchanged(tmp_path, width):
    original = source(width)
    assert rewrite_packet_helpers(original)[0] == original
    result, proof = rewrite_packet_helpers(original, host_isa="rv64gc")
    assert len(proof["routes"]) == 1 and proof["routes"][0]["lanes"] == width
    path = tmp_path / "packet.ll"
    path.write_text(result)
    subprocess.run(
        [
            str(clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-O2",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "packet.o"),
        ],
        check=True,
        capture_output=True,
    )
    assert result.count(" asm ") == 1


@pytest.mark.parametrize("change", ["memory", "call", "strict", "half", "overflow", "missing_lane"])
def test_unsupported_context_or_numeric_graph_is_unchanged(change):
    original = source(3)
    if change == "memory":
        original = original.replace("  %v1.lane0", "  %memory = load i8, ptr null\n  %v1.lane0", 1)
    elif change == "call":
        original = original.replace("  %v1.lane0", "  call void @effect()\n  %v1.lane0", 1) + "declare void @effect()\n"
    elif change == "strict":
        original = original.replace(") {", ") strictfp {", 1)
    elif change == "half":
        original = original.replace("5.000000e-01", "4.000000e-01", 1)
    elif change == "overflow":
        original = original.replace("add i8", "add nsw i8", 1)
    else:
        original = original.replace("i8 %v17.lane1, 1", "i8 %v17.lane1, 0")
    result, proof = rewrite_packet_helpers(original, host_isa="rv64gc")
    assert result == original and not proof["routes"]


def test_existing_local_names_are_not_reused():
    original = source(2).replace("  %v1.lane0", "  %merlin.packet.0.asm = add i8 0, 0\n  %v1.lane0", 1)
    result, proof = rewrite_packet_helpers(original, host_isa="rv64gc")
    assert proof["routes"] and "%merlin.packet.1.asm = call" in result


@pytest.mark.parametrize("width", [5, 6, 7, 8])
def test_wide_packet_requires_explicit_budget_and_compiles(tmp_path, width):
    original = source(width)
    assert rewrite_packet_helpers(original, host_isa="rv64gc")[0] == original
    result, proof = rewrite_packet_helpers(original, host_isa="rv64gc", max_lanes=8)
    assert len(proof["routes"]) == 1 and proof["routes"][0]["lanes"] == width
    path = tmp_path / "packet.ll"
    path.write_text(result)
    subprocess.run(
        [
            str(clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-O2",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "packet.o"),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("budget", [None, True, 1, 9, "8"])
def test_unknown_or_unbounded_lane_budget_refuses(budget):
    with pytest.raises(ValueError, match="lane budget"):
        rewrite_packet_helpers(source(2), host_isa="rv64gc", max_lanes=budget)


def test_explicit_budget_does_not_grant_new_numeric_or_effect_permissions():
    original = source(8).replace(") {", ") strictfp {", 1)
    result, proof = rewrite_packet_helpers(original, host_isa="rv64gc", max_lanes=8)
    assert result == original and not proof["routes"]
    assert rewrite_packet_helpers(source(9), host_isa="rv64gc", max_lanes=8)[0] == source(9)


def test_explicit_wide_budget_preserves_the_original_four_lane_bytes_and_report():
    original = source(4)
    assert rewrite_packet_helpers(original, host_isa="rv64gc") == rewrite_packet_helpers(
        original, host_isa="rv64gc", max_lanes=8
    )
