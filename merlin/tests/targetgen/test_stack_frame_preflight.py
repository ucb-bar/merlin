"""A target-bound LLVM object is admitted only with fresh, bounded static-stack evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract import compile as compiler


def _recipe(*, budget: int = 4096, policy: bool = True) -> HarnessBuildRecipe:
    stack = KernelStackFramePolicy("fixture_entry", budget) if policy else None
    return HarnessBuildRecipe(
        compiler=Path("/fixture/cc"), include_roots=(), support_sources=(),
        link_script=Path("/fixture/link.ld"), load_address=0,
        cflags=("-march=rv64gc",), error_cls=ValueError, kernel_stack_frame=stack)


def _object_build(tmp_path, monkeypatch, report: str | None, *, budget: int = 4096,
                  stale: str | None = None):
    from merlin.llvmlower import codegen, pipeline
    from merlin.runtime.backends import base

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda text, *, workdir: "; fixture llvm\n")
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: _recipe(budget=budget))
    calls = []

    def compile_ll(source, output, target, *, extra_flags):
        calls.append((Path(source), Path(output), target, extra_flags))
        Path(output).write_bytes(b"fixture object")
        if report is not None:
            Path(output).with_suffix(".su").write_text(report, encoding="utf-8")
        return output

    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    if stale is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "kernel.su").write_text(stale, encoding="utf-8")
    result = compiler.llvm_mlir_to_object("module {}", tmp_path, target="fixture")
    return result, calls


def test_target_bound_object_records_hash_bound_static_frame(tmp_path, monkeypatch):
    obj, calls = _object_build(
        tmp_path, monkeypatch,
        f"{tmp_path}/kernel.ll:helper\t32\tstatic\n"
        f"{tmp_path}/kernel.ll:fixture_entry\t384\tstatic\n")

    assert obj == tmp_path / "kernel.o"
    assert calls == [(tmp_path / "kernel.ll", tmp_path / "kernel.o", "riscv",
                      ("-march=rv64gc", "-fstack-usage"))]
    canonical_report = (
        "kernel.ll:helper\t32\tstatic\nkernel.ll:fixture_entry\t384\tstatic\n")
    assert (tmp_path / "kernel.su").read_text() == canonical_report
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt == {
        "schema": "merlin_kernel_stack_frame_preflight_v1",
        "status": "passed",
        "entry_symbol": "fixture_entry",
        "max_static_bytes": 4096,
        "frame_bytes": 384,
        "headroom_bytes": 3712,
        "report_rows": 2,
        "llvm_ir_sha256": hashlib.sha256(b"; fixture llvm\n").hexdigest(),
        "object_sha256": hashlib.sha256(b"fixture object").hexdigest(),
        "stack_usage_report_sha256": hashlib.sha256(canonical_report.encode()).hexdigest(),
        "diagnostic": None,
        # A frame that fit exactly as emitted. The key is always present so that "nothing was done
        # to make this fit" and "the field was not recorded" cannot read the same.
        "repair": None,
    }


@pytest.mark.parametrize(("report", "message"), [
    (None, "no regular stack-usage report"),
    ("not a report\n", "three tab-separated fields"),
    ("kernel.ll:fixture_entry\t64\tdynamic\n", "dynamic allocation"),
    ("kernel.ll:other_entry\t64\tstatic\n", "0 rows for entrypoint"),
    ("other.ll:fixture_entry\t64\tstatic\n", "names source"),
])
def test_target_bound_object_fails_closed_on_unproven_report(
        tmp_path, monkeypatch, report, message):
    with pytest.raises(ValueError, match=message):
        _object_build(tmp_path, monkeypatch, report)
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt["status"] == "rejected"
    assert message.replace("\\", "") in receipt["diagnostic"]
    assert receipt["llvm_ir_sha256"] and receipt["object_sha256"]


def test_stale_sidecar_is_removed_before_compile(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no regular stack-usage report"):
        _object_build(
            tmp_path, monkeypatch, None,
            stale="kernel.ll:fixture_entry\t64\tstatic\n")
    assert not (tmp_path / "kernel.su").exists()


def test_oversized_frame_is_rejected_with_numeric_receipt(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exceeding the target-declared 4096-byte budget by 1"):
        _object_build(
            tmp_path, monkeypatch,
            "kernel.ll:fixture_entry\t4097\tstatic\n")
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt["status"] == "rejected"
    assert receipt["frame_bytes"] == 4097
    assert receipt["headroom_bytes"] == -1


def test_target_bound_recipe_must_declare_stack_policy(tmp_path, monkeypatch):
    from merlin.llvmlower import codegen, pipeline
    from merlin.runtime.backends import base

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda text, *, workdir: "; llvm\n")
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: _recipe(policy=False))
    monkeypatch.setattr(codegen, "compile_ll", lambda *args, **kwargs: pytest.fail("compiled"))
    with pytest.raises(ValueError, match="no kernel stack-frame policy"):
        compiler.llvm_mlir_to_object("module {}", tmp_path, target="fixture")


@pytest.mark.parametrize(("entry", "budget"), [("", 1), ("entry", 0), ("entry", True)])
def test_stack_policy_rejects_ambiguous_declarations(entry, budget):
    with pytest.raises(ValueError):
        KernelStackFramePolicy(entry, budget)


# ---------------------------------------------------------------------------------------------
# Repairing an over-budget frame by moving its static temporaries into one arena
# ---------------------------------------------------------------------------------------------
#
# The host lane hoists one `alloca` per intermediate into the entry frame with no reuse, so the
# frame scales with the model: 816 bytes on ResNet-50 and 99,897,984 on SmolVLA's flow_denoise
# against a 65,536-byte budget. The repair runs ONLY after the emitted frame has been measured over
# budget, which is what keeps an already-fitting build byte-identical -- and the one bundle known to
# have run correctly on hardware is an already-fitting build, so that property is load-bearing.

_OVERSIZED_LLVM = """; fixture
define void @fixture_entry(ptr %0) {
  %1 = alloca i8, i64 8192, align 64
  %2 = alloca i8, i64 8192, align 64
  ret void
}
"""


def _repairable_build(tmp_path, monkeypatch, *, frames, budget=4096, llvm=_OVERSIZED_LLVM):
    """Drive the object build with a scripted frame size per compiler invocation.

    ``frames`` is consumed in order, so the first entry is the frame of the EMITTED program and the
    second (if any) is the frame after the arena rewrite. That ordering is the thing under test:
    the repair must be measured, never assumed.
    """
    from merlin.llvmlower import codegen, pipeline
    from merlin.runtime.backends import base

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda text, *, workdir: llvm)
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: _recipe(budget=budget))
    sizes = list(frames)
    calls = []

    def compile_ll(source, output, target, *, extra_flags):
        source, output = Path(source), Path(output)
        calls.append(source.name)
        output.write_bytes(b"fixture object " + source.name.encode())
        frame = sizes.pop(0)
        output.with_suffix(".su").write_text(
            f"{source}:fixture_entry\t{frame}\tstatic\n", encoding="utf-8")
        return output

    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    result = compiler.llvm_mlir_to_object("module {}", tmp_path, target="fixture")
    return result, calls


def test_an_oversized_frame_is_REPAIRED_and_the_receipt_says_so(tmp_path, monkeypatch):
    obj, calls = _repairable_build(tmp_path, monkeypatch, frames=[16512, 384])
    assert calls == ["kernel.ll", "kernel.arena.ll"], "the emitted program is measured FIRST"
    assert obj == tmp_path / "kernel.arena.o"
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["frame_bytes"] == 384 and receipt["headroom_bytes"] == 4096 - 384
    repair = receipt["repair"]
    assert repair["transform"] == "stack_arena_bind"
    assert repair["frame_bytes_before"] == 16512
    assert "exceeding the target-declared 4096-byte budget" in repair["diagnostic_before"]
    assert repair["n_bound"] == 2 and repair["n_refused"] == 0
    assert repair["moved_bytes"] == 16384
    assert repair["arena_bytes"] >= 16384 and repair["shares_bytes"] is False
    # The receipt must be bound to the bytes that were MEASURED, not to the emitted ones.
    assert repair["emitted_llvm_ir_sha256"] == hashlib.sha256(_OVERSIZED_LLVM.encode()).hexdigest()
    assert receipt["llvm_ir_sha256"] != repair["emitted_llvm_ir_sha256"]


def test_a_frame_that_ALREADY_FITS_is_never_rewritten(tmp_path, monkeypatch):
    """The regression that would invalidate the hardware-validated bundle."""
    obj, calls = _repairable_build(tmp_path, monkeypatch, frames=[384])
    assert calls == ["kernel.ll"], "no second compile: nothing was rewritten"
    assert obj == tmp_path / "kernel.o"
    assert not (tmp_path / "kernel.arena.ll").exists()
    assert not (tmp_path / "kernel.arena.o").exists()
    assert json.loads((tmp_path / "kernel.stack_frame.json").read_text())["repair"] is None


def test_a_repair_that_STILL_does_not_fit_raises_the_ORIGINAL_refusal(tmp_path, monkeypatch):
    """A repaired variant nobody asked for must not become the subject of the diagnostic."""
    with pytest.raises(ValueError, match="budget by 12416"):
        _repairable_build(tmp_path, monkeypatch, frames=[16512, 9000])
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt["status"] == "rejected"
    assert receipt["frame_bytes"] == 16512, "the EMITTED frame, not the repaired one"
    assert receipt["repair"] is None


def test_an_oversized_frame_with_NOTHING_BINDABLE_is_rejected_unchanged(tmp_path, monkeypatch):
    """No alloca to move means no repair is available, and the refusal must stand."""
    with pytest.raises(ValueError, match="budget by 12416"):
        _repairable_build(tmp_path, monkeypatch, frames=[16512],
                          llvm="; fixture\ndefine void @fixture_entry(ptr %0) {\n  ret void\n}\n")
    receipt = json.loads((tmp_path / "kernel.stack_frame.json").read_text())
    assert receipt["status"] == "rejected" and receipt["repair"] is None
    assert not (tmp_path / "kernel.arena.ll").exists()


def test_a_module_the_arena_pass_cannot_READ_is_rejected_unchanged(tmp_path, monkeypatch):
    """A StackArenaError must leave the original refusal in place, not surface as a new error."""
    with pytest.raises(ValueError, match="budget by 12416"):
        _repairable_build(tmp_path, monkeypatch, frames=[16512], llvm="; no function at all\n")
    assert json.loads((tmp_path / "kernel.stack_frame.json").read_text())["repair"] is None
