import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_source_region import sibling_source
from xdsl.dialects import builtin, llvm

from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.source_program import SourceEmission, emit_source_program, prepare_source_program
from merlin.xdsl_dialects._common import text


def prepared():
    return prepare_source_program(text(sibling_source(), generic=True), target="neutral",
        effects=BorrowedBufferEffects("neutral-test-only-not-hardware", True, True, True))


def test_original_ssa_mixed_outputs_and_complete_abi_are_preserved():
    plan = prepared()
    assert [row[2:] for row in plan.abi] == [("i8", "read")] * 3 + [("f32", "write"), ("i32", "write")]
    assert "arith.sitofp" in plan.host and "arith.muli" not in plan.host
    assert len(plan.source_members) == 2
    assert "source_device(" in plan.core_transport
    assert "_mlir_ciface_forward(&d0,&d1,&d2,&d3,&d4);" in plan.entry_transport
    assert "source_shared_core_borrowed" in plan.host
    assert plan.revalidate().core == plan.core


def test_actual_emitter_callbacks_consume_returned_components_and_writers():
    from merlin.llvmlower.host_builder import FnBuilder

    plan = prepared()
    seen = []
    def device(module, *, symbol, abi, source_members):
        assert source_members == plan.source_members
        # Neutral callback output is actual verified LLVM, not an assertion of
        # source arithmetic or a runnable target implementation.
        builder = FnBuilder([llvm.LLVMPointerType()] * len(abi))
        builder.add(llvm.CallOp("neutral_issue"))
        target = builtin.ModuleOp([llvm.FuncOp("neutral_issue", llvm.LLVMFunctionType([]),
            linkage=llvm.LinkageAttr("external")),
            llvm.FuncOp(symbol, llvm.LLVMFunctionType([llvm.LLVMPointerType()] * len(abi)), body=builder.finish(),
                        linkage=llvm.LinkageAttr("external"))])
        target.verify()
        seen.append("device")
        return SourceEmission(hashlib.sha256(text(module, generic=True).encode()).hexdigest(),
            text(target, generic=True).encode(), "llvm_dialect")
    def host(module, *, writers):
        assert writers == (plan.prepared.writer,)
        assert "arith.sitofp" in text(module, generic=True)
        seen.append("host")
        host = builtin.ModuleOp([llvm.FuncOp("neutral_host_owner", llvm.LLVMFunctionType([]),
                                           body=FnBuilder([]).finish())])
        host.verify()
        return SourceEmission(hashlib.sha256(text(module, generic=True).encode()).hexdigest(),
            text(host, generic=True).encode(), "llvm_dialect")
    result = emit_source_program(plan, emit_device=device, emit_host=host)
    assert seen == ["device", "host"]
    assert not result.receipt["physical_effects_qualified"]
    assert result.device.data.startswith(b'"builtin.module"')


def test_stale_component_wrong_emission_and_missing_effects_refuse():
    plan = prepared()
    with pytest.raises(ValueError, match="changed"):
        replace(plan, core=plan.core + "\n").revalidate()
    with pytest.raises(ValueError, match="does not match"):
        emit_source_program(plan, emit_device=lambda *a, **kw: SourceEmission("0" * 64, b"x", "object"),
                            emit_host=lambda *a, **kw: pytest.fail("must refuse before host emission"))
    with pytest.raises(ValueError, match="effects"):
        prepare_source_program(plan.source, target="neutral", effects=None)


def test_unknown_effect_and_nonzero_seed_keep_original_refusal():
    for source in (sibling_source(unknown_effect=True), sibling_source(seed=1)):
        with pytest.raises(ValueError):
            prepare_source_program(text(source, generic=True), target="neutral",
                effects=BorrowedBufferEffects("test", True, True, True))


def test_selected_stock_host_tools_consume_original_ssa_and_link_transports(tmp_path):
    """An opt-in tool test, not a neutral stub numerical qualification."""
    from merlin.llvmlower.host_builder import FnBuilder

    selected = os.environ.get("MERLIN_TEST_STOCK_LLVM")
    sysroot = os.environ.get("MERLIN_TEST_RISCV_SYSROOT")
    linker = os.environ.get("MERLIN_TEST_RISCV_LD")
    if not all((selected, sysroot, linker)):
        pytest.skip("explicit stock LLVM, RV64 sysroot and linker selections required")
    tools = Path(selected)
    plan = prepared()
    passes = ("canonicalize,cse,"
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
        "buffer-results-to-out-params{modify-public-functions hoist-static-allocs},"
        "func.func(convert-linalg-to-loops),convert-scf-to-cf,expand-strided-metadata,lower-affine,"
        "convert-math-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,"
        "convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts")
    def run(*args):
        result = subprocess.run([str(arg) for arg in args], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result
    def device(module, *, symbol, abi, source_members):
        args = [llvm.LLVMPointerType()] * len(abi)
        product = builtin.ModuleOp([llvm.FuncOp(symbol, llvm.LLVMFunctionType(args), body=FnBuilder(args).finish(),
                                              linkage=llvm.LinkageAttr("external"))])
        product.verify()
        return SourceEmission(hashlib.sha256(text(module, generic=True).encode()).hexdigest(),
                              text(product, generic=True).encode(), "llvm_dialect")
    def host(module, *, writers):
        assert writers == (plan.prepared.writer,)
        source = tmp_path / "host.mlir"
        source.write_text(text(module, generic=True))
        lowered = tmp_path / "host.llvm.mlir"
        run(tools / "mlir-opt", source, "--pass-pipeline=builtin.module(" + passes + ")", "-o", lowered)
        product = run(tools / "mlir-translate", "--mlir-to-llvmir", lowered).stdout
        assert "sitofp" in product and "source_shared_core_borrowed" in product
        return SourceEmission(hashlib.sha256(source.read_bytes()).hexdigest(), product.encode(), "llvm_ir")
    product = emit_source_program(plan, emit_device=device, emit_host=host)
    (tmp_path / "host.ll").write_bytes(product.host.data)
    (tmp_path / "device.mlir").write_bytes(product.device.data)
    (tmp_path / "device.ll").write_text(run(tools / "mlir-translate", "--mlir-to-llvmir",
                                           tmp_path / "device.mlir").stdout)
    (tmp_path / "entry.c").write_text(product.entry_transport)
    (tmp_path / "core.c").write_text(product.core_transport)
    objects = []
    for filename in ("host.ll", "device.ll", "entry.c", "core.c"):
        obj = tmp_path / (filename + ".o")
        run(tools / "clang", "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
            "--sysroot=" + sysroot, "-isystem", Path(sysroot) / "include",
            "-O1", "-c", tmp_path / filename, "-o", obj)
        objects.append(obj)
    linked = tmp_path / "components.o"
    run(linker, "-r", *objects, "-o", linked)
    undefined = run(tools / "llvm-nm", "-u", linked).stdout
    assert "source_device" not in undefined and "source_shared_core_borrowed" not in undefined
    assert "forward" not in undefined
