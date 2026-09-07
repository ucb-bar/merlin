"""Optional compact caller: actual tiny target ELF, never target execution."""
import copy
import importlib
import json
from pathlib import Path

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.targetgen.contract import compile as compiler


@pytest.fixture
def edge():
    b = get_backend("gemmini")
    return importlib.import_module(b.__package__ + ".gemmini_compact_caller")


@pytest.fixture
def actual_tools():
    from merlin.llvmlower.toolchain import clang
    recipe = get_backend("gemmini").harness_build_recipe()
    if not Path(recipe.compiler).is_file() or not Path(clang()).is_file():
        pytest.skip("configured target compilers unavailable")


def source_and_inputs(pointer_bits=64):
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser
    from merlin.llvmlower.compact_abi import compact_pointer_entry, BaseBuffer, PointerBinding
    from merlin.xdsl_dialects._common import text

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(ctx, '''builtin.module {
      llvm.func @original(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr) {
        %x = llvm.load %a : !llvm.ptr -> i32
        %y = llvm.load %b : !llvm.ptr -> i32
        %z = llvm.add %x, %y : i32
        llvm.store %z, %out : i32, !llvm.ptr
        llvm.return
      }
    }''').parse_module()
    original = module.body.block.first_op
    compact, contract = compact_pointer_entry(original, symbol="compact_sum",
        bases=[BaseBuffer("read", 68, pointer_bits), BaseBuffer("write", 4, pointer_bits)],
        bindings=[PointerBinding(0, 0, 0, 4), PointerBinding(1, 0, 64, 4), PointerBinding(2, 1, 0, 4)],
        binding_provenance="tiny trusted test source; not model qualification")
    module.body.block.insert_op_before(compact, original)
    module.body.block.erase_op(original)
    module.verify()
    encoding = GroupedAxesStorage((), "i32", ((),), (1,), (1,), 1).to_dict()
    cb = {"abi_version": "0.1", "target": "gemmini", "commands": [],
          "kernel_abi": {"kind": "whole_program", "args": [
              {"tensor": "a", "access": "read"}, {"tensor": "b", "access": "read"},
              {"tensor": "out", "access": "write"}], "outputs": ["out"]},
          "tensors": {name: {"shape": [1], "dtype": "i32", "role": "output" if name == "out" else "input"}
                      for name in ("a", "b", "out")},
          "params": {"storage_encodings": {name: copy.deepcopy(encoding) for name in ("a", "b", "out")}}}
    payloads = {"a": (2).to_bytes(4, "little"), "b": (3).to_bytes(4, "little")}
    return cb, contract, payloads, text(module)


@pytest.fixture
def prepared(edge, actual_tools, tmp_path):
    cb, contract, payloads, source = source_and_inputs()
    p = edge.prepare_compact_caller(cb, contract, payloads, lowered_mlir_text=source, workdir=tmp_path)
    return cb, contract, payloads, source, p


def test_warm_one_measured_one_and_typed_readback_even_cold(edge, prepared, monkeypatch):
    cb, _, _, _, p = prepared
    monkeypatch.setenv("MERLIN_CACHE_STATE", "cold")
    source = edge.render_compact_caller(cb, p)
    call = "compact_sum((void*)merlin_compact_base_0, (void*)merlin_compact_base_1);"
    assert source.count(call) == 2
    assert source.index(call) < source.index("__builtin_memset") < source.index("uint64_t c0")
    assert source.index("uint64_t c0") < source.rindex(call) < source.index("uint64_t c1")
    assert "__builtin_memcpy(&value, merlin_compact_base_1" in source
    assert "int32_t value;" in source
    assert "original(" not in source


def test_restore_readwrite_initial_bytes_outside_timer(edge, actual_tools, tmp_path, monkeypatch):
    cb, contract, payloads, source = source_and_inputs()
    cb["kernel_abi"]["args"][-1]["access"] = "readwrite"
    payloads["out"] = (7).to_bytes(4, "little")
    p = edge.prepare_compact_caller(cb, contract, payloads, lowered_mlir_text=source, workdir=tmp_path)
    generated = edge.render_compact_caller(cb, p)
    assert "merlin_compact_initial_1[4] = {7,0,0,0}" in generated
    assert generated.index("__builtin_memcpy(merlin_compact_base_1") < generated.index("uint64_t c0")


def test_prepared_binding_is_not_transferable_to_other_command_buffer(edge, prepared):
    cb, _, _, _, p = prepared
    changed = copy.deepcopy(cb)
    changed["commands"] = [{"opcode": "changed"}]
    with pytest.raises(ValueError, match="another command buffer"):
        edge.render_compact_caller(changed, p)
    with pytest.raises(ValueError, match="host-prepared"):
        edge.render_compact_caller(cb, {"approved": True})


@pytest.mark.parametrize("change", ["count", "symbol", "address_space"])
def test_emitted_signature_mismatch_refuses(edge, change):
    _, contract, _, source = source_and_inputs()
    if change == "count": contract["compact_argument_count"] = 3
    elif change == "symbol": contract["compact_symbol"] = "another_symbol"
    else: source = source.replace("!llvm.ptr", "!llvm.ptr<1>")
    with pytest.raises(ValueError):
        edge._signature(source, contract)


def test_actual_complete_compact_elf_bypasses_cache(actual_tools, tmp_path, monkeypatch):
    from merlin.targetgen import build_cache
    cb, contract, payloads, source = source_and_inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("compact route must not read or publish ELF cache")

    monkeypatch.setattr(build_cache, "reuse", forbidden)
    monkeypatch.setattr(build_cache, "store", forbidden)
    elf = compiler.compile_lowered_to_elf(cb, source, tmp_path, target="gemmini",
        compact_contract=contract, logical_payloads=payloads)
    receipt = json.loads((tmp_path / "compact_caller_link.json").read_text())
    assert elf.is_file() and receipt["status"] == "static_build_validated"
    assert receipt["linked_allocations_verified"] and not receipt["target_executed"]
    assert receipt["warm_invocations_emitted"] == receipt["measured_invocations_emitted"] == 1
    assert receipt["signature"]["argument_count"] == 2
    assert receipt["binding"]["original_argument_tensors"] == ["a", "b", "out"]
    assert receipt["numerical_equivalence"] == "UNPROVEN"
    assert receipt["address_check"]["alignment_and_nonoverlap_checked"]


@pytest.mark.parametrize("kwargs", [{"compact_contract": {}}, {"logical_payloads": {}},
    {"compact_contract": {}, "logical_payloads": {}, "inputs": {"x": [1]}}])
def test_incomplete_or_ambiguous_opt_in_refuses_before_compile(monkeypatch, tmp_path, kwargs):
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **k: pytest.fail("must refuse before compilation"))
    with pytest.raises(ValueError):
        compiler.compile_lowered_to_elf({}, "", tmp_path, target="gemmini", **kwargs)
