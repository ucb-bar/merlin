"""Compile-only target facts: no ctypes, target execution, or model programs."""
from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from merlin.runtime.backends.base import get_backend


@pytest.fixture
def adapter():
    backend = get_backend("gemmini")
    return importlib.import_module(backend.__package__ + ".gemmini_compact_abi")


def documents():
    cb = {"kernel_abi": {"kind": "whole_program", "args": [
        {"tensor": "x", "access": "read"}, {"tensor": "y", "access": "write"}]},
        "tensors": {"x": {"dtype": "i8"}, "y": {"dtype": "f32"}}}
    contract = {"schema": "compact_pointer_entry_v1", "bases": [{}, {}], "bindings": [
        {"argument_index": 0, "base_index": 0}, {"argument_index": 1, "base_index": 1}]}
    return cb, contract


def layout_text(layout):
    return f'target datalayout = "{layout}"\ntarget triple = "test-triple"\n'


def test_explicit_pointer_index_is_not_pointer_storage_size(adapter):
    info = adapter._layout(layout_text("e-p:64:64:64:32"))
    assert info["pointer_bits"] == 64 and info["pointer_index_bits"] == 32
    assert adapter._layout(layout_text("E-p0:32:32"))["pointer_index_bits"] == 32


@pytest.mark.parametrize("layout", ["e-i64:64", "e-p:64:64-p0:64:64", "e-p:64:64:64:128",
                                      "e-p:64:no", "e-p:64:64-ni:0", "e-p:64:64-A1"])
def test_ambiguous_or_unsupported_layout_refuses(adapter, layout):
    with pytest.raises(ValueError):
        adapter._layout(layout_text(layout))


@pytest.mark.parametrize("field,value", [("pointer_bits", 32), ("pointer_alignment", 4),
    ("byte_order", 2), ("char_bits", 16), ("element_bits_0", 16), ("alignment_0", 3)])
def test_actual_fact_disagreement_refuses(adapter, field, value):
    facts = {"pointer_bits": 64, "pointer_alignment": 8, "char_bits": 8,
             "byte_order": 1, "element_bits_0": 8, "alignment_0": 16}
    facts[field] = value
    with pytest.raises(ValueError):
        adapter._agree(adapter._layout(layout_text("e-p:64:64")), facts, [8])


def test_symbol_reader_rejects_missing_sizes_and_duplicates(adapter):
    assert adapter._symbol_sizes("0000 0064 B merlin_abi_pointer_bits\n") == {"pointer_bits": 64}
    with pytest.raises(ValueError):
        adapter._symbol_sizes("0000 B merlin_abi_pointer_bits\n")
    with pytest.raises(ValueError):
        adapter._symbol_sizes("0 64 B merlin_abi_pointer_bits\n0 64 B merlin_abi_pointer_bits\n")


@pytest.fixture
def actual_tools():
    from merlin.llvmlower.toolchain import clang
    registered = get_backend("gemmini")
    backend = importlib.import_module(registered.__package__ + ".gemmini")
    recipe = backend.harness_build_recipe()
    if not Path(recipe.compiler).is_file() or not Path(clang()).is_file():
        pytest.skip("configured target compilers unavailable")
    return backend, recipe


def test_actual_target_compiles_facts_without_executing(adapter, actual_tools, tmp_path):
    result = adapter.derive_compact_target_abi(*documents(), workdir=tmp_path)
    evidence = result.to_evidence()
    assert evidence["target_executed"] is False
    assert evidence["linker_allocations_verified"] is False
    assert evidence["emitted_address_space_verified"] is False
    assert evidence["harness_facts"]["pointer_bits"] == evidence["kernel_layout"]["pointer_bits"]
    assert result.target_abi.pointer_index_bits == (evidence["kernel_layout"]["pointer_index_bits"],) * 2
    assert all(adapter._file_sha(path) == digest for path, digest in evidence["source_pins"].items())
    assert all(command["returncode"] == 0 for command in evidence["commands"])
    assert not any(Path(command["argv"][0]).name == "facts.o" for command in evidence["commands"])


def test_actual_earlier_header_is_selected_and_wrong_element_width_refuses(
        adapter, actual_tools, tmp_path, monkeypatch):
    backend, recipe = actual_tools
    include = tmp_path / "shadow" / "include"
    include.mkdir(parents=True)
    # Deliberately different authoritative C facts in an earlier include root.
    header = include / "gemmini_params.h"
    (include / "gemmini_testutils.h").write_text('#include "gemmini_params.h"\n')
    header.write_text("#include <stdint.h>\ntypedef int16_t elem_t;\n"
                      "#define row_align(n) __attribute__((aligned(32)))\n"
                      "#define row_align_acc(n) __attribute__((aligned(64)))\n")
    monkeypatch.setattr(backend, "harness_build_recipe", lambda: replace(
        recipe, include_roots=(include.parent, *recipe.include_roots)))
    with pytest.raises(ValueError, match="container/declared dtype width"):
        adapter.derive_compact_target_abi(*documents(), workdir=tmp_path / "proof")


def test_changed_selected_header_pin_refuses(adapter, actual_tools, tmp_path, monkeypatch):
    backend, recipe = actual_tools
    include = tmp_path / "shadow" / "include"
    include.mkdir(parents=True)
    header = include / "gemmini_params.h"
    (include / "gemmini_testutils.h").write_text('#include "gemmini_params.h"\n')
    header.write_text("#include <stdint.h>\ntypedef int8_t elem_t;\n"
                      "#define row_align(n) __attribute__((aligned(32)))\n"
                      "#define row_align_acc(n) __attribute__((aligned(64)))\n")
    monkeypatch.setattr(backend, "harness_build_recipe", lambda: replace(
        recipe, include_roots=(include.parent, *recipe.include_roots)))
    original = adapter._symbol_sizes

    def mutate_after_real_object_compile(text):
        result = original(text)
        header.write_text(header.read_text() + "\n/* changed after actual compile */\n")
        return result

    monkeypatch.setattr(adapter, "_symbol_sizes", mutate_after_real_object_compile)
    with pytest.raises(ValueError, match="pins changed"):
        adapter.derive_compact_target_abi(*documents(), workdir=tmp_path / "proof")


@pytest.mark.parametrize("timeout", [0, 61, True, float("nan")])
def test_deadline_validated_before_compiler(adapter, tmp_path, timeout):
    with pytest.raises(ValueError, match="deadline"):
        adapter.derive_compact_target_abi(*documents(), workdir=tmp_path, timeout_s=timeout)
