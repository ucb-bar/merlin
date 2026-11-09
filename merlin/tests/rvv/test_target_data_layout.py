"""A host loop is lowered under the cross target's own data layout, so its accesses are aligned."""

from __future__ import annotations

import pytest

from merlin.llvmlower import target_data_layout as TDL


def test_the_layout_line_is_read_structurally():
    ir = 'source_filename = "-"\ntarget datalayout = "e-m:e-p:64:64-i64:64"\ntarget triple = "x"\n'
    assert TDL.parse(ir) == "e-m:e-p:64:64-i64:64"
    assert TDL.parse("no layout here\n") is None


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("e-m:e-p:32:32-i64:64-n32-S128", 32),
        ("e-m:e-p:64:64-i64:64-n32:64-S128", 64),
    ],
)
def test_default_pointer_index_width_is_derived_from_reported_layout(layout, expected):
    assert TDL.default_index_bits(layout) == expected


@pytest.mark.parametrize(
    "layout",
    ["", "e-i64:64", "e-p:64", "e-p:64:64-p0:32:32", "e-p:64:64:64:nope", "e-p0:64:64:64:32"],
)
def test_missing_or_ambiguous_default_pointer_width_refuses(layout):
    with pytest.raises(ValueError):
        TDL.default_index_bits(layout)


def test_selected_index_observation_requires_complete_nonempty_compiler_flags():
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral compiler observation",
    }
    assert TDL.selected_index_bits(selected) == 64
    for key, value in (("cross_flags", []), ("cross_flags", [""]), ("index_bits", True), ("data_layout", "")):
        assert TDL.selected_index_bits({**selected, key: value}) is None
    assert TDL.selected_index_bits({**selected, "extra": 0}) is None


def test_absent_selected_compiler_cannot_supply_a_width(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        TDL.observe_index_width(tmp_path / "no-clang", ["--target=unknown"])


def test_all_index_converters_share_explicit_width():
    from merlin.llvmlower import pipeline

    for bits in (32, 64):
        selected = pipeline._bind_index_width(pipeline._upstream_pipeline(), bits)
        for name in (
            "convert-index-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
        ):
            assert selected.count(f"{name}{{index-bitwidth={bits}}}") == 1
    with pytest.raises(ValueError):
        pipeline._bind_index_width("builtin.module(convert-index-to-llvm)", 32)


@pytest.mark.parametrize(
    "name",
    [
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
    ],
)
def test_preconfigured_duplicate_cannot_hide_beside_a_bare_converter(name):
    from merlin.llvmlower import pipeline

    passes = ",".join(pipeline._INDEX_CONVERSION_PASSES)
    for extra in (name, f"{name}{{index-bitwidth=32}}", f"{name}{{index-bitwidth=64}}"):
        with pytest.raises(ValueError, match="no unique bare"):
            pipeline._bind_index_width(f"builtin.module({passes},{extra})", 64)
    for before, after in ((" ", ""), ("", " "), ("\t", "\n"), ("\n ", " \t")):
        extra = f"{before}{name}{after}{{index-bitwidth=32}}"
        with pytest.raises(ValueError, match="no unique bare"):
            pipeline._bind_index_width(f"builtin.module({passes},{extra})", 64)


def test_the_layout_is_asked_of_the_compiler_for_its_flags():
    from merlin.llvmlower import toolchain

    clang = toolchain.clang()
    if not clang or not __import__("pathlib").Path(str(clang)).is_file():
        pytest.skip("no cross clang in this checkout")
    layout = TDL.of(clang, ["--target=riscv64-unknown-elf", "-march=rv64gc", "-O2"])
    assert layout.startswith("e-") and "i64:64" in layout


@pytest.mark.parametrize(
    ("flags", "bits"),
    [
        (["--target=riscv32-unknown-elf", "-march=rv32gc", "-mabi=ilp32d"], 32),
        (["--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d"], 64),
    ],
)
def test_fresh_selected_compiler_observation_binds_exact_cross_flags(flags, bits):
    from merlin.llvmlower import toolchain

    clang = toolchain.clang()
    if not clang or not __import__("pathlib").Path(str(clang)).is_file():
        pytest.skip("no cross clang in this checkout")
    selected = TDL.observe_index_width(clang, flags)
    assert selected["cross_flags"] == flags
    assert selected["index_bits"] == bits
    from merlin.common.digest import sha256_file

    assert selected["compiler_sha256"] == sha256_file(selected["compiler_resolved"])


def test_selected_index_width_changes_actual_llvm_argument_type(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    src = """module {
      func.func @f(%arg0: index) -> i64 {
        %0 = arith.index_cast %arg0 : index to i64
        return %0 : i64
      }
    }"""
    passes = (
        "convert-index-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,"
        "convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts"
    )
    for bits in (32, 64):
        selected = {}
        ll = lower_to_llvm_ir(
            src,
            workdir=tmp_path / str(bits),
            pipeline=passes,
            data_layout=f"e-p:{bits}:{bits}",
            index_bits=bits,
            lowering_selection=selected,
        )
        assert f"define i64 @f(i{bits}" in ll
        assert selected["index_bits"] == bits
        assert selected["effective_pipeline"].count(f"index-bitwidth={bits}") == 5


def test_selected_width_and_effective_pipeline_survive_lower_model_receipt(tmp_path):
    from merlin.llvmlower.lower import lower_model

    src = """module {
      func.func @f(%arg0: index) -> i64 {
        %0 = arith.index_cast %arg0 : index to i64
        return %0 : i64
      }
    }"""
    lowered = lower_model(src, tmp_path, targets=(), data_layout="e-p:64:64", index_bits=64)
    selection = lowered.stats["index_lowering"]
    assert selection["index_bits"] == 64
    assert selection["data_layout"] == "e-p:64:64"
    assert selection["effective_pipeline"].count("index-bitwidth=64") == 5
    assert 'target datalayout = "e-p:64:64"' in lowered.ll_path.read_text()


def test_selfcopy_effective_pipeline_is_exact_runner_argument(tmp_path, monkeypatch):
    from merlin.llvmlower import pipeline
    from merlin.llvmlower.selfcopy import FEATURE

    original_run = pipeline.subprocess.run
    commands = []

    def observed_run(command, **kwargs):
        commands.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(pipeline.subprocess, "run", observed_run)
    src = """module {
      func.func @f(%arg0: index) -> i64 {
        %0 = arith.index_cast %arg0 : index to i64
        return %0 : i64
      }
    }"""
    selected = {}
    pipeline.lower_to_llvm_ir(
        src,
        workdir=tmp_path,
        features=frozenset({FEATURE}),
        data_layout="e-p:64:64",
        index_bits=64,
        lowering_selection=selected,
    )
    assert commands and selected["effective_pipeline"] == commands[0][4]
    assert selected["effective_pipeline"].count("index-bitwidth=64") == 5


def test_a_lowering_with_the_layout_aligns_64_bit_accesses_naturally(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    src = """module {
      func.func @f(%a: memref<8xi64>, %b: memref<8xf64>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        scf.for %i = %c0 to %c8 step %c1 {
          %x = memref.load %a[%i] : memref<8xi64>
          %y = arith.sitofp %x : i64 to f64
          memref.store %y, %b[%i] : memref<8xf64>
        }
        return
      }
    }"""
    default = lower_to_llvm_ir(src, workdir=tmp_path / "default")
    assert "load i64, ptr" in default and "align 4" in default, "LLVM's default layout under-aligns i64"
    layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
    got = lower_to_llvm_ir(src, workdir=tmp_path / "target", data_layout=layout)
    assert f'target datalayout = "{layout}"' in got
    loads = [line for line in got.splitlines() if "load i64" in line or "store double" in line]
    assert loads and all("align 8" in line for line in loads)


def test_implicit_pointer_observation_uses_actual_compiler_and_llvm_and_rejects_substitution(monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from merlin.llvmlower import toolchain

    clang = toolchain.clang()
    if not clang or not Path(clang).is_file():
        pytest.skip("select the actual Clang/LLVM toolchain for implicit-pointer observation")
    flags = ["--target=x86_64-unknown-linux-gnu"]
    observed = TDL.observe_index_width(clang, flags)
    assert observed["schema"] == "merlin.selected-index-lowering.v2"
    assert observed["index_bits"] == TDL.selected_index_bits(observed) == 64
    assert not TDL._has_default_pointer(observed["data_layout"])
    proof = observed["pointer_observation"]
    assert proof["representation_source"] == TDL._POINTER_SOURCE
    assert proof["representation_argv"] == [
        observed["compiler_resolved"],
        *flags,
        "-O1",
        "-S",
        "-emit-llvm",
        "-x",
        "c",
        "-",
        "-o",
        "-",
    ]
    assert proof["observer_sha256"] == TDL.sha256_file(proof["observer_source"])
    for field in ("representation_ir", "gep_source", "gep_ir"):
        changed = deepcopy(observed)
        changed["pointer_observation"][field] += "changed"
        assert TDL.selected_index_bits(changed) is None
    for field, value in (
        ("cross_flags", ["--target=other"]),
        ("index_bits", 32),
        ("data_layout", observed["data_layout"] + "-p:64:64:64:32"),
        ("schema", "merlin.selected-index-lowering.v1"),
    ):
        assert TDL.selected_index_bits({**observed, field: value}) is None
    changed = deepcopy(observed)
    changed["pointer_observation"]["gep_argv"] = proof["representation_argv"]
    assert TDL.selected_index_bits(changed) is None
    changed = deepcopy(observed)
    changed["pointer_observation"]["gep_ir"] = proof["gep_ir"].replace("  ret i65", "  call void @unbound()\n  ret i65")
    changed["pointer_observation"]["gep_ir_sha256"] = TDL._hash(changed["pointer_observation"]["gep_ir"])
    assert TDL.selected_index_bits(changed) is None

    # Actual LLVM semantics with a wide pointer but narrower indexing discard
    # that high bit. No architecture-name assumption may turn this into success.
    opt = Path(clang).with_name("opt")
    if opt.is_file():
        import subprocess

        layout = "e-p:64:64:64:32"
        result = subprocess.run(
            [str(opt), "-S", "-passes=instcombine", "-"],
            input=TDL._gep_source(layout, 64),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "ret i65 0" in result.stdout
        with pytest.raises(ValueError, match="widths differ"):
            TDL._check_gep(result.stdout, layout, 64)
        with pytest.raises(ValueError, match="widths differ"):
            TDL.default_index_bits(layout)
    original = TDL.sha256_file
    compiler = Path(observed["compiler_resolved"])
    for watched, reason in (
        (compiler, "cross compiler changed"),
        (Path(proof["observer_source"]), "observer source changed"),
    ):
        count = 0

        def changed_hash(path):
            nonlocal count
            if Path(path) == watched:
                count += 1
                if count > 1:
                    return "0" * 64
            return original(path)

        with monkeypatch.context() as custody:
            custody.setattr(TDL, "sha256_file", changed_hash)
            with pytest.raises(ValueError, match=reason):
                TDL.observe_index_width(clang, flags)
