"""Post-codegen census — the gate that makes an erased model impossible to ship.

``smolvla_int8_consistent`` linked a 512 MB ELF whose ``forward`` was 3,654 bytes and called only
``malloc``/``memset``/``roundevenf``. The build reported success, and on a board that binary would
have produced a spectacular speedup for computing nothing. The cause is fixed elsewhere
(``fix_bool_fptosi``); this file gates the *class*, so the next poison source cannot ship silently.

Every test here is a MUTATION test where it can be: the census is shown to FAIL on an object that
is not the model, not merely to pass on one that is. A check that cannot fail is the failure mode
this repo keeps re-discovering.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _module_with_ops(n: int) -> str:
    """A module whose ``forward`` chains ``n`` elementwise ``linalg.generic`` ops into its result."""
    body = ["builtin.module { func.func @forward(%x: tensor<8xf32>) -> tensor<8xf32> {"]
    cur = "%x"
    for i in range(n):
        body.append(
            f"%e{i} = tensor.empty() : tensor<8xf32> "
            f"%v{i} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, "
            f"affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]}} "
            f"ins({cur} : tensor<8xf32>) outs(%e{i} : tensor<8xf32>) {{ "
            f"^bb0(%a{i}: f32, %o{i}: f32): "
            f"%m{i} = arith.mulf %a{i}, %a{i} : f32 "
            f"linalg.yield %m{i} : f32 }} -> tensor<8xf32>")
        cur = f"%v{i}"
    body.append(f"func.return {cur} : tensor<8xf32> }} }}")
    return " ".join(body)


def _module_with_dead_ops(live: int, dead: int) -> str:
    """``live`` ops reach the result; ``dead`` ops hang off an unused ``tensor.empty``."""
    text = _module_with_ops(live)
    extra = []
    for i in range(dead):
        extra.append(
            f"%de{i} = tensor.empty() : tensor<8xf32> "
            f"%dv{i} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, "
            f"affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]}} "
            f"ins(%de{i} : tensor<8xf32>) outs(%de{i} : tensor<8xf32>) {{ "
            f"^bb0(%da{i}: f32, %do{i}: f32): "
            f"%dm{i} = arith.mulf %da{i}, %da{i} : f32 "
            f"linalg.yield %dm{i} : f32 }} -> tensor<8xf32>")
    return text.replace("func.return", " ".join(extra) + " func.return", 1)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- the obligation count, derived from the prepared IR --------------------------------------

def test_live_count_excludes_ops_that_reach_nothing():
    from merlin.llvmlower.codegen_census import live_structured_ops
    from merlin.frontends.linalg_mlir import parse_mlir_text

    live, total = live_structured_ops(parse_mlir_text(_module_with_dead_ops(5, 3)))
    assert (live, total) == (5, 8)      # dead ops are NOT charged to codegen


def test_live_count_follows_values_captured_by_nested_regions():
    """A gather reads its source with a ``tensor.extract`` INSIDE a linalg body — the captured
    value never appears in the enclosing op's operand list. Walking only operands severs the chain
    there and under-counts the live set by an order of magnitude, which would leave the census
    blind on exactly the models it exists for."""
    from merlin.llvmlower.codegen_census import live_structured_ops
    from merlin.frontends.linalg_mlir import parse_mlir_text

    src = (
        "builtin.module { func.func @forward(%i: tensor<4xindex>) -> tensor<4xf32> { "
        "%t = tensor.empty() : tensor<4xf32> "
        # produced ONLY for the gather below to read from inside a region
        "%tbl = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        "ins(%t : tensor<4xf32>) outs(%t : tensor<4xf32>) { "
        "^bb0(%a: f32, %o: f32): "
        "%m = arith.mulf %a, %a : f32 "
        "linalg.yield %m : f32 } -> tensor<4xf32> "
        "%e = tensor.empty() : tensor<4xf32> "
        "%g = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        "ins(%i : tensor<4xindex>) outs(%e : tensor<4xf32>) { "
        "^bb0(%iv: index, %go: f32): "
        "%x = tensor.extract %tbl[%iv] : tensor<4xf32> "
        "linalg.yield %x : f32 } -> tensor<4xf32> "
        "func.return %g : tensor<4xf32> } }")
    live, total = live_structured_ops(parse_mlir_text(src))
    assert total == 2
    assert live == 2, "the gathered table is live; only an operand-only walk would miss it"


# --- the delivery count, read off the object ---------------------------------------------------

def _clang():
    from merlin.llvmlower import toolchain

    return toolchain.clang() if toolchain.clang().is_file() else None


def _objdump():
    from merlin.llvmlower import toolchain

    return toolchain.objdump() if toolchain.objdump().is_file() else None


def _compile(tmp_path, name, c_source):
    clang = _clang()
    proc = subprocess.run([str(clang), "-O1", "-c", "-x", "c", "-", "-o", str(tmp_path / name)],
                          input=c_source, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    return tmp_path / name


_REAL_FORWARD = """
void forward(float *out, const float *in, long n) {
  for (long i = 0; i < n; i++) { float v = in[i]; out[i] = v * v + v; }
}
"""
_EMPTY_FORWARD = "void forward(float *out, const float *in, long n) { (void)out; (void)in; (void)n; }"


@pytest.mark.skipif(not _clang() or not _objdump(), reason="clang-23 / llvm-objdump missing")
def test_census_refuses_an_object_that_cannot_be_the_model(tmp_path):
    """THE mutation: same prepared IR, an entry point that does nothing. It must be refused."""
    from merlin.llvmlower.codegen_census import CodegenCensusError, require_commensurate

    prepared = _write(tmp_path, "model.prepared.mlir", _module_with_ops(400))
    obj = _compile(tmp_path, "empty.o", _EMPTY_FORWARD)
    with pytest.raises(CodegenCensusError) as excinfo:
        require_commensurate(prepared, obj, "forward")
    assert "live structured ops" in str(excinfo.value)


@pytest.mark.skipif(not _clang() or not _objdump(), reason="clang-23 / llvm-objdump missing")
def test_census_accepts_an_object_commensurate_with_the_ir(tmp_path):
    from merlin.llvmlower.codegen_census import require_commensurate

    prepared = _write(tmp_path, "model.prepared.mlir", _module_with_ops(2))
    obj = _compile(tmp_path, "real.o", _REAL_FORWARD)
    c = require_commensurate(prepared, obj, "forward")
    assert c.live_structured_ops == 2
    assert c.emitted_instructions >= c.live_structured_ops


@pytest.mark.skipif(not _clang() or not _objdump(), reason="clang-23 / llvm-objdump missing")
def test_census_refuses_a_missing_entry_point(tmp_path):
    """Fail closed: an absent symbol is a refusal, never an assumed pass."""
    from merlin.llvmlower.codegen_census import CodegenCensusError, require_commensurate

    prepared = _write(tmp_path, "model.prepared.mlir", _module_with_ops(2))
    obj = _compile(tmp_path, "real.o", _REAL_FORWARD)
    with pytest.raises(CodegenCensusError):
        require_commensurate(prepared, obj, "not_the_entry_point")


@pytest.mark.skipif(not _clang() or not _objdump(), reason="clang-23 / llvm-objdump missing")
def test_census_counts_outlined_helpers_not_just_the_entry_symbol(tmp_path):
    """Outlining is legitimate — an OpenMP lowering leaves ``forward`` a ~40-instruction shell that
    calls ``forward..omp_par``. Charging only the entry symbol would refuse every correct multicore
    build, so the delivery count is the whole object and the entry symbol only has to EXIST."""
    from merlin.llvmlower.codegen_census import require_commensurate

    outlined = """
    __attribute__((noinline)) void body(float *out, const float *in, long n) {
      for (long i = 0; i < n; i++) { float v = in[i]; out[i] = v * v + v; }
    }
    void forward(float *out, const float *in, long n) { body(out, in, n); }
    """
    prepared = _write(tmp_path, "model.prepared.mlir", _module_with_ops(6))
    obj = _compile(tmp_path, "outlined.o", outlined)
    c = require_commensurate(prepared, obj, "forward")
    assert c.entry_instructions < c.emitted_instructions
    assert c.emitted_instructions >= c.live_structured_ops


@pytest.mark.skipif(not _objdump(), reason="llvm-objdump missing")
def test_missing_objdump_is_a_refusal_not_a_skip(tmp_path):
    from merlin.llvmlower.codegen_census import CodegenCensusError, disassembly_census

    with pytest.raises(CodegenCensusError):
        disassembly_census(tmp_path / "nothing.o", objdump=tmp_path / "no-such-objdump")


# --- the gate is actually WIRED into the build ------------------------------------------------

def test_build_paths_call_the_census():
    """A gate nobody calls is a comment. Both object-emitting sites in the board build must run it.

    Asserted on the source because the alternative is a whole-model cross-compile in a unit test.
    """
    from merlin.common.paths import merlin_dir

    src = (merlin_dir() / "python" / "merlin" / "mining" / "k1.py").read_text(encoding="utf-8")
    assert src.count("require_commensurate") >= 2, "both model.o sites must run the census"
    # ...and after the compile, not somewhere it can be skipped when the compile is skipped.
    compile_at = src.index('"-c", res.ll_path, "-o", model_o')
    census_at = src.index("_census_require(prepared, model_o", compile_at)
    assert census_at > compile_at
