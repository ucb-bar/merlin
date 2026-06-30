"""Universal asm-normalizer (S8.3): build a curated kernel to objdump RVV asm, honestly.

Toolchain-independent tests always run (bogus-path -> None, parse a captured objdump
string, the dossier hook). The real saturn-build tests are skipped when the riscv
compiler/objdump are absent (CI without the chipyard toolchain).
"""
import pytest

from merlin.kernels import build_asm as B
from merlin.kernels.compare import RvvFingerprint
from merlin.kernels.types import NormalizedKernel

_HAVE_TOOLCHAIN = B.asm_toolchain_available()
_needs_tc = pytest.mark.skipif(not _HAVE_TOOLCHAIN,
                               reason="riscv gcc/objdump unavailable (set MERLIN_CHIPYARD)")

# A small, real objdump -d excerpt (column layout: addr<TAB>hex<TAB>mnemonic ...). This is
# the contract build_kernel_asm emits and RvvFingerprint.from_objdump consumes — no toolchain.
_CAPTURED_OBJDUMP = """
dotproduct.o:     file format elf64-littleriscv

Disassembly of section .text:

0000000000000000 <dotp_v32b>:
   0:\t0d07f057          \tvsetvli\tzero,a5,e32,m8,ta,ma
   4:\t02078207          \tvle32.v\tv4,(a5)
   8:\t0207c287          \tvle32.v\tv5,(a5)
   c:\tb6422257          \tvmul.vv\tv4,v4,v5
  10:\tb6422257          \tvmacc.vv\tv4,v4,v5
  14:\t06402257          \tvredsum.vs\tv4,v4,v0
  18:\t422022d7          \tvmv.x.s\ta5,v4
  1c:\t00008067          \tret
"""


# ------------------------------------------------------------------- toolchain-independent
def test_bogus_path_returns_none():
    assert B.build_kernel_asm("/no/such/kernel/file.c") is None


def test_vopacc_excluded():
    # the mining contract excludes VOPACC benches by name (never even attempts a build)
    assert B.saturn_benchmark_asm("vec-VOPACC-whatever") is None


def test_parse_captured_objdump_histogram():
    fp = RvvFingerprint.from_objdump(_CAPTURED_OBJDUMP, {"op": "dotprod", "dtype": "i32"}, "x")
    assert fp.histogram, "captured objdump should yield a non-empty mnemonic histogram"
    # canonical-op counts (compare.py canonicalizes vle32.v -> vle32, vsetvli -> vsetvl)
    assert fp.histogram.get("vle32") == 2
    assert fp.histogram.get("vmacc") == 1
    assert fp.decisions["vl_strategy"] == "vsetvl_loop"   # vsetvli => register-VL loop
    assert fp.decisions["fma_form"] == "vv"               # vmacc.vv


def test_top_mnemonics_on_captured():
    top = dict(B.top_mnemonics(_CAPTURED_OBJDUMP))
    assert top.get("vle32.v") == 2
    assert "vredsum.vs" in top
    assert all(m.startswith("v") for m in top)            # only vector mnemonics counted


def test_bench_name_extracted_from_path():
    assert B._bench_from_path("saturn-vectors/benchmarks/vec-dotprod/dotproduct.c") == "vec-dotprod"
    assert B._bench_from_path("nothing/here.c") is None


def test_dossier_asm_routes_unknown_source_to_none():
    nk = NormalizedKernel(source="exo", target="rvv", path="foo.py", op="gemm", dtype="f32")
    assert B.dossier_asm(nk) is None


# --------------------------------------------------------------------------- toolchain-gated
@_needs_tc
def test_saturn_dotprod_builds_rvv():
    asm = B.saturn_benchmark_asm("vec-dotprod")
    assert asm is not None, "vec-dotprod should compile standalone with the riscv toolchain"
    mnem = {m for m, _ in B.top_mnemonics(asm, 30)}
    # the disassembly must actually contain RVV instructions, not scalar-only code
    assert any(m.startswith("vsetvl") for m in mnem)
    assert any(m.startswith("vle") for m in mnem)
    assert any("macc" in m or "mul" in m for m in mnem)
    # and it must be consumable downstream
    fp = RvvFingerprint.from_objdump(asm, {"op": "dotprod", "dtype": "i32"}, "run")
    assert fp.histogram


@_needs_tc
def test_saturn_igemm_builds_rvv():
    asm = B.saturn_benchmark_asm("vec-igemm")
    assert asm is not None
    assert any(m.startswith("vsetvl") for m, _ in B.top_mnemonics(asm, 30))


@_needs_tc
def test_dossier_with_asm_sets_has_asm():
    from pathlib import Path
    tu = B.saturn_root() / "benchmarks/vec-dotprod/dotproduct.c"
    nk = NormalizedKernel(
        source="saturn", target="rvv",
        path="saturn-vectors/benchmarks/vec-dotprod/dotproduct.c",
        op="dotprod", dtype="i32",
        raw_text=tu.read_text() if tu.is_file() else "")
    d = B.build_dossier_with_asm(nk)
    assert d.to_dict()["has_asm"] is True


@_needs_tc
def test_xnnpack_best_effort_returns_none_gracefully():
    import glob
    files = glob.glob(str(B.repo_root() / "tmp/kernels/XNNPACK/src/**/*-rvv.c"),
                      recursive=True)
    if not files:
        pytest.skip("XNNPACK corpus not present")
    # a single-TU compile needs framework headers/params structs -> expected None, no exception
    assert B.framework_kernel_asm("xnnpack", files[0]) is None
