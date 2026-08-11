"""The residency check is only worth having if it can fail, so a non-resident kernel is compiled here.

The emitted microkernel reads the accumulator out after the reduction, so it is resident. A check that
only ever saw that kernel would pass for the wrong reason — it could be reading nothing at all, since these
instructions occupy reserved slots and a disassembler prints them as unnamed words. So the interesting test
compiles a variant that reads out *inside* the loop and requires the lifter to say so.

The other thing guarded here is the None case. A kernel with one accumulate has no reduction, and claiming
residency for it would be the strongest possible verdict on the weakest possible evidence.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.kernels import cca_matrix as CM
from merlin.kernels.opu_kernel import KernelSpec, emit_microkernel


class _Enc:
    def __init__(self, opcode: int, funct3: int, funct6: int):
        self.opcode, self.funct3, self.funct6 = opcode, funct3, funct6

    @property
    def funct7(self) -> int:
        return (self.funct6 << 1) | 1

    def insn_r(self, rd: str, rs1: str, rs2: str) -> str:
        return f".insn r {self.opcode:#x}, {self.funct3:#x}, {self.funct7:#x}, {rd}, {rs1}, {rs2}"


_TABLE = {"ACC": _Enc(0x57, 2, 40), "BCAST": _Enc(0x57, 6, 44), "READOUT": _Enc(0x57, 6, 46)}
_SPEC = KernelSpec(accumulate="ACC", broadcast="BCAST", readout="READOUT")

def _compile(src: str, tmp_path: Path, *, link: bool = True) -> Path:
    """Compile and (by default) LINK, because residency is scoped to a loop.

    An unlinked object still has placeholder branch displacements, so no back-edge span resolves and any
    loop-scoped count reads zero. `spans_reliable()` catches that and the lifter reports UNKNOWN -- which
    is the honest answer but tests nothing, so these fixtures link.
    """
    from merlin.llvmlower import toolchain
    if not toolchain.available():
        pytest.skip("needs the pinned clang")
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = tmp_path / "k.c"
    c.write_text(src, encoding="utf-8")
    obj = tmp_path / "k.o"
    proc = subprocess.run([toolchain.clang(), "--target=riscv64-unknown-elf", "-march=rv64gcv",
                           "-mabi=lp64d", "-O2", "-c", str(c), "-o", str(obj)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"compile failed:\n{proc.stderr[-2000:]}")
    if not link:
        return obj
    elf = tmp_path / "k.elf"
    ld = shutil.which("ld.lld") or shutil.which("riscv64-unknown-elf-ld")
    if ld is None:
        pytest.skip("needs a linker to resolve branch displacements (ld.lld)")
    proc = subprocess.run([ld, "-e", "0", "--no-check-sections", str(obj), "-o", str(elf)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"link failed:\n{proc.stderr[-2000:]}")
    return elf


#: The same reduction, but the accumulator is read out on every step. This is what losing residency looks
#: like on a unit whose accumulator is architected state: there is no register to spill, so the way to
#: round-trip it is to extract it early.
_NON_RESIDENT = f"""
#include <stdint.h>
#include <stddef.h>
void non_resident(int32_t *c, const int8_t *at, const int8_t *b, size_t m, size_t n, size_t k,
                  size_t ml, size_t nl) {{
  for (size_t kk = 0; kk < k; ++kk) {{
    asm volatile("vsetvli zero, %[nl], e8, m1, ta, ma\\n\\t"
                 "vle8.v v4, (%[bp])\\n\\t"
                 "vsetvli zero, %[ml], e8, m1, ta, ma\\n\\t"
                 "vle8.v v5, (%[ap])\\n\\t"
                 "{_TABLE['ACC'].insn_r('x1', 'x5', 'x4')}"
                 :: [ml] "r"(ml), [nl] "r"(nl), [ap] "r"(at + kk * m), [bp] "r"(b + kk * n)
                 : "memory");
    /* the commit, INSIDE the reduction */
    asm volatile("vsetvli zero, %[nl], e32, m1, ta, ma\\n\\t"
                 "{_TABLE['READOUT'].insn_r('x0', 'x0', 'x1')}\\n\\t"
                 "vse32.v v0, (%[cp])"
                 :: [nl] "r"(nl), [cp] "r"(c)
                 : "memory");
  }}
}}
"""

#: One accumulate, no reduction. Residency is not a meaningful question about this.
_SINGLE = f"""
#include <stdint.h>
#include <stddef.h>
void single(size_t ml, size_t nl, const int8_t *ap, const int8_t *bp) {{
  asm volatile("vsetvli zero, %[nl], e8, m1, ta, ma\\n\\t"
               "vle8.v v4, (%[bp])\\n\\t"
               "{_TABLE['ACC'].insn_r('x1', 'x5', 'x4')}"
               :: [ml] "r"(ml), [nl] "r"(nl), [ap] "r"(ap), [bp] "r"(bp) : "memory");
}}
"""

#: No accumulate at all — the unit is never driven.
_ABSENT = """
#include <stdint.h>
#include <stddef.h>
void absent(int32_t *c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = (int32_t)i; }
"""


@pytest.fixture(scope="module")
def resident_stream(tmp_path_factory):
    return _compile(emit_microkernel(_TABLE, _SPEC), tmp_path_factory.mktemp("res"))


@pytest.fixture(scope="module")
def non_resident_stream(tmp_path_factory):
    return _compile(_NON_RESIDENT, tmp_path_factory.mktemp("non"))


class TestResidencyIsReadFromTheStream:
    def test_the_emitted_kernel_is_resident(self, resident_stream):
        got = CM.stream_facts(resident_stream, _TABLE, accumulate="ACC", readout="READOUT")
        assert got.accumulator_resident is True
        # ONE static accumulate: the reduction is a loop, so it is emitted once and executed k times.
        # Expecting several here is what the linear "between first and last accumulate" rule assumed,
        # and why that rule could never judge a real looping kernel.
        assert got.accumulates == 1 and got.readouts >= 1
        assert got.reduction_is_loop is True

    def test_the_non_resident_variant_is_caught(self, non_resident_stream):
        # If this passed as resident, the check would be vacuous for every kernel.
        got = CM.stream_facts(non_resident_stream, _TABLE, accumulate="ACC", readout="READOUT")
        assert got.accumulator_resident is False
        assert any("per reduction step" in n for n in got.notes)

    def test_the_two_kernels_are_distinguished(self, resident_stream, non_resident_stream):
        a = CM.stream_facts(resident_stream, _TABLE, accumulate="ACC", readout="READOUT")
        b = CM.stream_facts(non_resident_stream, _TABLE, accumulate="ACC", readout="READOUT")
        assert a.accumulator_resident != b.accumulator_resident

    def test_the_reduction_is_recognised_as_a_loop(self, resident_stream):
        got = CM.stream_facts(resident_stream, _TABLE, accumulate="ACC", readout="READOUT")
        assert got.reduction_is_loop is True

    def test_a_single_accumulate_is_undetermined_not_resident(self, tmp_path):
        got = CM.stream_facts(_compile(_SINGLE, tmp_path), _TABLE, accumulate="ACC", readout="READOUT")
        assert got.accumulator_resident is None
        assert any("not a reduction" in n for n in got.notes)

    def test_a_stream_that_never_drives_the_unit_says_so(self, tmp_path):
        got = CM.stream_facts(_compile(_ABSENT, tmp_path), _TABLE, accumulate="ACC", readout="READOUT")
        assert got.accumulates == 0 and got.accumulator_resident is None
        assert any("not driven" in n for n in got.notes)

    def test_a_missing_encoding_raises_rather_than_reading_an_empty_stream(self, resident_stream):
        with pytest.raises(ValueError, match="no "):
            CM.stream_facts(resident_stream, {"ACC": _TABLE["ACC"]}, accumulate="ACC",
                            readout="READOUT")

    def test_identity_comes_from_the_table_not_from_a_mnemonic(self, resident_stream):
        # Shift every funct6: the same stream must now decode as containing none of the unit's ops. This
        # is what proves the lifter reads the derived table rather than pattern-matching text.
        shifted = {k: _Enc(v.opcode, v.funct3, v.funct6 + 2) for k, v in _TABLE.items()}
        got = CM.stream_facts(resident_stream, shifted, accumulate="ACC", readout="READOUT")
        assert got.accumulates == 0


class TestTheLiftedCCA:
    def test_residency_lands_on_the_compute_facet_too(self, resident_stream):
        # It belongs on compute because that is where the same question is asked for every other backend;
        # under `spatial` alone it would never diverge against a vector expert.
        cca = CM.lift_matrix_unit(resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT")
        assert cca.compute.accumulator_resident is True
        assert cca.spatial.accumulator_resident is True

    def test_the_contraction_form_is_its_own_token(self, resident_stream):
        cca = CM.lift_matrix_unit(resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT")
        assert cca.compute.contraction_form == CM.CONTRACTION_FORM != "systolic"

    def test_the_tile_geometry_is_carried_when_supplied(self, resident_stream):
        cca = CM.lift_matrix_unit(resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT", tile_rows=32, tile_cols=32)
        assert (cca.spatial.pe_rows, cca.spatial.pe_cols) == (32, 32)

    def test_the_stream_facts_are_kept_in_provenance(self, resident_stream):
        cca = CM.lift_matrix_unit(resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT")
        assert cca.provenance["stream"]["accumulates"] >= 1
        assert cca.provenance["stream"]["reduction_is_loop"] is True

    def test_a_non_resident_lift_reports_false_not_none(self, non_resident_stream):
        cca = CM.lift_matrix_unit(non_resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT")
        assert cca.compute.accumulator_resident is False


class TestTileOccupancy:
    def test_a_full_tile_is_one(self):
        assert CM.tile_occupancy(32, 32, 32) == 1.0

    def test_a_narrow_extent_is_the_reciprocal_of_the_tile(self):
        # The number a MAC count cannot see: correct, busy, and using one row in thirty-two.
        assert CM.tile_occupancy(1, 32, 32) == pytest.approx(1 / 32)
        assert CM.tile_occupancy(1, 1, 32) == pytest.approx(1 / 1024)

    def test_crossing_a_boundary_drops_occupancy(self):
        assert CM.tile_occupancy(33, 32, 32) == pytest.approx(33 / 64)

    def test_a_nonpositive_tile_raises(self):
        with pytest.raises(ValueError):
            CM.tile_occupancy(8, 8, 0)


class TestTheRoutes:
    def test_a_lost_residency_routes_to_an_epilogue_pass(self):
        from merlin.kernels import action_catalog as AC
        from merlin.kernels.cca_compare import Divergence
        CM.register_routes("matrix_test")
        got = AC.route(Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                                  backend="matrix_test"))
        assert got is not None and got.action_class == "PASS"
        assert got.intended_facet == {"compute.accumulator_resident": True}

    def test_registration_is_idempotent(self):
        from merlin.kernels import action_catalog as AC
        CM.register_routes("matrix_idem")
        CM.register_routes("matrix_idem")
        from merlin.kernels.cca_compare import Divergence
        # Registering twice must not produce two competing actions for the same seam.
        n = len(AC._ROUTES.get("matrix_idem", []))
        CM.register_routes("matrix_idem")
        assert len(AC._ROUTES.get("matrix_idem", [])) == n

    def test_the_codegen_route_excludes_the_narrow_regimes(self):
        from merlin.kernels import action_catalog as AC
        from merlin.kernels.cca_compare import Divergence
        CM.register_routes("matrix_regime")
        got = AC.route(Divergence(axis="compute.contraction_form", expert=CM.CONTRACTION_FORM,
                                  ours="vector", backend="matrix_regime"))
        assert got is not None and got.action_class == "CODEGEN"
        # An M=1 contraction is exactly what this action must not claim.
        assert not AC.applies_to_shape(got, "vector")
        assert not AC.applies_to_shape(got, "skinny")
        assert AC.applies_to_shape(got, "square_large")

    def test_the_profitable_regimes_are_real_regime_tokens(self):
        # A typo here would silently make the action apply to nothing.
        from merlin.kernels.bench_ceiling import shape_regime
        produced = {shape_regime("matmul", *mnk) for mnk in
                    [(1, 1, 64), (4, 4, 64), (64, 64, 64), (512, 512, 512), (256, 8, 64)]}
        assert set(CM.PROFITABLE_REGIMES) <= produced | {"square_large", "square_medium",
                                                         "rectangular"}
        assert produced & set(CM.PROFITABLE_REGIMES), "no sampled shape lands in a profitable regime"


class TestMatrixRegisterOccupancy:
    """Whether MRF depth is a lever reduces to: how many accumulator banks does the kernel occupy?"""

    def test_the_emitted_kernel_occupies_one_bank(self):
        # It accumulates into a single matrix register, so a unit with four leaves three idle. No MAC
        # count or cycle total says that; the destination field does.
        src = emit_microkernel(_TABLE, _SPEC)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            got = CM.stream_facts(_compile(src, Path(d)), _TABLE, accumulate="ACC", readout="READOUT")
        assert got.matrix_registers_used == 1

    def test_a_kernel_using_two_banks_is_distinguished(self, tmp_path):
        # A 1x2 register block over the matrix file: if this read 1, the facet could not see the lever.
        two = f"""
#include <stdint.h>
#include <stddef.h>
void two_banks(const int8_t *ap, const int8_t *bp, size_t ml, size_t nl, size_t k) {{
  for (size_t kk = 0; kk < k; ++kk) {{
    asm volatile("vsetvli zero, %[nl], e8, m1, ta, ma\\n\\t"
                 "vle8.v v4, (%[bp])\\n\\t"
                 "{_TABLE['ACC'].insn_r('x1', 'x5', 'x4')}\\n\\t"
                 "{_TABLE['ACC'].insn_r('x2', 'x5', 'x4')}"
                 :: [ml] "r"(ml), [nl] "r"(nl), [ap] "r"(ap), [bp] "r"(bp) : "memory");
  }}
}}
"""
        got = CM.stream_facts(_compile(two, tmp_path), _TABLE, accumulate="ACC", readout="READOUT")
        assert got.matrix_registers_used == 2

    def test_an_undriven_unit_reports_none_not_zero(self, tmp_path):
        # Zero banks would read as a measurement; None says the question does not apply.
        got = CM.stream_facts(_compile(_ABSENT, tmp_path), _TABLE, accumulate="ACC", readout="READOUT")
        assert got.matrix_registers_used is None

    def test_it_is_carried_in_the_lifted_provenance(self, resident_stream):
        cca = CM.lift_matrix_unit(resident_stream, _TABLE, op="matmul", source="t",
                                  accumulate="ACC", readout="READOUT")
        assert cca.provenance["stream"]["matrix_registers_used"] == 1
