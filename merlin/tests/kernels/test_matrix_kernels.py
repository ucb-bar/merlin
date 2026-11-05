"""Hand-written matrix kernels: the registry, selection, header, reference twin and constraint refusal.

Everything here is synthetic -- a package directory written by the test, encodings built from the
derivation's own classes -- so nothing names or needs a real unit. The C checks compile for the HOST.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.kernels import matrix_kernels as MK
from merlin.targetgen.rtl.matrix_isa import MaskedEncoding
from merlin.targetgen.rtl.opu_isa import Encoding

_HAS_CC = shutil.which("cc") is not None

_REGISTRY = """\
schema: merlin.matrix_kernels.v1
header:
  file: unit_x.h
  prefix: UX
defaults:
  unit_x: generated
kernels:
  generated:
    unit: unit_x
    generated: true
  plain:
    unit: unit_x
    file: plain.c
    function: plain_gemm
    description: a plain loop
  tiles_only:
    unit: unit_x
    file: plain.c
    function: plain_gemm
    constraints:
      m_multiple: tile
      n_multiple: 2*tile
      max_k: 64
      bias: false
  other_unit:
    unit: unit_y
    file: plain.c
    function: plain_gemm
"""

_PLAIN = """\
#include "unit_x.h"
#include "helper.h"
UX_KERNEL(plain_gemm)
{
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      int32_t s = bias ? bias[j] : 0;
      for (size_t q = 0; q < k; ++q) s += HELPER_MUL(at[q * m + i], b[q * n + j]);
      c[i * n + j] = s;
    }
}
"""


def _package(tmp_path: Path, registry: str = _REGISTRY) -> Path:
    root = tmp_path / "pkg"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "kernels.yaml").write_text(registry)
    (root / "kernels" / "plain.c").write_text(_PLAIN)
    (root / "kernels" / "helper.h").write_text("#define HELPER_MUL(a, b) ((int32_t)(a) * (int32_t)(b))\n")
    return root


# ---- registry ------------------------------------------------------------------------------------------


def test_registry_loads_defaults_header_and_constraints(tmp_path):
    reg = MK.load_registry(_package(tmp_path))
    assert reg.header == "unit_x.h" and reg.header_prefix == "UX"
    assert reg.default_for("unit_x").generated
    assert {e.name for e in reg.for_unit("unit_x")} == {"generated", "plain", "tiles_only"}
    c = reg.kernels["tiles_only"].constraints
    assert c.resolved(32) == {"m_multiple": 32, "n_multiple": 64, "k_multiple": 1, "max_m": None, "max_n": None,
                              "max_k": 64}  # fmt: skip
    assert reg.kernels["plain"].constraints.any_shape


def test_a_package_without_kernels_has_no_registry(tmp_path):
    assert MK.load_registry(tmp_path) is None


@pytest.mark.parametrize(
    "edit, message",
    [
        (("merlin.matrix_kernels.v1", "v0"), "schema"),
        (("    description: a plain loop", "    colour: blue"), "unknown key"),
        (("    function: plain_gemm\n    description", "    description"), "needs `file` and `function`"),
        (("  unit_x: generated", "  unit_x: nope"), "not a registered kernel"),
        (("      max_k: 64", "      max_q: 64"), "unknown constraint"),
        (("      n_multiple: 2*tile", "      n_multiple: lots"), "not an extent"),
        (("  prefix: UX", "  prefix: 9X"), "C identifier"),
    ],
)
def test_a_malformed_registry_is_refused_precisely(tmp_path, edit, message):
    old, new = edit
    assert old in _REGISTRY
    with pytest.raises(MK.KernelSelectionError, match=message):
        MK.load_registry(_package(tmp_path, _REGISTRY.replace(old, new, 1)))


# ---- selection -----------------------------------------------------------------------------------------


def test_no_selection_is_the_package_default(tmp_path):
    got = MK.resolve(None, package_root=_package(tmp_path), unit="unit_x")
    assert got.generated and got.origin == "default" and got.sha256 is None
    assert got.record()["generated"] is True


def test_no_registry_and_no_selection_is_the_generated_kernel(tmp_path):
    got = MK.resolve("", package_root=None, unit="unit_x")
    assert got.generated and got.name == "generated"


def test_a_registered_name_carries_its_file_sha256(tmp_path):
    root = _package(tmp_path)
    got = MK.resolve("plain", package_root=root, unit="unit_x")
    want = hashlib.sha256((root / "kernels" / "plain.c").read_bytes()).hexdigest()
    assert got.sha256 == want and got.function == "plain_gemm" and got.origin == "package"
    rec = got.record()
    assert rec["sha256"] == want and rec["file"].endswith("plain.c") and "reference" in rec["twin"]


def test_a_kernel_for_another_unit_is_refused(tmp_path):
    with pytest.raises(MK.KernelSelectionError, match="targets unit 'unit_y'"):
        MK.resolve("other_unit", package_root=_package(tmp_path), unit="unit_x")


def test_an_unknown_name_lists_what_is_registered(tmp_path):
    with pytest.raises(MK.KernelSelectionError, match="registered: generated, plain, tiles_only"):
        MK.resolve("plian", package_root=_package(tmp_path), unit="unit_x")


def test_a_path_outside_the_package_names_its_function_by_stem_or_suffix(tmp_path):
    root = _package(tmp_path)
    mine = tmp_path / "elsewhere" / "my_gemm.c"
    mine.parent.mkdir()
    mine.write_text(_PLAIN.replace("plain_gemm", "my_gemm"))
    got = MK.resolve(str(mine), package_root=root, unit="unit_x")
    assert got.origin == "path" and got.function == "my_gemm" and got.entry.constraints.any_shape
    assert got.header == "unit_x.h"  # compiled against the package's generated header
    other = tmp_path / "elsewhere" / "v2.c"
    other.write_text(_PLAIN.replace("plain_gemm", "fast_gemm"))
    assert MK.resolve(f"{other}:fast_gemm", package_root=root, unit="unit_x").function == "fast_gemm"
    with pytest.raises(MK.KernelSelectionError, match="does not define v2"):
        MK.resolve(str(other), package_root=root, unit="unit_x")


def test_a_path_inside_the_package_is_its_registered_entry(tmp_path):
    root = _package(tmp_path)
    got = MK.resolve(str(root / "kernels" / "plain.c"), package_root=root, unit="unit_x")
    assert got.name == "plain" and got.function == "plain_gemm"


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(MK.KernelSelectionError, match="neither a registered kernel"):
        MK.resolve(str(tmp_path / "nope.c"), package_root=_package(tmp_path), unit="unit_x")


# ---- constraints ---------------------------------------------------------------------------------------


def test_routed_shapes_outside_the_constraints_are_refused_by_name(tmp_path):
    k = MK.resolve("tiles_only", package_root=_package(tmp_path), unit="unit_x")
    MK.check_signatures(k, {"ok_0": (32, 64, 64), "ok_b": (3, 64, 128, 16)}, tile_edge=32)
    with pytest.raises(MK.KernelConstraintError) as exc:
        MK.check_signatures(k, {"ok_0": (32, 64, 64), "bad_1": (33, 96, 65)}, tile_edge=32)
    text = str(exc.value)
    assert "cannot take 1 of the 2 routed contraction(s)" in text
    assert "bad_1 (M=33 N=96 K=65)" in text
    assert "M=33 is not a multiple of m_multiple ('tile' = 32)" in text
    assert "N=96 is not a multiple of n_multiple ('2*tile' = 64)" in text
    assert "K=65 exceeds max_k (64)" in text
    assert "ok_0" not in text


def test_the_generated_and_unconstrained_kernels_take_any_shape(tmp_path):
    root = _package(tmp_path)
    for name in ("generated", "plain"):
        MK.check_signatures(MK.resolve(name, package_root=root, unit="unit_x"), {"s": (1, 7, 3)}, tile_edge=32)


# ---- the header ----------------------------------------------------------------------------------------


def _encodings():
    # A fixed-funct R-type word with all three registers free, and one whose rs2 the encoding fixes. Built
    # from the derivation's own classes; the values are arbitrary test data, not any unit's.
    free = Encoding(mnemonic="ACC", form="VV", opcode=0x2B, funct3=0x1, funct6=0x10, funct6_member="ACC")
    match = 0x2B | (0x2 << 12) | (5 << 20)
    mask = 0x7F | (0x7 << 12) | (0x1F << 20) | (0x7F << 25)
    fixed = MaskedEncoding(mnemonic="ZERO", match=match, mask=mask, origin="test")
    return {"ACC": free, "ZERO": fixed}


def test_the_header_spells_every_word_from_the_encodings():
    text = MK.emit_header(
        _encodings(), prefix="UX", header="unit_x.h", facts={"TILE_EDGE": (32, "edge"), "MASK": 0x60000000},
        aliases={"ACC": "AUTHOR_ACC"},
    )  # fmt: skip
    assert '#define UX_ACC(rd, rs1, rs2) ".insn r 0x2b, 0x1, 0x21, " rd ", " rs1 ", " rs2' in text
    # rs2 is fixed by the encoding, so the macro does not take it and spells it
    assert '#define UX_ZERO(rd, rs1) ".insn r 0x2b, 0x2, 0x0, " rd ", " rs1 ", x5"' in text
    assert "#define UX_ZERO_MATCH" in text and "#define UX_ZERO_MASK" in text
    assert "#define UX_ACC_R(rd, rs1, rs2) __asm__ volatile(UX_ACC(UX_REG(rd), UX_REG(rs1), UX_REG(rs2))" in text
    assert "#define UX_TILE_EDGE 32  /* edge */" in text and "#define UX_MASK 0x60000000" in text
    assert "AUTHOR_ACC" in text and "#define UX_KERNEL(name) void name(int32_t *c" in text


def test_the_header_refuses_an_unresolved_derivation():
    with pytest.raises(ValueError, match="did not cross-check"):
        MK.emit_header(_encodings(), prefix="UX", header="h.h", derivation_ok=False)


@pytest.mark.skipif(not _HAS_CC, reason="no host C compiler")
def test_the_header_preprocesses_to_the_words(tmp_path):
    (tmp_path / "unit_x.h").write_text(MK.emit_header(_encodings(), prefix="UX", header="unit_x.h"))
    (tmp_path / "use.c").write_text('#include "unit_x.h"\nconst char *s = UX_ACC(UX_REG(1), UX_REG(5), "%[a]");\n')
    out = subprocess.run(["cc", "-E", "-P", str(tmp_path / "use.c")], capture_output=True, text=True, check=True)
    assert '".insn r 0x2b, 0x1, 0x21, " "x" "1" ", " "x" "5" ", " "%[a]"' in out.stdout


def test_unit_header_reads_only_the_plugin_contract():
    provider = SimpleNamespace(
        load_contract=lambda unit: SimpleNamespace(declarations={"crosscheck_pairs": {"ACC": "VOPACC"}}),
        derive_encodings=lambda contract: SimpleNamespace(
            encodings=_encodings(), ok=True, sources={"rtl": "/x/I.scala"}
        ),
        geometry=lambda *, unit, config: (16, 8),
        kernel_header_facts=lambda *, unit, config: {"KSTEP": (4, "rows per instruction")},
    )
    text = MK.unit_header(provider, unit="u", config="C", header="u.h", prefix="UX")
    assert "#define UX_TILE_EDGE 16" in text and "#define UX_ALIGN_BYTES 8" in text and "#define UX_KSTEP 4" in text
    assert "VOPACC" in text and "rtl=I.scala" in text


# ---- the reference twin and the prepared build -------------------------------------------------------


def _compile_shared(tmp_path: Path, sources: list[Path], name: str) -> ctypes.CDLL:
    lib = tmp_path / f"{name}.so"
    subprocess.run(
        ["cc", "-O1", "-shared", "-fPIC", "-I", str(tmp_path), *map(str, sources), "-o", str(lib)], check=True
    )
    return ctypes.CDLL(str(lib))


def _call(lib, func: str, at, b, bias, m, n, k):
    c = np.zeros((m, n), dtype=np.int32)
    p = ctypes.POINTER
    fn = getattr(lib, func)
    fn.argtypes = [p(ctypes.c_int32), p(ctypes.c_int8), p(ctypes.c_int8), p(ctypes.c_int32)] + [ctypes.c_size_t] * 3
    fn(c.ctypes.data_as(p(ctypes.c_int32)), at.ctypes.data_as(p(ctypes.c_int8)), b.ctypes.data_as(p(ctypes.c_int8)),
       None if bias is None else bias.ctypes.data_as(p(ctypes.c_int32)), m, n, k)  # fmt: skip
    return c


@pytest.mark.skipif(not _HAS_CC, reason="no host C compiler")
def test_the_reference_is_the_abi_and_wraps(tmp_path):
    (tmp_path / "ref.c").write_text(MK.reference_source("ref_gemm"))
    lib = _compile_shared(tmp_path, [tmp_path / "ref.c"], "ref")
    rng = np.random.default_rng(0)
    for m, n, k, with_bias in ((5, 7, 3, False), (17, 4, 33, True)):
        at = rng.integers(-128, 128, size=(k, m), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        bias = rng.integers(-1000, 1000, size=n, dtype=np.int32) if with_bias else None
        want = at.astype(np.int64).T @ b.astype(np.int64) + (0 if bias is None else bias.astype(np.int64))
        assert np.array_equal(_call(lib, "ref_gemm", at, b, bias, m, n, k), want.astype(np.int32))
    # int32 wrap, as an accumulator without saturation does
    big = np.full((1, 1), 2**31 - 10, dtype=np.int32)
    one = np.ones((1, 1), dtype=np.int8) * 10
    assert _call(lib, "ref_gemm", one, one, big[0], 1, 1, 1)[0, 0] == np.int32(-(2**31) + 90)


@pytest.mark.skipif(not _HAS_CC, reason="no host C compiler")
def test_prepare_snapshots_the_kernel_and_the_twin_compiles_the_reference(tmp_path):
    root = _package(tmp_path)
    k = MK.resolve("plain", package_root=root, unit="unit_x")
    header = MK.emit_header(_encodings(), prefix="UX", header="unit_x.h")
    ship = MK.prepare(k, tmp_path / "ship", header_text=header, scalar_tile=False)
    twin = MK.prepare(k, tmp_path / "twin", header_text=header, scalar_tile=True)
    assert '#include "kernel/plain.c"' in ship.source and ship.function == twin.function == "plain_gemm"
    assert "plain.c" not in twin.source and "merlin.kernels.matrix_kernels" in twin.source
    assert set(ship.files) == {"plain.c", "helper.h", "unit_x.h"} and ship.files["plain.c"] == k.sha256
    assert ship.record["compiled"] == "kernel" and twin.record["compiled"] == "reference"
    rng = np.random.default_rng(1)
    at = rng.integers(-8, 9, size=(9, 6), dtype=np.int8)
    b = rng.integers(-8, 9, size=(9, 5), dtype=np.int8)
    outs = []
    for where, prepared in (("ship", ship), ("twin", twin)):
        (tmp_path / where / "tu.c").write_text(prepared.source)
        lib = _compile_shared(tmp_path / where, [tmp_path / where / "tu.c"], where)
        outs.append(_call(lib, "plain_gemm", at, b, None, 6, 5, 9))
    assert np.array_equal(outs[0], outs[1])


def test_prepare_refuses_a_file_named_like_the_header_and_an_edited_kernel(tmp_path):
    root = _package(tmp_path)
    k = MK.resolve("plain", package_root=root, unit="unit_x")
    (root / "kernels" / "plain.c").write_text(_PLAIN + "\n/* edited */\n")
    with pytest.raises(MK.KernelSelectionError, match="changed after it was selected"):
        MK.prepare(k, tmp_path / "w", header_text="", scalar_tile=False)
    (root / "kernels" / "unit_x.h").write_text("/* stale copy */\n")
    k = MK.resolve("plain", package_root=root, unit="unit_x")
    with pytest.raises(MK.KernelSelectionError, match="generated header's name"):
        MK.prepare(k, tmp_path / "w", header_text="", scalar_tile=False)


# ---- the routing hands the build exactly what it selected ---------------------------------------------


def test_the_default_kernel_adds_nothing_to_the_build_call(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm

    root = _package(tmp_path)
    monkeypatch.setattr(MK, "package_root", lambda target: root)
    routing = zm.MatrixRouting(support_target="pkg", unit="unit_x", config="C")
    assert MK.build_kwargs(routing, {"s": (1, 2, 3)}) == {}
    assert MK.for_routing(routing).generated


def test_a_selected_kernel_is_checked_then_handed_over_with_its_header(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm

    root = _package(tmp_path)
    provider = SimpleNamespace(
        load_contract=lambda unit: SimpleNamespace(declarations={}),
        derive_encodings=lambda contract: SimpleNamespace(encodings=_encodings(), ok=True, sources={}),
        geometry=lambda *, unit, config: (32, 16),
    )
    monkeypatch.setattr(MK, "package_root", lambda target: root)
    monkeypatch.setattr(zm.MatrixRouting, "provider", lambda self: provider)
    routing = zm.MatrixRouting(support_target="pkg", unit="unit_x", config="C", kernel="tiles_only")
    got = MK.build_kwargs(routing, {"s": (32, 64, 16)})["kernel"]
    assert isinstance(got, MK.KernelChoice) and got.selected.name == "tiles_only"
    assert "#define UX_TILE_EDGE 32" in got.header_text
    with pytest.raises(MK.KernelConstraintError, match="s \\(M=31 N=64 K=16\\)"):
        MK.build_kwargs(routing, {"s": (31, 64, 16)})


def test_an_unresolvable_package_still_has_a_default_but_never_a_named_kernel(monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen.plugins import PluginError

    def missing(target):
        raise PluginError("not selected")

    monkeypatch.setattr(MK, "package_root", missing)
    assert MK.for_routing(zm.MatrixRouting(support_target="t", unit="u", config="c")).generated
    with pytest.raises(PluginError):
        MK.for_routing(zm.MatrixRouting(support_target="t", unit="u", config="c", kernel="mine"))
