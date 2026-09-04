"""The register-group width (LMUL) as a DIRECT seam — derivation, default-off, and emitted code.

The point of these tests is the last one. This repo is full of levers that report "applied" and emit
byte-identical code, so the load-bearing assertion here is not that a feature resolves or that a flag
list grew: it is that the LMUL field of the ``vsetvli`` the compiler emits actually moves, read back
off a real object file by the same decoder the CCA uses to measure the divergence in the first place.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.kernels.decode import rvv as rvv_decode
from merlin.llvmlower import impr_features as F
from merlin.llvmlower.lmul_group import (LMUL_LADDER, LMUL_OPTION, LmulDerivationError,
                                         extent_ceiling, group_elements, group_lmul,
                                         group_lmul_for_elem_types, group_lmul_for_shapes,
                                         lmul_cflags)


# ---- the derivation ---------------------------------------------------------------------

def test_width_is_the_ratio_not_a_constant():
    """acc_bits/operand_bits, rounded up to a whole-register group. Nobody types a 4."""
    assert group_lmul(operand_bits=8, acc_bits=32) == 4      # i8 x i8 -> i32 (the mined qd8 datapath)
    assert group_lmul(operand_bits=16, acc_bits=32) == 2     # bf16/f16 -> f32
    assert group_lmul(operand_bits=32, acc_bits=32) == 1     # f32 -> f32: no fraction to make whole
    assert group_lmul(operand_bits=8, acc_bits=64) == 8


def test_the_floor_is_the_matrix_paths_own_inequality():
    """``LMUL * operand_bits >= SEW`` — the same constraint cca_matrix states, read with SEW=acc."""
    from merlin.kernels.cca_matrix import vtype_spans_tile_row
    for operand_bits, acc_bits in ((8, 32), (16, 32), (32, 32), (8, 16)):
        lmul = group_lmul(operand_bits=operand_bits, acc_bits=acc_bits)
        assert vtype_spans_tile_row(acc_bits, lmul, operand_bits=operand_bits)
        # and it is the SMALLEST such group: one rung down must fail the same test.
        lower = [m for m in LMUL_LADDER if m < lmul]
        if lower:
            assert not vtype_spans_tile_row(acc_bits, lower[-1], operand_bits=operand_bits)


def test_vlen_enters_as_the_ceiling_not_the_floor():
    """The floor is VLEN-free; VLEN only caps the group against the work it will cover."""
    # the floor does not move with VLEN: with a cap too large to bind, a 128-bit and a 512-bit part
    # give the same width, because the ratio is all the floor is made of.
    assert (group_lmul(operand_bits=8, acc_bits=32, vlen=128, max_group_elems=4096)
            == group_lmul(operand_bits=8, acc_bits=32, vlen=512, max_group_elems=4096) == 4)
    # A 256-bit part holds 8 i32 elements per LMUL=1 group, so an extent of 8 admits only m1.
    assert group_elements(1, acc_bits=32, vlen=256) == 8
    assert group_lmul(operand_bits=8, acc_bits=32, vlen=256, max_group_elems=8) == 1
    assert group_lmul(operand_bits=8, acc_bits=32, vlen=256, max_group_elems=32) == 4
    # A narrower part reaches the same element count at a wider group.
    assert group_lmul(operand_bits=8, acc_bits=32, vlen=128, max_group_elems=16) == 4
    # the ceiling is readable on its own, and may legitimately sit BELOW the datapath floor
    assert extent_ceiling(acc_bits=32, vlen=256, max_group_elems=8) == 1
    assert extent_ceiling(acc_bits=32, vlen=256, max_group_elems=64) == 8


def test_fails_closed_rather_than_defaulting():
    with pytest.raises(LmulDerivationError):
        group_lmul(operand_bits=0, acc_bits=32)
    with pytest.raises(LmulDerivationError):
        group_lmul(operand_bits=32, acc_bits=8)             # accumulator narrower than its operands
    with pytest.raises(LmulDerivationError):
        group_lmul(operand_bits=1, acc_bits=32)             # would need LMUL=32; no such group
    with pytest.raises(LmulDerivationError):
        group_lmul(operand_bits=8, acc_bits=32, max_group_elems=32)   # a cap with no VLEN
    with pytest.raises(LmulDerivationError):
        group_lmul_for_elem_types("i8", "i8", "quux")
    with pytest.raises(LmulDerivationError):
        lmul_cflags(3)


def test_element_types_reach_the_same_answer():
    assert group_lmul_for_elem_types("i8", "i8", "i32") == 4
    assert group_lmul_for_elem_types("bf16", "bf16", "f32") == 2
    assert group_lmul_for_elem_types("f32", "f32", "f32") == 1


# ---- the feature: default OFF -----------------------------------------------------------

def test_baseline_flags_are_untouched():
    base = ["-march=rv64gcv", "-mabi=lp64d", "-O2"]
    assert F.apply_cflags(base, frozenset()) == base
    # a feature set with no cflag hook is equally inert
    assert F.apply_cflags(base, F.normalize(["lmul_widen_n"])) == base


def test_the_feature_appends_exactly_the_one_option():
    base = ["-march=rv64gcv", "-O2"]
    name = F.ensure_lmul_group(operand_bits=8, acc_bits=32)
    assert name == "lmul_group_m4" and name in F.known()
    out = F.apply_cflags(base, F.normalize([name]))
    assert out[:len(base)] == base
    assert out[len(base):] == ["-mllvm", f"{LMUL_OPTION}=4"]


def test_every_whole_register_width_is_registered_eagerly():
    """Names must resolve in the lowering SUBPROCESS too, so none of them may be lazy."""
    for lmul in LMUL_LADDER:
        assert F.lmul_group_feature(lmul) in F.known()
    assert F.ensure_lmul_group_for_elem_types("i8", "i8", "i32") == "lmul_group_m4"
    assert F.ensure_lmul_group_for_elem_types("f32", "f32", "f32") == "lmul_group_m1"


def test_the_feature_edits_no_schedule_and_no_pipeline():
    """The whole point: it moves the group width WITHOUT moving a tile size."""
    schedule = "tile_sizes [4, 8, 1] vector_sizes [4, 8, 1]"
    passes = ["canonicalize", "cse"]
    feats = F.normalize(["lmul_group_m4"])
    assert F.apply_schedule(schedule, feats) == schedule
    assert F.apply_pipeline(passes, feats) == passes


# ---- the routing ------------------------------------------------------------------------

def test_vector_lmul_routes_to_the_group_width_not_to_a_wider_n():
    """The N-tile route is the one that measured 5.18 ms -> 20.45 ms + a whole-model scalar fallback."""
    from merlin.kernels.cca_compare import Divergence
    from merlin.mining.fork_from_action import action_to_fork
    from merlin.kernels.action_catalog import route

    d = Divergence(axis="vector.lmul", expert=4.0, ours=2.0, backend="rvv")
    action = route(d)
    assert action is not None and action.divergence_axis == "vector.lmul"
    knobs = {"op_match": [{"op": "matmul", "tile": [1, 32, 1], "vector": [1, 32, 1]}]}
    prop = action_to_fork(action, knobs)
    assert prop.forkable and prop.lever == "feature"
    assert prop.overrides == {"compiler_features": [F.LMUL_GROUP_SENTINEL]}
    assert "op_match" not in prop.overrides          # no tile was touched
    # and the seam the router advertises no longer names the N-width knob
    assert "vector_sizes" not in action.target_seam
    # the narrowing direction is the same seam (one rung, both signs), not a tile edit either
    down = route(Divergence(axis="vector.lmul", expert=2.0, ours=4.0, backend="rvv"))
    assert action_to_fork(down, knobs).overrides == {"compiler_features": [F.LMUL_GROUP_SENTINEL]}


def test_the_sentinel_is_a_request_not_a_width():
    """It must NOT be registered: an unresolved request has to raise, not read as a group width."""
    assert F.LMUL_GROUP_SENTINEL not in F.known()
    with pytest.raises(KeyError):
        F.normalize([F.LMUL_GROUP_SENTINEL])


def test_the_sentinel_resolves_from_the_prepared_irs_own_element_types():
    """prepare_for_lowering swaps the request for the width the module's arithmetic derives."""
    from merlin.kernels.microkernel import ContractionShape
    from merlin.llvmlower.lmul_group import group_lmul_for_shapes

    i8 = ContractionShape(op="linalg.matmul", parallel=(128, 256), reduction=(256,),
                          dtypes=("i8", "i8", "i32"))
    f32 = ContractionShape(op="linalg.matmul", parallel=(128, 256), reduction=(256,),
                           dtypes=("f32", "f32", "f32"))
    assert group_lmul_for_shapes([i8], vlen=256) == 4
    assert group_lmul_for_shapes([f32], vlen=256) == 1
    # one flag configures the whole object, so the most demanding datapath in the module wins
    assert group_lmul_for_shapes([f32, i8], vlen=256) == 4
    # a contraction too narrow to fill even one group caps it, VLEN-dependently
    narrow = ContractionShape(op="linalg.matmul", parallel=(1, 8), reduction=(64,),
                              dtypes=("i8", "i8", "i32"))
    assert group_lmul_for_shapes([narrow], vlen=256) == 1
    assert group_lmul_for_shapes([narrow], vlen=64) == 4
    # dtypes the observer could not read are skipped, not defaulted; nothing readable fails closed
    unknown = ContractionShape(op="linalg.matmul", parallel=(128, 256), reduction=(256,))
    assert group_lmul_for_shapes([unknown, i8], vlen=256) == 4
    with pytest.raises(LmulDerivationError):
        group_lmul_for_shapes([unknown], vlen=256)


def test_the_sentinel_is_actually_consumed_by_the_preparation():
    """A request nobody resolves is worse than no request: it raises in `normalize` at lowering time.

    Pinned the same way `test_per_op_register_block_is_forkable_because_it_is_wired` pins its
    sentinel -- by reading the source of the step that has to consume it.
    """
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin/python/merlin/runtime/backends/zephyr_model.py").read_text()
    assert "if LMUL_GROUP_SENTINEL in features:" in src           # the sentinel IS consumed
    assert "group_lmul_for_shapes(_cshapes0(prepared), vlen=vlen)" in src   # ...from the prepared IR
    assert "lmul_group_feature(" in src                           # ...and swapped for a real feature
    # and the resolution sits ABOVE the `blocking` gate, since a `blocking=False` build would
    # otherwise carry the unresolved sentinel into `normalize`.
    assert src.index("if LMUL_GROUP_SENTINEL in features:") < src.index("    if not blocking:")


# ---- the only assertion that matters: the emitted vsetvli --------------------------------

_KERNEL_C = """\
void mac(int n, const int *a, const int *b, int *c) {
  for (int i = 0; i < n; i++) c[i] += a[i] * b[i];
}
"""


def _dominant_lmul(obj: Path) -> float:
    """The LMUL of the most frequent vtype in ``obj`` — the same reading the CCA's vector facet takes."""
    hist = rvv_decode.decode(obj).vtype_histogram()
    assert hist, f"{obj} disassembled to no vector configuration at all"
    top = max(hist.items(), key=lambda kv: kv[1])[0]         # e.g. "e32m2tama"
    rest = top[top.index("m"):]
    if rest.startswith("mf"):
        digits = "".join(c for c in rest[2:] if c.isdigit())
        return 1.0 / int(digits)
    digits = ""
    for ch in rest[1:]:
        if not ch.isdigit():
            break
        digits += ch
    return float(int(digits))


def test_the_emitted_vsetvli_lmul_field_actually_moves(tmp_path):
    """Compile the SAME source twice, changing only the feature, and read the group width back.

    A lever that reports "applied" and emits byte-identical code is the default failure mode in this
    repo, so this compiles, disassembles, and compares the digest as well as the vtype.
    """
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.codegen import RISCV_FLAGS
    try:
        clang = toolchain.clang()
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"no clang: {exc}")
    if not Path(str(clang)).is_file():
        pytest.skip(f"clang not installed at {clang}")
    src = tmp_path / "mac.c"
    src.write_text(_KERNEL_C)

    def build(features) -> Path:
        obj = (tmp_path / (("f_" + "_".join(sorted(features))) if features else "baseline")
               ).with_suffix(".o")
        flags = F.apply_cflags(list(RISCV_FLAGS), F.normalize(features))
        proc = subprocess.run([str(clang), *flags, "-c", str(src), "-o", str(obj)],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        return obj

    base = build([])
    wider = build(["lmul_group_m4"])
    base_lmul, wide_lmul = _dominant_lmul(base), _dominant_lmul(wider)
    assert base.read_bytes() != wider.read_bytes(), (
        "the feature changed nothing in the emitted object — an inert lever, not a seam")
    assert wide_lmul > base_lmul, (
        f"register group did not widen: baseline m{base_lmul:g} -> feature m{wide_lmul:g}")
    # and the narrowing direction is reachable from the same seam
    narrow = build(["lmul_group_m1"])
    assert _dominant_lmul(narrow) < base_lmul


def test_baseline_object_is_byte_identical_without_the_feature(tmp_path):
    """The default-off invariant, measured on bytes rather than asserted in a docstring."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.codegen import RISCV_FLAGS
    try:
        clang = toolchain.clang()
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"no clang: {exc}")
    if not Path(str(clang)).is_file():
        pytest.skip(f"clang not installed at {clang}")
    src = tmp_path / "mac.c"
    src.write_text(_KERNEL_C)
    outs = []
    for i, feats in enumerate((frozenset(), F.normalize(["lmul_widen_n"]))):
        obj = tmp_path / f"b{i}.o"
        flags = F.apply_cflags(list(RISCV_FLAGS), feats)
        assert flags == list(RISCV_FLAGS)
        proc = subprocess.run([str(clang), *flags, "-c", str(src), "-o", str(obj)],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        outs.append(obj.read_bytes())
    assert outs[0] == outs[1]
