"""The recipe surface exposed over the FROZEN gemmini backend: is it the same compiler, and are the
levers real?

Three things have to hold before a recipe number means anything, and each has burned this tree before:

* **the default recipe must emit the certified compiler's artifact, byte for byte.** Otherwise a
  measured delta is attributable to the refactor rather than to the recipe, and the frozen package's
  recorded Verilator cycles no longer describe what runs.
* **each value must change the EMITTED CODE** -- counts for a deletion, order for a reordering. A
  lever proven only by a different recipe string is the inert-lever failure
  (see the sibling ``test_gemmini_schedule.py``, whose arithmetic this mirrors).
* **an illegal recipe must be refused with a named reason**, never silently emitted. The frozen
  matmul path has no capacity check at all, so a shape whose operands do not fit collides instead of
  spilling; the refusal is what makes that a verdict rather than a wrong answer.

⚠️ ``test_single_n_tile_makes_panel_a_noop`` is the load-bearing negative control. Every shape in the
GSIM equivalence certificate is m=n=16, i.e. ``Nt=1``, and this test states why that set cannot see
the residency lever: with no N sweep there is no cross-column reuse to win, so the two values MUST
emit identical code. A "win" measured there would be measuring something else.

The package is driven as a SUBPROCESS, never imported: it is ``integrity_exempt: false`` (it may not
import merlin, and merlin must not import it), and a subprocess is also how the real harness invokes
it, so the test exercises the delivered path.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir, repo_root

FROZEN = artifacts_dir() / "targets/gemmini/gemmini_xdsl_rtl_v0/mlir_oot"
FORK = artifacts_dir() / "targets/gemmini/gemmini_xdsl_recipe_v0/mlir_oot"

pytestmark = pytest.mark.skipif(
    not (FORK / "gemmini_opt.py").exists() or not (FROZEN / "gemmini_opt.py").exists(),
    reason="the frozen backend and/or its recipe fork are not materialised in this checkout")

IFACE = """module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "gemmini", \
merlin_iface.abi_version = "0.1"}} {{
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{K}x{N}xi8>
  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{M}x{K}xi8>
  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{K}x{N}xi8>) \
-> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{M}x{K}xi8>, !merlin_iface.resident) \
-> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = [], output_dtype = "i32"}} : \
(!merlin_iface.acc<i32>) -> tensor<{M}x{N}xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}}
"""

DIM = 16


def _emit(pkg: Path, mlir: Path, recipe: dict | None) -> str:
    env = dict(os.environ)
    env.pop("MERLIN_CODEGEN_RECIPE", None)
    if recipe is not None:
        env["MERLIN_CODEGEN_RECIPE"] = json.dumps(recipe)
    r = subprocess.run([sys.executable, "gemmini_opt.py", "--convert-iface-to-gemmini",
                        "--emit-target-artifact", str(mlir)],
                       cwd=str(pkg), capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, f"emit failed ({recipe}): {r.stderr[-800:]}"
    return r.stdout


def _shape_mlir(tmp_path: Path, m: int, n: int, k: int) -> Path:
    p = tmp_path / f"w_{m}x{n}x{k}.mlir"
    p.write_text(IFACE.format(M=m, N=n, K=k), encoding="utf-8")
    return p


def _mvins(artifact: str) -> int:
    """MVIN instructions in the emitted stream, counted structurally (no regex, per the repo rule).

    Each accelerator op is one ``.insn r <opcode>, <funct3>, <funct>, ...``; MVIN's funct comes from
    the emitter's own table rather than being written down here.
    """
    sys.path.insert(0, str(FORK))
    from lowering.isa import FUNCT                                    # noqa: PLC0415
    want = FUNCT["MVIN"]
    n = 0
    for line in artifact.splitlines():
        if "llvm.inline_asm" not in line or ".insn " not in line:
            continue
        body = line.partition(".insn ")[2].split('"', 1)[0]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) >= 3 and parts[2].startswith("0x") and int(parts[2], 16) == want:
            n += 1
    return n


def _classes(artifact: str) -> list[str]:
    """The ORDERED sequence of instruction functs, so a reordering is observable at all."""
    out: list[str] = []
    for line in artifact.splitlines():
        if "llvm.inline_asm" not in line:
            continue
        if '"fence"' in line:
            out.append("fence")
            continue
        body = line.partition(".insn ")[2].split('"', 1)[0]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) >= 3 and parts[2].startswith("0x"):
            out.append(parts[2])
    return out


# --------------------------------------------------------------------------- is it the same compiler

CERTIFIED = ["isa/A2_single_tile_matmul", "isa/A3_k_accumulation", "isa/A5_relu_epilogue",
             "isa/A6_resident_reuse", "isa/A7_edge_padding", "_perf/PK03_k128"]


@pytest.mark.parametrize("capsule", CERTIFIED)
def test_default_recipe_emits_the_certified_artifact(capsule: str) -> None:
    """With no recipe selected the fork is the frozen compiler, byte for byte.

    This is the gate that makes every other number in the experiment attributable: it fails the moment
    the refactor changes the default path, which is the one thing a recipe surface must never do.
    """
    mlir = repo_root() / "merlin/contract/capsules" / capsule / "capsule.interface.mlir"
    if not mlir.exists():
        pytest.skip(f"{capsule} carries no interface MLIR in this checkout")
    frozen = _emit(FROZEN, mlir, None)
    fork = _emit(FORK, mlir, None)
    assert hashlib.sha256(fork.encode()).hexdigest() == \
        hashlib.sha256(frozen.encode()).hexdigest(), (
        f"{capsule}: the fork's DEFAULT recipe diverged from the certified backend")


# --------------------------------------------------------------------------- are the levers real

def test_panel_residency_matches_the_reuse_arithmetic(tmp_path: Path) -> None:
    """64x64x64: activation transfers drop from Mt*Nt*Kt to Mt*Kt; weight transfers are unchanged.

    The same arithmetic the in-tree emitter's lever is held to, so the two implementations of this
    lever are pinned to one statement rather than drifting apart.
    """
    mlir = _shape_mlir(tmp_path, 64, 64, 64)
    mt = nt = kt = 4
    per_tile = _mvins(_emit(FORK, mlir, {"activation_residency": "per_tile", "drain": "inline"}))
    panel = _mvins(_emit(FORK, mlir, {"activation_residency": "panel", "drain": "inline"}))
    assert per_tile == kt * nt + mt * nt * kt      # weights + one activation move per output column
    assert panel == kt * nt + mt * kt              # weights + one activation panel per row
    assert panel < per_tile


def test_panel_saving_scales_with_the_n_sweep(tmp_path: Path) -> None:
    """The saving is Mt*Kt*(Nt-1) transfers, so it GROWS with N -- which is why the best recipe is
    shape-dependent even though one value happens to win everywhere."""
    for (m, n, k) in ((16, 512, 256), (64, 64, 64), (32, 32, 32)):
        mt, nt, kt = -(-m // DIM), -(-n // DIM), -(-k // DIM)
        mlir = _shape_mlir(tmp_path, m, n, k)
        per_tile = _mvins(_emit(FORK, mlir, {"activation_residency": "per_tile", "drain": "inline"}))
        panel = _mvins(_emit(FORK, mlir, {"activation_residency": "panel", "drain": "inline"}))
        assert per_tile - panel == mt * kt * (nt - 1), f"at {m}x{n}x{k}"


def test_single_n_tile_saves_no_transfers(tmp_path: Path) -> None:
    """⚠️ THE NEGATIVE CONTROL. With Nt=1 there is no cross-column reuse, so panel residency can save
    NOTHING -- the transfer counts and the whole instruction multiset are identical.

    It is not, however, a literal no-op, and that distinction was worth discovering: hoisting the
    activation transfer above the (single-iteration) N loop still emits it BEFORE the weight transfer
    instead of after, so the ORDER changes while the counts do not. Asserting byte-identical output
    here is therefore wrong; asserting "no saving is available" is the real invariant.

    This is also why the GSIM equivalence certificate cannot validate this lever: all four of its
    members are m=n=16, i.e. Nt=1, so the saving this recipe exists to buy is exactly zero on every
    certified shape. Any "win" measured there is measuring something else.
    """
    mlir = _shape_mlir(tmp_path, 16, 16, 128)          # Nt=1, Kt=8
    a = _emit(FORK, mlir, {"activation_residency": "per_tile", "drain": "inline"})
    b = _emit(FORK, mlir, {"activation_residency": "panel", "drain": "inline"})
    assert _mvins(a) == _mvins(b), "with no N sweep there is no activation reuse to win"
    assert sorted(_classes(a)) == sorted(_classes(b)), "and no instruction may appear or vanish"


def test_deferred_drain_reorders_without_changing_the_multiset(tmp_path: Path) -> None:
    """`drain` moves the stores, it does not add or remove any.

    The store COUNT is an RTL-derived invariant (``test_rtl_filecheck`` asserts Mt*Nt), so this lever
    must be observable in the ORDER and nowhere else. Judging it by a histogram reports it as inert.
    """
    mlir = _shape_mlir(tmp_path, 64, 64, 64)
    inline = _classes(_emit(FORK, mlir, {"activation_residency": "per_tile", "drain": "inline"}))
    deferred = _classes(_emit(FORK, mlir, {"activation_residency": "per_tile", "drain": "deferred"}))
    assert sorted(inline) == sorted(deferred), "the instruction multiset must be preserved"
    assert inline != deferred, "a drain policy that changes nothing about the order is inert"


# --------------------------------------------------------------------------- refusals are named

def _recipe_mod():
    sys.path.insert(0, str(FORK / "lowering"))
    import recipe                                                    # noqa: PLC0415
    return recipe


def test_an_unknown_recipe_is_refused_not_ignored() -> None:
    R = _recipe_mod()
    with pytest.raises(ValueError, match="unknown recipe dimension"):
        R.Recipe.parse('{"tile_size": 32}')
    with pytest.raises(ValueError, match="not a value this compiler can emit"):
        R.Recipe(activation_residency="row_panel")


def test_a_shape_that_does_not_fit_is_refused_with_the_bound_named() -> None:
    """The frozen lowering stages both operand grids whole, so 32x512x512 collides. The refusal has to
    name the bound and the arithmetic -- 'illegal' with no reason is what makes a search walk a
    fictional space."""
    R = _recipe_mod()
    f = R.fit(R.Recipe(), m=32, n=512, k=512, dim=DIM, spad_rows=16384, acc_rows=1024)
    assert not f.ok
    assert "operand store" in f.reason and "16384" in f.reason
    assert f.operand_rows > f.operand_capacity
    ok = R.fit(R.Recipe(), m=64, n=64, k=64, dim=DIM, spad_rows=16384, acc_rows=1024)
    assert ok.ok and ok.reason == ""


def test_the_catalog_reports_legality_per_value_for_a_shape() -> None:
    """`choices` is the agent's whole view of the space, so every value carries its own verdict; a
    caller must never have to infer a refusal from an omission."""
    R = _recipe_mod()
    cat = R.catalog(m=64, n=64, k=64, dim=DIM, spad_rows=16384, acc_rows=1024)
    # DERIVED from the catalog's own dimensions rather than hardcoded, so adding a value cannot
    # break this test for a reason that has nothing to do with what it checks -- while still
    # asserting the catalog is internally consistent about how many points it claims.
    import math
    expected = math.prod(len(v) for v in cat["dimensions"].values())
    assert cat["n_total"] == expected, "n_total disagrees with the dimensions it enumerates"
    assert cat["n_legal"] == cat["n_total"], "every point should be legal at a shape that fits"
    assert cat["n_legal"] >= 15, ("the agentic arm needs a space a 16-evaluation budget cannot "
                                  "exhaust; below ~15 points a search measures nothing")
    assert set(cat["dimensions"]) == {"activation_residency", "config_policy", "drain",
                                      "block_m", "block_n", "block_k"}
    for entries in cat["dimensions"].values():
        assert sum(1 for e in entries if e["is_default"]) == 1
    # 32x512x512 is past the SINGLE-BLOCK bound: before blocking it had no legal point at all.
    # It now has many, and the catalog must say both things -- that it does not fit whole, and why --
    # because "expressible" and "needs no cutting" are different properties and a caller reasoning
    # about locality needs the second one.
    blocked = R.catalog(m=32, n=512, k=512, dim=DIM, spad_rows=16384, acc_rows=1024)
    assert not blocked["fits_without_cutting"]
    assert "operand store" in blocked["why_cutting_is_needed"]
    assert blocked["n_legal"] > 0, "blocking exists precisely so this shape is emittable"
    assert blocked["derived_block"]["block_k"] < 512, "the derived cut must actually cut something"


def test_auto_is_identical_to_panel_on_every_expressible_shape(tmp_path: Path) -> None:
    """``auto`` is the PROPOSED NEW DEFAULT, and its whole claim is that it needs no predicate.

    It began as a capacity rule ("panel if both grids fit"), which the code falsified: the two
    residency values reserve the SAME rows -- the frozen lowering stages the whole activation grid
    either way -- so capacity cannot discriminate between them. What survives is stronger: `panel` is
    never slower and never less expressible, so `auto` is just `panel`. This pins that equivalence on
    the emitted code, including the Nt=1 shapes where the saving is zero, so a future edit that
    reintroduces a predicate has to justify itself against a failing test.
    """
    for (m, n, k) in ((32, 32, 32), (64, 64, 64), (16, 512, 256),
                      (16, 16, 128), (128, 16, 128), (48, 96, 48)):
        mlir = _shape_mlir(tmp_path, m, n, k)
        auto = _emit(FORK, mlir, {"activation_residency": "auto", "drain": "inline"})
        panel = _emit(FORK, mlir, {"activation_residency": "panel", "drain": "inline"})
        assert auto == panel, f"auto diverged from panel at {m}x{n}x{k}"


def test_past_capacity_defeats_both_residency_values_equally() -> None:
    """The capacity cliff is a COMPILER COVERAGE gap, not a reason to prefer one residency value.

    If `per_tile` survived a shape `panel` could not, keeping it as a fallback would be justified. It
    does not: both reserve Kt*(Mt+Nt) rows, so the same shape defeats both, and the fix is a
    blocked-residency value this surface does not yet have.
    """
    R = _recipe_mod()
    for arm in ("per_tile", "panel", "auto"):
        f = R.fit(R.Recipe(activation_residency=arm), m=32, n=512, k=512,
                  dim=DIM, spad_rows=16384, acc_rows=1024)
        assert not f.ok, f"{arm} unexpectedly fits a shape past the operand-store bound"
        assert "operand store" in f.reason


# --------------------------------------------------------------------------- wave B

def test_every_residency_value_emits_distinct_code(tmp_path: Path) -> None:
    """Four staging values, four distinct emissions -- no value is a relabelling of another.

    `panel` and `a_prefetch` move the SAME number of activation tiles (Mt*Kt), so a count-only check
    would call one of them inert. They differ in WHERE the transfers sit, which is the whole point,
    so the comparison is on the ordered emission.
    """
    mlir = _shape_mlir(tmp_path, 64, 64, 64)
    seen: dict[str, str] = {}
    for value in ("per_tile", "panel", "a_prefetch", "prefetch_all"):
        art = _emit(FORK, mlir, {"activation_residency": value, "config_policy": "per_mvin",
                                 "drain": "inline"})
        digest = hashlib.sha256(art.encode()).hexdigest()
        assert digest not in seen.values(), f"{value} emits the same code as {seen}"
        seen[value] = digest
    assert len(set(seen.values())) == 4


def test_prefetch_values_move_the_same_tiles_as_panel(tmp_path: Path) -> None:
    """Hoisting further must not change WHAT is transferred, only when.

    `panel`, `a_prefetch` and `prefetch_all` all reduce activation transfers to Mt*Kt; weight
    transfers stay Kt*Nt in every value. If a hoist changed a count it would be moving different
    work, and the cycle comparison between these values would not be like-for-like.
    """
    mlir = _shape_mlir(tmp_path, 64, 64, 64)
    mt = nt = kt = 4
    for value in ("panel", "a_prefetch", "prefetch_all"):
        n = _mvins(_emit(FORK, mlir, {"activation_residency": value,
                                      "config_policy": "per_mvin", "drain": "inline"}))
        assert n == kt * nt + mt * kt, f"{value} moved {n} tiles, expected {kt * nt + mt * kt}"


def test_on_change_config_cuts_configs_without_touching_transfers(tmp_path: Path) -> None:
    """`on_change` must remove CONFIG_LDs and NOTHING else.

    That is the safety property: the stride bleed that produced wrong values came from a config that
    no longer described the following transfer. Here every transfer still has its stride programmed;
    only the redundant re-programming is dropped. So transfer counts must be untouched.
    """
    sys.path.insert(0, str(FORK))
    from lowering.isa import FUNCT                                    # noqa: PLC0415
    mlir = _shape_mlir(tmp_path, 64, 64, 64)
    for residency in ("per_tile", "panel"):
        base = _emit(FORK, mlir, {"activation_residency": residency,
                                  "config_policy": "per_mvin", "drain": "inline"})
        lean = _emit(FORK, mlir, {"activation_residency": residency,
                                  "config_policy": "on_change", "drain": "inline"})
        assert _mvins(lean) == _mvins(base), "transfers must be untouched"
        cfg_base = sum(1 for c in _classes(base) if c == hex(FUNCT["CONFIG_LD"]))
        cfg_lean = sum(1 for c in _classes(lean) if c == hex(FUNCT["CONFIG_LD"]))
        assert cfg_lean < cfg_base, f"{residency}: on_change removed no configs ({cfg_lean})"


# --------------------------------------------------------------------------- blocking (wave C)
#
# WHY THIS SECTION EXISTS. The frozen lowering staged both operand grids whole and kept every output
# tile live across the reduction, so it needed `Kt*(Mt+Nt) <= 1024` AND `Mt*Nt <= 64` -- and enforced
# NEITHER on the matmul path. MEASURED consequence (2026-09-03, `census_workloads.py` over the two
# captured claim models): of ResNet-50's 21 distinct contraction shapes and TinyLlama's 5, **zero**
# satisfied both bounds. The certified compiler could not emit one real layer of either model.
#
# Blocking is therefore a capability, not a tuning knob, and these tests hold it to the two things
# that make it safe: a shape that already fit must be emitted exactly as before, and a shape that is
# cut must still be the same computation.


def _blocks(m: int, n: int, k: int, recipe: dict | None = None):
    R = _recipe_mod()
    rec = R.Recipe.parse(recipe) if recipe else R.Recipe()
    return R.blocks(rec, m=m, n=n, k=k, dim=DIM, spad_rows=16384, acc_rows=1024)


@pytest.mark.parametrize("m,n,k", [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 512, 256),
                                   (16, 16, 2304), (8, 16, 5632)])
def test_a_shape_that_already_fits_is_one_block_and_is_emitted_unchanged(
        tmp_path: Path, m: int, n: int, k: int) -> None:
    """Byte-identity under blocking is STRUCTURAL, not a coincidence: a fitting shape yields exactly
    one block and the block body is the frozen nest. Asserting both together is what makes the
    equivalence gate mean 'the same compiler' rather than 'the same output on the cases we tried'."""
    plan = _blocks(m, n, k)
    assert plan.ok and plan.n_blocks == 1, f"{m}x{n}x{k} should need no cutting: {plan.as_dict()}"
    w = _shape_mlir(tmp_path, m, n, k)
    assert _emit(FROZEN, w, None) == _emit(FORK, w, None)


@pytest.mark.parametrize("m,n,k", [(64, 12544, 147),      # ResNet-50 conv1, im2col
                                   (512, 49, 4608),       # ResNet-50 layer4 3x3
                                   (8, 2048, 2048),       # TinyLlama q_proj
                                   (8, 32000, 2048),      # TinyLlama lm_head
                                   (1, 1000, 2048)])      # ResNet-50 classifier, M=1
def test_every_real_model_shape_becomes_expressible(m: int, n: int, k: int) -> None:
    """The capability claim, stated over the shapes the two claim models actually contain.

    Each of these is refused outright by the pre-blocking `fit` -- that is the measured 0/26. What
    blocking has to deliver is a legal cut for every one of them, derived with no agent involvement.
    """
    R = _recipe_mod()
    assert not R.fit(R.Recipe(), m=m, n=n, k=k, dim=DIM, spad_rows=16384,
                     acc_rows=1024).ok, "shape was expected to be inexpressible before blocking"
    plan = _blocks(m, n, k)
    assert plan.ok, f"{m}x{n}x{k} still has no legal block: {plan.reason}"
    assert plan.derived, "the default must need no chosen value"
    assert plan.n_blocks > 1


@pytest.mark.parametrize("m,n,k", [(1, 1000, 2048), (8, 2048, 2048)])
def test_blocking_never_rounds_a_sub_tile_extent_up(m: int, n: int, k: int) -> None:
    """M=1 (a classifier) and M=8 (decode at sequence 8) are REAL regimes in these two models.

    A block extent that floors at one tile would silently turn M=1 into M=16 -- not a smaller version
    of the workload but a different one, and 16x the output. The bug existed in the sizing helper
    before it was caught, so it is pinned here.
    """
    plan = _blocks(m, n, k)
    assert plan.ok and plan.bm <= m and plan.bn <= n and plan.bk <= k


def test_the_store_count_is_invariant_under_blocking(tmp_path: Path) -> None:
    """`test_rtl_filecheck.py` asserts MVOUT_COUNT == ceil(M/DIM)*ceil(N/DIM) from RTL-derived facts.

    Blocking reorders when a tile is stored; it may not change how many stores there are. A K cut in
    particular must NOT store a partial sum -- the store waits for the last K block.
    """
    sys.path.insert(0, str(FORK))
    from lowering.isa import FUNCT                                    # noqa: PLC0415
    m, n, k = 32, 32, 32
    want = (m // DIM) * (n // DIM)
    w = _shape_mlir(tmp_path, m, n, k)
    for recipe in (None, {"block_k": "16"}, {"block_n": "16"}, {"block_m": "16"},
                   {"block_m": "16", "block_n": "16", "block_k": "16"},
                   {"block_k": "16", "drain": "deferred"}):
        classes = _classes(_emit(FORK, w, recipe))
        got = sum(1 for c in classes if c.startswith("0x") and int(c, 16) == FUNCT["MVOUT"])
        assert got == want, f"{recipe}: {got} stores, expected {want}"


def test_a_k_cut_accumulates_onto_the_tile_instead_of_overwriting_it(tmp_path: Path) -> None:
    """The one way a K cut silently produces wrong numbers: clearing the accumulator per block.

    A K reduction split across blocks is still ONE reduction, so exactly the first K step of the
    first K block may overwrite; every later step accumulates. That shows up as the count of
    non-accumulating PRELOAD destinations, which must not grow with the number of K blocks.
    """
    m, n, k = 16, 16, 64
    w = _shape_mlir(tmp_path, m, n, k)

    def _fresh_dests(artifact: str) -> int:
        sys.path.insert(0, str(FORK))
        from lowering.isa import ACC_ACCUMULATE, FUNCT                # noqa: PLC0415
        n_fresh = 0
        for line in artifact.splitlines():
            if "llvm.inline_asm" not in line or ".insn " not in line:
                continue
            parts = [p.strip() for p in line.partition(".insn ")[2].split('"', 1)[0].split(",")]
            if len(parts) >= 3 and parts[2].startswith("0x") and int(parts[2], 16) == FUNCT["PRELOAD"]:
                n_fresh += 1
        return n_fresh

    base = _fresh_dests(_emit(FORK, w, None))
    for bk in ("16", "32"):
        assert _fresh_dests(_emit(FORK, w, {"block_k": bk})) == base, (
            f"block_k={bk} changed the preload count; a K cut must not add or drop reduction steps")


def test_an_illegal_block_is_refused_with_the_bound_named() -> None:
    """A chosen block is not trusted. The frozen failure mode for an oversized shape was a NEGATIVE
    weight base -- a wrong answer, not an error -- so a caller must not be able to reselect it."""
    plan = _blocks(8, 2048, 2048, {"block_k": "99999"})
    assert not plan.ok
    assert "operand store" in plan.reason or "accumulator" in plan.reason
    assert "does not fit" in plan.reason
    ok = _blocks(8, 2048, 2048, {"block_n": "256", "block_k": "512"})
    assert ok.ok and not ok.derived and (ok.bn, ok.bk) == (256, 512)


def test_the_derived_block_is_maximal_under_both_bounds() -> None:
    """`derive_blocks` claims to be an exhaustive maximisation, not a heuristic, so check the claim:
    no legal block has more work than the derived one. A heuristic that merely looks reasonable is
    how a 'derived' rule quietly becomes a tuned constant."""
    R = _recipe_mod()
    for (m, n, k) in [(64, 12544, 147), (512, 49, 4608), (8, 2048, 2048), (16, 512, 256)]:
        bm, bn, bk = R.derive_blocks(m, n, k, dim=DIM, spad_rows=16384, acc_rows=1024)
        ceil = lambda x: -(-x // DIM)  # noqa: E731 -- a partial tile still occupies one
        best = ceil(bm) * ceil(bn) * ceil(bk)
        for mt in range(1, min(-(-m // DIM), 64) + 1):
            for nt in range(1, min(-(-n // DIM), 64 // mt) + 1):
                kt = min(-(-k // DIM), 1024 // (mt + nt))
                if kt >= 1 and R._fits(mt, nt, kt, dim=DIM, spad_rows=16384, acc_rows=1024):
                    assert mt * nt * kt <= best, (
                        f"{m}x{n}x{k}: block {mt}x{nt}x{kt} beats the derived {bm}x{bn}x{bk}")


def test_a_fused_pool_refuses_to_be_split_rather_than_pooling_a_partial_image() -> None:
    """The pooled store reads a whole flattened image out of the accumulator in one MVOUT, so it
    cannot see tiles a later block has not computed. Refusing is the only correct answer."""
    R = _recipe_mod()
    plan = R.blocks(R.Recipe(), m=64, n=12544, k=147, dim=DIM, spad_rows=16384, acc_rows=1024)
    assert plan.ok and plan.n_blocks > 1, "shape must need cutting for this test to mean anything"
