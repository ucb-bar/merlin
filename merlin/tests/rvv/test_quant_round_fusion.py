"""Gates for `fuse_quantize_round_convert` -- the quantize round/clamp/convert fusion.

Four properties, each of which has a matching failure mode this repo has already paid for:

* the rewrite is EXACT, checked as arithmetic rather than asserted in a comment;
* it is REFUSED, not approximated, wherever the equivalence argument does not hold;
* it is REACHABLE -- resolvable by name in a process that imports no proposer, which is the
  registration trap `wholemodel_proposer` documents and which this lever hit for real;
* it turns the tagger's `skip_math` refusal into tags, which is the entire point of it and the
  one thing a "the pass ran" counter cannot tell you.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root


def _parse(text):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    return parse_mlir_text(text)


_QUANTIZE = """
module {
  func.func @q(%a: tensor<4x8xf32>, %s: f32) -> tensor<4x8xi8> {
    %e = tensor.empty() : tensor<4x8xi8>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %s : tensor<4x8xf32>, f32) outs(%e : tensor<4x8xi8>) {
    ^bb0(%x: f32, %sv: f32, %o: i8):
      %hi = arith.constant 1.270000e+02 : f32
      %lo = arith.constant -1.270000e+02 : f32
      %d = arith.divf %x, %sv : f32
      %rd = math.roundeven %d : f32
      %m1 = arith.minimumf %rd, %hi : f32
      %m2 = arith.maximumf %m1, %lo : f32
      %q = arith.fptosi %m2 : f32 to i8
      linalg.yield %q : i8
    } -> tensor<4x8xi8>
    return %r : tensor<4x8xi8>
  }
}
"""


def _body_names(module):
    for op in module.walk():
        if op.name == "linalg.generic":
            return [o.name for o in op.regions[0].blocks[0].ops]
    return []


def test_the_quantize_chain_is_fused_and_leaves_no_math_op():
    from merlin.llvmlower.quant_round import fuse_round_clamp_convert

    module = _parse(_QUANTIZE)
    assert "math.roundeven" in _body_names(module)
    report: dict = {}
    assert fuse_round_clamp_convert(module, report_out=report) == 1
    names = _body_names(module)
    # The point of the pass is not that it is shorter -- it is that NO `math.*` op survives, because
    # the `merlin.vec_r{rank}` tagger refuses a generic that has one and that refusal is why these
    # ops carry no tag today.
    assert not [n for n in names if n.startswith("math.")], names
    assert "arith.fptosi" in names and "arith.select" in names
    assert report["rewrites"] == 1


@pytest.mark.parametrize("mutation,reason", [
    # a non-integral bound: the clamp no longer commutes with round-half-to-even (Lemma 1), so the
    # clamp cannot be moved in front of the round and the inline form has no bounded range.
    ("1.270000e+02 : f32", "clamp_bound_not_integral"),
    # a one-sided clamp: the argument is unbounded below, so `c - trunc(c)` is not exact.
    ("__drop_lower__", "clamp_not_two_sided"),
    # a bound the destination integer type cannot hold: `t` would overflow i8.
    ("__wide_bound__", "clamp_bound_outside_dest_int"),
])
def test_an_unprovable_chain_is_refused_and_counted(mutation, reason):
    """Each refusal is a case where the equivalence argument FAILS. It must decline and say which,
    not approximate -- an approximation here is an integer-valued op answering a different number."""
    from merlin.llvmlower.quant_round import fuse_round_clamp_convert

    text = _QUANTIZE
    if mutation == "__drop_lower__":
        text = text.replace("      %m2 = arith.maximumf %m1, %lo : f32\n", "")
        text = text.replace("arith.fptosi %m2", "arith.fptosi %m1")
    elif mutation == "__wide_bound__":
        text = text.replace("1.270000e+02 : f32", "1.000000e+03 : f32")
        text = text.replace("-1.270000e+02 : f32", "-1.000000e+03 : f32")
    else:
        text = text.replace("1.270000e+02 : f32", "1.275000e+02 : f32")
    module = _parse(text)
    report: dict = {}
    assert fuse_round_clamp_convert(module, report_out=report) == 0
    assert report.get(f"refused_{reason}"), report
    assert "math.roundeven" in _body_names(module), "a refused chain must be left ALONE"


def test_the_emitted_arithmetic_equals_the_chain_it_replaces():
    """The equivalence, checked as ARITHMETIC over every f32 the clamp cannot hide.

    `numpy` evaluates the two forms in f32 exactly as the emitted code does: `np.rint` is IEEE
    roundToIntegralTiesToEven, `np.minimum`/`np.maximum` propagate NaN like `arith.minimumf`, and
    `np.trunc` is `arith.fptosi`'s round-toward-zero. Sweeping every bit pattern with |v| <= 128
    covers every tie, every sign, every denormal and both zeroes; above that the clamp pins the
    result to a bound in both forms, which the explicit cases check.

    NaN is EXCLUDED, and that exclusion is the honest part: `arith.minimumf` propagates it, so the
    BASELINE reaches `fptosi(NaN)` and is POISON in LLVM. The rewrite is poison in the same op.
    Neither form is defined there, so there is nothing for a test to pin.
    """
    np = pytest.importorskip("numpy")

    hi, lo, half, zero = (np.float32(x) for x in (127.0, -127.0, 0.5, 0.0))

    def baseline(v):
        return np.maximum(np.minimum(np.rint(v), hi), lo)

    def fused(v):
        c = np.maximum(np.minimum(v, hi), lo)
        t = np.trunc(c).astype(np.float32)
        d = (c - t).astype(np.float32)
        ad = np.maximum(d, (-d).astype(np.float32))
        ti = t.astype(np.int32)
        bump = (ad > half) | ((ad == half) & ((ti & 1) != 0))
        step = np.where(c < zero, np.int32(-1), np.int32(1))
        return (ti + np.where(bump, step, np.int32(0))).astype(np.float32)

    checked = 0
    top = np.float32(128.0).view(np.uint32)
    chunk = 1 << 22
    for sign in (np.uint32(0), np.uint32(0x80000000)):
        start = 0
        while start <= int(top):
            n = min(chunk, int(top) - start + 1)
            v = (np.arange(start, start + n, dtype=np.uint32) | sign).view(np.float32)
            assert np.array_equal(baseline(v), fused(v)), f"mismatch near {v[0]!r}"
            checked += n
            start += n
    assert checked > 2_000_000_000, checked

    edge = np.array([127.5, -127.5, 126.5, -126.5, 128.0, -128.0, 1e30, -1e30,
                     np.inf, -np.inf, 2.0 ** 23, -(2.0 ** 23), 2.5, -2.5, 3.5, -3.5,
                     0.5, -0.5, 0.0, -0.0], dtype=np.float32)
    assert np.array_equal(baseline(edge), fused(edge))


def test_the_lever_resolves_in_a_process_that_imports_no_proposer():
    """The registration trap, gated. `k1.build_k1_binary` -> `normalize` imports no proposer, and
    the lowering SUBPROCESS re-imports `impr_features` fresh -- so a name registered only where it
    is RANKED raises "unknown impr feature" there. This failed for real before the lazy hook existed.
    """
    import subprocess
    import sys

    code = (
        "from merlin.llvmlower.impr_features import normalize\n"
        "print(','.join(sorted(normalize({'fuse_quantize_round_convert'}))))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(repo_root()))
    assert out.returncode == 0, out.stderr
    got = out.stdout.strip().split(",")
    # It must also drag in the lever that gives it its payoff: alone the rewrite trades a call for
    # inline arith and measures as a wash.
    assert "fuse_quantize_round_convert" in got
    assert "vectorize_non_contraction_generics" in got, got


def test_it_is_ranked_so_the_search_can_reach_it():
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS

    entry = [e for e in RANKED_LEVERS if e[0] == "fuse_quantize_round_convert"]
    assert len(entry) == 1, RANKED_LEVERS
    # Second element is `is_full_schedule_replacement`; this lever edits neither schedule nor
    # pipeline (it is a prepare-stage IR rewrite), so it composes with everything.
    assert entry[0][1] is False


def test_an_empty_feature_set_leaves_the_preparation_untouched():
    """The frozen-baseline invariant, at the seam this lever was added to: with no features the
    peephole must not be reachable at all, so a baseline build is byte-identical."""
    import inspect

    from merlin.runtime.backends import zephyr_model

    src = inspect.getsource(zephyr_model._prepare_model_mlir)
    assert "if fuse_quant_round:" in src, "the peephole must be behind an explicit flag"
    sig = inspect.signature(zephyr_model._prepare_model_mlir)
    assert sig.parameters["fuse_quant_round"].default is False
