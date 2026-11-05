"""Any int8 contraction, read as a GEMM and rewritten into the matrix kernel's one layout -- checked by VALUE.

Each case writes a contraction in a layout a frontend can produce (a transposed operand, two batch dims,
an NCHW 1x1 conv, a strided and dilated direct conv, a permuted output), routes it, and then evaluates
BOTH modules in numpy: the original op by its own indexing maps, and the rewritten one op by op, with
the kernel call computed as ``(b?, m, k) @ (b?, k, n)``. Equal int32 results are the claim; a structural
check alone would accept a rewrite that hands the kernel the right shapes in the wrong order.

The refusals are tested the same way they are reported: by reason, for the forms that are not a GEMM.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import contraction_normal_form as cnf
from merlin.llvmlower import int8_contractions as IC

PREFIX = "cnf_test_mm"
_BODY = """
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
"""


def _contraction(a_shape, b_shape, c_shape, maps: list[str], iterators: list[str]) -> str:
    def t(shape, el):
        return "tensor<" + "x".join(map(str, shape)) + f"x{el}>"

    its = ", ".join(f'"{i}"' for i in iterators)
    amaps = ", ".join(f"affine_map<{m}>" for m in maps)
    return f"""
builtin.module {{
  func.func @forward(%a: {t(a_shape, "i8")}, %b: {t(b_shape, "i8")}) -> {t(c_shape, "i32")} {{
    %e = tensor.empty() : {t(c_shape, "i32")}
    %z = arith.constant 0 : i32
    %f = linalg.fill ins(%z : i32) outs(%e : {t(c_shape, "i32")}) -> {t(c_shape, "i32")}
    %r = linalg.generic {{indexing_maps = [{amaps}], iterator_types = [{its}]}}
        ins(%a, %b : {t(a_shape, "i8")}, {t(b_shape, "i8")})
        outs(%f : {t(c_shape, "i32")}) {{{_BODY}    }} -> {t(c_shape, "i32")}
    func.return %r : {t(c_shape, "i32")}
  }}
}}
"""


# ----------------------------------------------------------------------------------- numpy reference


def _np_dtype(ty):
    width = ty.get_element_type().width.data
    return {8: np.int8, 32: np.int32}[width]


def _eval_expr(expr, grid):
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expr, AffineDimExpr):
        return grid[expr.position]
    if isinstance(expr, AffineConstantExpr):
        return np.full_like(grid[0], expr.value)
    if isinstance(expr, AffineBinaryOpExpr):
        lhs, rhs = _eval_expr(expr.lhs, grid), _eval_expr(expr.rhs, grid)
        return {
            AffineBinaryOpKind.Add: lambda: lhs + rhs,
            AffineBinaryOpKind.Mul: lambda: lhs * rhs,
            AffineBinaryOpKind.Mod: lambda: lhs % rhs,
            AffineBinaryOpKind.FloorDiv: lambda: lhs // rhs,
        }[expr.kind]()
    raise AssertionError(f"unexpected affine expr {expr}")


def _eval_generic(op, env):
    maps = [m.data for m in op.indexing_maps]
    ins = [env[id(v)] for v in op.inputs]
    out = env[id(op.outputs[0])].copy()
    extents: dict[int, int] = {}
    from xdsl.ir.affine import AffineDimExpr

    for arr, amap in zip([*ins, out], maps):
        for axis, e in enumerate(amap.results):
            if isinstance(e, AffineDimExpr):
                extents[e.position] = arr.shape[axis]
    grid = np.meshgrid(*[np.arange(extents[d]) for d in range(len(extents))], indexing="ij")
    grid = [g.ravel() for g in grid]
    idx = [tuple(_eval_expr(e, grid) for e in amap.results) for amap in maps]
    names = [o.name for o in op.body.blocks[0].ops]
    if "arith.muli" in names:  # the contraction
        prod = ins[0][idx[0]].astype(np.int64) * ins[1][idx[1]].astype(np.int64)
        acc = out.astype(np.int64)
        np.add.at(acc, idx[2], prod)
        return acc.astype(np.int32)
    out[idx[1]] = ins[0][idx[0]]  # a pure gather: yields its input
    return out


def _run(module, a, b):
    func = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    block = func.body.blocks[0]
    env = {id(block.args[0]): a, id(block.args[1]): b}
    for op in block.ops:
        name = op.name
        if name == "func.return":
            return env[id(op.operands[0])]
        res = op.results[0] if op.results else None
        if name == "arith.constant":
            env[id(res)] = op.value.value.data
        elif name == "tensor.empty":
            env[id(res)] = np.zeros(res.type.get_shape(), _np_dtype(res.type))
        elif name == "linalg.fill":
            env[id(res)] = np.full(res.type.get_shape(), env[id(op.inputs[0])], _np_dtype(res.type))
        elif name == "linalg.transpose":
            env[id(res)] = np.transpose(env[id(op.inputs[0])], tuple(op.permutation.get_values()))
        elif name in ("tensor.collapse_shape", "tensor.expand_shape"):
            env[id(res)] = env[id(op.operands[0])].reshape(res.type.get_shape())
        elif name == "linalg.generic":
            env[id(res)] = _eval_generic(op, env)
        elif name == "func.call":
            lhs, rhs = (env[id(v)].astype(np.int64) for v in op.operands[:2])
            env[id(res)] = np.matmul(lhs, rhs).astype(np.int32)
            assert env[id(res)].shape == tuple(res.type.get_shape())
        else:
            raise AssertionError(f"the reference does not evaluate {name}")
    raise AssertionError("no func.return")


class _Force:
    """A selector that routes everything in a fixed orientation and records that decision."""

    def __init__(self, orientation):
        self.orientation = orientation

    def decide(self, shape, *, layout_elements=None):
        assert self.orientation in layout_elements, "the rewrite must offer every orientation it can emit"
        return SimpleNamespace(
            route=True, orientation=self.orientation, reason="forced", to_dict=lambda: {"reason": "forced"}
        )

    def __call__(self, shape):
        return True


def _check(text, a_shape, b_shape, orientation=cnf.AS_IS, *, steps=None):
    rng = np.random.default_rng(0)
    a = rng.integers(-127, 128, size=a_shape, dtype=np.int8)
    b = rng.integers(-127, 128, size=b_shape, dtype=np.int8)
    want = _run(parse_mlir_text(text), a, b)
    module = parse_mlir_text(text)
    got = IC.rewrite_contractions(module, select=_Force(orientation), symbol_prefix=PREFIX)
    assert got.count == 1, got.not_routed
    module.verify()
    # The contraction itself is gone: what is left is data movement and the kernel call.
    assert not any(o.name == "linalg.generic" and "arith.muli" in str(o) for o in module.walk())
    if steps is not None:
        assert set(got.routed[0].steps) == set(steps), got.routed[0].steps
    np.testing.assert_array_equal(_run(module, a, b), want)
    return got


# --------------------------------------------------------------------------------------------- cases

_MK, _KN, _MN = "(d0, d1, d2) -> (d0, d2)", "(d0, d1, d2) -> (d2, d1)", "(d0, d1, d2) -> (d0, d1)"
_PPR = ["parallel", "parallel", "reduction"]


class TestLayoutsAreRewrittenNotRefused:
    @pytest.mark.parametrize("orientation", [cnf.AS_IS, cnf.SWAPPED])
    def test_canonical(self, orientation):
        got = _check(_contraction((24, 40), (40, 20), (24, 20), [_MK, _KN, _MN], _PPR), (24, 40), (40, 20), orientation)
        if orientation == cnf.AS_IS:
            assert got.routed[0].steps == ()  # already the kernel's layout: nothing added
        else:
            assert (got.routed[0].m, got.routed[0].n) == (20, 24)  # the kernel sees C^T

    @pytest.mark.parametrize("orientation", [cnf.AS_IS, cnf.SWAPPED])
    def test_transposed_b(self, orientation):
        text = _contraction((24, 40), (20, 40), (24, 20), [_MK, "(d0, d1, d2) -> (d1, d2)", _MN], _PPR)
        _check(text, (24, 40), (20, 40), orientation)

    def test_transposed_a(self):
        text = _contraction((40, 24), (40, 20), (24, 20), ["(d0, d1, d2) -> (d2, d0)", _KN, _MN], _PPR)
        _check(text, (40, 24), (40, 20), steps={"transpose_a"})

    def test_rank4_attention_scores_q_kt(self):
        # (batch, head, s, d) . (batch, head, t, d) -> (batch, head, s, t): two batch dims, K transposed.
        maps = [
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)",
        ]
        its = ["parallel"] * 4 + ["reduction"]
        got = _check(
            _contraction((2, 3, 10, 8), (2, 3, 12, 8), (2, 3, 10, 12), maps, its), (2, 3, 10, 8), (2, 3, 12, 8)
        )
        r = got.routed[0]
        assert (r.batch, r.m, r.n, r.k) == (6, 10, 12, 8)  # batch dims collapsed into the batched entry

    def test_rank4_attention_pv_into_a_permuted_output(self):
        # P(b, h, s, t) . V(b, h, t, d) -> O(b, s, h, d): the output's head and seq dims swapped.
        maps = [
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)",
            "(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3)",
        ]
        its = ["parallel"] * 4 + ["reduction"]
        _check(
            _contraction((2, 3, 10, 12), (2, 3, 12, 8), (2, 10, 3, 8), maps, its),
            (2, 3, 10, 12),
            (2, 3, 12, 8),
            steps={"transpose_c", "collapse"},
        )

    def test_nchw_pointwise_conv_without_im2col(self):
        # out(n, f, h, w) = sum_c x(n, c, h, w) w(f, c): M = (n, h, w) is not contiguous in the output.
        maps = [
            "(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)",
            "(d0, d1, d2, d3, d4) -> (d1, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)",
        ]
        its = ["parallel"] * 4 + ["reduction"]
        _check(_contraction((2, 6, 5, 7), (9, 6), (2, 9, 5, 7), maps, its), (2, 6, 5, 7), (9, 6))

    @pytest.mark.parametrize("orientation", [cnf.AS_IS, cnf.SWAPPED])
    def test_strided_dilated_direct_conv(self, orientation):
        # out(n, f, oh, ow) = sum_{c,kh,kw} x(n, c, 2*oh + 2*kh, 2*ow + 2*kw) w(f, c, kh, kw)
        maps = [
            "(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2 + d5 * 2, d3 * 2 + d6 * 2)",
            "(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)",
            "(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)",
        ]
        its = ["parallel"] * 4 + ["reduction"] * 3
        # input 11x11, k=3, dilation 2, stride 2 -> output 3x3
        text = _contraction((1, 4, 11, 11), (5, 4, 3, 3), (1, 5, 3, 3), maps, its)
        got = _check(text, (1, 4, 11, 11), (5, 4, 3, 3), orientation)
        r = got.routed[0]
        assert any(s.startswith("im2col") for s in r.steps)
        assert r.k == 4 * 3 * 3


class TestWhatIsNotAGemm:
    def _why(self, text) -> str:
        got = IC.rewrite_contractions(parse_mlir_text(text), select=lambda _s: True, symbol_prefix=PREFIX)
        assert got.count == 0
        (r,) = got.not_routed
        return r.why

    def test_depthwise_is_refused_by_name(self):
        # out(n, c, oh, ow) = sum_{kh,kw} x(n, c, oh + kh, ow + kw) w(c, kh, kw): the weight has no free dim.
        maps = [
            "(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d3 + d5)",
            "(d0, d1, d2, d3, d4, d5) -> (d1, d4, d5)",
            "(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)",
        ]
        its = ["parallel"] * 4 + ["reduction"] * 2
        why = self._why(_contraction((1, 4, 6, 6), (4, 3, 3), (1, 4, 4, 4), maps, its))
        assert "depthwise" in why

    def test_a_dim_reduced_by_one_operand_alone(self):
        maps = ["(d0, d1, d2, d3) -> (d0, d2, d3)", "(d0, d1, d2, d3) -> (d2, d1)", "(d0, d1, d2, d3) -> (d0, d1)"]
        its = ["parallel", "parallel", "reduction", "reduction"]
        assert "one operand alone" in self._why(_contraction((4, 5, 3), (5, 6), (4, 6), maps, its))

    def test_both_inputs_windowed(self):
        maps = ["(d0, d1) -> (d0 + d1)", "(d0, d1) -> (d0 + d1)", "(d0, d1) -> (d0)"]
        assert "both inputs" in self._why(_contraction((9,), (9,), (5,), maps, ["parallel", "reduction"]))


class TestTheCensusSeesWhatTheObserverDoesNot:
    def test_a_transposed_lhs_is_in_the_census_and_counted(self):
        # The shape observer skips a generic whose A does not carry K last; it was absent from the report.
        text = _contraction((40, 24), (40, 20), (24, 20), ["(d0, d1, d2) -> (d2, d0)", _KN, _MN], _PPR)
        got = IC.rewrite_contractions(parse_mlir_text(text), select=lambda _s: False, symbol_prefix=PREFIX)
        (r,) = got.not_routed
        assert r.macs == 24 * 20 * 40

    def test_the_kernel_view_of_a_rank4_contraction(self):
        maps = [
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)",
            "(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)",
        ]
        text = _contraction((2, 3, 10, 8), (2, 3, 12, 8), (2, 3, 10, 12), maps, ["parallel"] * 4 + ["reduction"])
        ((_op, shape),) = IC.routable_contractions(parse_mlir_text(text))
        assert shape.op == "linalg.batch_matmul" and shape.parallel == (6, 10, 12) and shape.reduction == (8,)


def test_the_site_is_read_from_provenance_only():
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(_contraction((24, 40), (40, 20), (24, 20), [_MK, _KN, _MN], _PPR))
    (op,) = [o for o in module.walk() if o.name == "linalg.generic"]
    assert cnf.site_of(op) == "unknown"
    op.attributes["prov.conv_path"] = StringAttr("im2col_matmul")
    assert cnf.site_of(op) == "conv"
