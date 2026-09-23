"""The board matmul-routing matcher is ONE structural reader, and it fails closed.

Companion to test_xnnpack_board_backend.py / test_openblas_board_backend.py, which cover each
backend's own surface. This file covers the shared reader
(:mod:`merlin.runtime.backends._matmul_routing`) that replaced the three byte-identical copies of a
compiled ``linalg.matmul`` pattern:

  * the three backends share ONE routable set (three copies is three chances to drift);
  * it accepts everything the retired pattern accepted, PLUS the valid spellings that pattern
    silently dropped (nested attribute dict, nested ``<...>`` in a type, ``indexing_maps``,
    a ``-`` in an SSA name);
  * a matmul it cannot parse RAISES instead of quietly routing nothing — a silently unrouted op
    is indistinguishable downstream from "the backend produced nothing".
"""
from __future__ import annotations

import pytest

from merlin.common.paths import bench_dir
from merlin.runtime.backends import _matmul_routing as mr
from merlin.runtime.backends import openblas_board as ob
from merlin.runtime.backends import ours_board as our
from merlin.runtime.backends import xnnpack_board as xb

_CANON = (
    'module {\n'
    '  func.func @forward(%a: tensor<32x256xf32>, %b: tensor<256x128xf32>, '
    '%c: tensor<32x128xf32>) -> tensor<32x128xf32> {\n'
    '    %1 = linalg.matmul ins(%a, %b : tensor<32x256xf32>, tensor<256x128xf32>) '
    'outs(%c : tensor<32x128xf32>) -> tensor<32x128xf32>\n'
    '    return %1 : tensor<32x128xf32>\n'
    '  }\n}\n')


def _wrap(op_text: str) -> str:
    return ('module {\n  func.func @forward() {\n    ' + op_text + '\n    return\n  }\n}\n')


# --------------------------------------------------------------- one reader, three backends

def test_the_three_backends_share_one_routable_set():
    """Not "the same logic three times" -- literally the same callable, so they cannot drift."""
    assert xb._is_routable is ob._is_routable is our._is_routable is mr.is_routable


def test_the_three_backends_route_the_same_ops():
    counts = {
        "xnn": xb.rewrite_matmuls_to_xnn(_CANON)[1],
        "openblas": ob.rewrite_matmuls_to_openblas(_CANON)[1],
        "ours": our.rewrite_matmuls_to_ours(_CANON)[1],
    }
    assert set(counts.values()) == {1}, counts
    covs = {xb.matmul_routing_coverage(_CANON), ob.matmul_routing_coverage(_CANON),
            our.matmul_routing_coverage(_CANON)}
    assert covs == {(1, 1)}


# ------------------------------------------------ accepts everything the old pattern accepted

def test_canonical_spelling_still_routes():
    out, n = mr.rewrite_matmuls(_CANON, "SYM")
    assert n == 1
    assert "call @SYM_0(%a, %b, %c)" in out
    assert "linalg.matmul" not in out


def test_attribute_dictionary_before_ins_still_routes():
    """The old `(\\s*\\{[^}]*\\})?` alternative -- the spelling the model corpus actually uses."""
    t = _wrap('%1 = linalg.matmul {prov.op = "matmul", prov.fqn = "lm.q_proj"} '
              'ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>')
    assert mr.rewrite_matmuls(t, "SYM")[1] == 1


def test_multi_line_op_still_routes():
    t = _wrap('%1 = linalg.matmul\n        ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)\n'
              '        outs(%c : tensor<4x4xf32>)\n        -> tensor<4x4xf32>')
    assert mr.rewrite_matmuls(t, "SYM")[1] == 1


def test_bytes_outside_the_op_are_preserved():
    out, n = mr.rewrite_matmuls(_CANON, "SYM")
    assert n == 1
    # everything before the func and after the routed op survives verbatim
    assert out.startswith("module {\n")
    assert out.endswith("    return %1 : tensor<32x128xf32>\n  }\n}\n")


def test_no_matmul_is_byte_identical():
    t = "module {\n  func.func @f(%a: tensor<2xf32>) -> tensor<2xf32> { return %a : tensor<2xf32> }\n}\n"
    assert mr.rewrite_matmuls(t, "SYM") == (t, 0)


# --------------------------------- accepts MORE: the spellings the old pattern silently dropped

def test_nested_attribute_dictionary_is_no_longer_dropped():
    """`\\{[^}]*\\}` stopped at the first `}`, so a nested dict broke the whole match and the op
    was left unrouted with no complaint."""
    t = _wrap('%1 = linalg.matmul {prov = {op = "matmul", nested = {k = 1 : i64}}} '
              'ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>')
    assert mr.rewrite_matmuls(t, "SYM")[1] == 1


def test_indexing_maps_spelling_is_no_longer_dropped():
    """MLIR prints `indexing_maps = [...]` for a non-default `linalg.matmul`; the old pattern
    allowed only a `{...}` dict there, so such an op was silently skipped."""
    t = _wrap('%1 = linalg.matmul indexing_maps = ['
              'affine_map<(d0, d1, d2) -> (d0, d2)>, '
              'affine_map<(d0, d1, d2) -> (d2, d1)>, '
              'affine_map<(d0, d1, d2) -> (d0, d1)>] '
              'ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>')
    assert mr.rewrite_matmuls(t, "SYM")[1] == 1


def test_nested_angle_bracket_type_is_parsed_not_dropped():
    """`tensor<[^>]+>` cannot cross the inner `>` of `tensor<4x4xcomplex<f32>>`, so the op fell out
    of the match entirely. It is now a REPORTED candidate that simply is not f32-routable."""
    t = _wrap('%1 = linalg.matmul ins(%a, %b : tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) '
              'outs(%c : tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>')
    sites = mr.scan_matmuls(t)
    assert len(sites) == 1 and sites[0].op is not None
    assert sites[0].op.at == "4x4xcomplex<f32>"
    assert mr.routing_coverage(t) == (1, 0)      # a candidate, correctly ineligible
    assert mr.rewrite_matmuls(t, "SYM") == (t, 0)


def test_hyphenated_ssa_name_is_no_longer_truncated():
    """`%[\\w$.]+` stopped at the `-`, so `%a-0` failed the following `\\s*=` and the op was lost."""
    t = _wrap('%r-0 = linalg.matmul ins(%a-1, %b-2 : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c-3 : tensor<4x4xf32>) -> tensor<4x4xf32>')
    out, n = mr.rewrite_matmuls(t, "SYM")
    assert n == 1
    assert "%r-0 = call @SYM_0(%a-1, %b-2, %c-3)" in out


def test_matmul_text_inside_a_string_attribute_is_not_rewritten():
    """A rewrite that fired inside a string attribute would corrupt the module. The old pattern had
    no string awareness at all."""
    t = _wrap('%1 = "some.op"() {note = "%z = linalg.matmul ins(%a, %b : tensor<4x4xf32>, '
              'tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>"} : () -> i32')
    assert mr.scan_matmuls(t) == []
    assert mr.rewrite_matmuls(t, "SYM") == (t, 0)


# ------------------------------------------------------------------------------- fail closed

def test_unparseable_matmul_raises_instead_of_routing_nothing():
    """The whole point: an op the reader cannot read must be an ERROR, not a silent zero."""
    t = _wrap('%1 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              '-> tensor<4x4xf32>')          # no outs(...)
    with pytest.raises(mr.MatmulRoutingError) as exc:
        mr.rewrite_matmuls(t, "SYM")
    assert "outs" in str(exc.value)
    with pytest.raises(mr.MatmulRoutingError):
        mr.routing_coverage(t)


def test_multi_result_matmul_raises():
    t = _wrap('%1:2 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>')
    with pytest.raises(mr.MatmulRoutingError):
        mr.rewrite_matmuls(t, "SYM")


def test_routed_calls_without_a_func_anchor_raise():
    """A rewritten module with no `func.func` to hang the private decls on would call undeclared
    symbols. The retired code returned that text; we refuse it."""
    t = ('%1 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
         'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>\n')
    with pytest.raises(mr.MatmulRoutingError) as exc:
        mr.rewrite_matmuls(t, "SYM")
    assert "func.func" in str(exc.value)


def test_memref_form_is_reported_not_silently_ignored():
    """The destination-passing memref form has no tensor result to bind to a `func.call`, so it is
    outside the rewrite domain -- but it is REPORTED with a reason, not dropped without a trace."""
    t = _wrap('linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>) '
              'outs(%c : memref<4x4xf32>)')
    sites = mr.scan_matmuls(t)
    assert len(sites) == 1
    assert sites[0].op is None and "no result binding" in sites[0].reason
    assert mr.routing_coverage(t) == (0, 0)      # not a candidate -> not in the denominator
    assert mr.rewrite_matmuls(t, "SYM") == (t, 0)


def test_linalg_matmul_transpose_b_is_a_different_op():
    """`\\b` after `linalg.matmul` excluded the `_transpose_b` variant; the token boundary keeps
    that exclusion (its operand order is not the routed kernel's)."""
    t = _wrap('%1 = linalg.matmul_transpose_b ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) '
              'outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>')
    assert mr.scan_matmuls(t) == []
    assert mr.rewrite_matmuls(t, "SYM") == (t, 0)


# ------------------------------------------------------------------- against real model MLIR

def test_real_recapture_model_routes_every_matmul_token():
    """A tracked whole-model lowering: every `linalg.matmul` token in it is accounted for as a
    parsed candidate, and the f32 ones are routed."""
    model = bench_dir() / "dse_guidance" / "recaptures_loop" / "tiny_llama" / "model.mlir"
    if not model.is_file():
        pytest.skip(f"recapture corpus not present at {model}")
    text = model.read_text()
    tokens = text.count("linalg.matmul")
    candidates, eligible = mr.routing_coverage(text)
    assert candidates == tokens == 30       # nothing dropped between token and candidate
    assert eligible == 30
    out, n = mr.rewrite_matmuls(text, "SYM")
    assert n == eligible
    assert "linalg.matmul" not in out
