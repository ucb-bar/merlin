"""A host-lane element budget must charge only elements the emitter EVALUATES.

`estimate_cost` refuses a whole model before emission, so an overcharge refuses models the backend
could compile. It charged four op classes for work no emitter performs, and charged every op by the
widest tensor it TOUCHED rather than what it writes -- so an op merely reading a large constant
weight was billed that weight's whole extent. On a 22-layer decoder that read 3,216,234,988 elements
against a 400,000 budget; charging only evaluated elements reads 43,562,132.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

_PKG = repo_root() / "out/artifacts/perf-bench/gemmini" / "development_automatic_loop_ws_resadd_20260908/compiler"
pytestmark = pytest.mark.skipif(
    not (_PKG / "mlir_oot/codegen/host_linalg.py").is_file(),
    reason="generated gemmini package is not present in this checkout",
)


def _host_linalg():
    # Imported as part of its package: the module uses relative imports
    # (`from ..lowering.plan import ...`), so loading the file standalone raises
    # "attempted relative import with no known parent package".
    if str(_PKG) not in sys.path:
        sys.path.insert(0, str(_PKG))
    for stale in [n for n in sys.modules if n == "mlir_oot" or n.startswith("mlir_oot.")]:
        del sys.modules[stale]
    return importlib.import_module("mlir_oot.codegen.host_linalg")


HL = None


def setup_module(_module) -> None:
    global HL
    HL = _host_linalg()


def _eager_transpose(elems, src_shape, perm):
    """What the materialising implementation did: walk the RESULT space and read the source."""
    import itertools

    out_shape = [src_shape[p] for p in perm]
    strides = HL._strides(tuple(src_shape))
    out = []
    for idx in itertools.product(*[range(b) for b in out_shape]):
        src_idx = [0] * len(perm)
        for d, p in enumerate(perm):
            src_idx[p] = idx[d]
        out.append(elems[sum(i * s for i, s in zip(src_idx, strides))])
    return out, tuple(out_shape)


@pytest.mark.parametrize(
    "shape,perm",
    [
        ((2, 3), [1, 0]),
        ((3, 2), [1, 0]),
        ((2, 3, 4), [2, 0, 1]),
        ((2, 3, 4), [1, 2, 0]),
        ((4, 1, 5), [0, 2, 1]),
        ((2, 2, 2, 2), [3, 1, 0, 2]),
        ((7,), [0]),
        ((1, 6), [1, 0]),
    ],
)
def test_lazy_transpose_view_matches_materialising_it(shape, perm) -> None:
    n = 1
    for d in shape:
        n *= d
    elems = [f"v{i}" for i in range(n)]
    want, out_shape = _eager_transpose(elems, list(shape), perm)

    view = HL._TransposedElems(elems, out_shape, perm)

    assert len(view) == len(want)
    assert list(view) == want, "a permuted view that disagrees with the eager form miscompiles"
    assert [view[i] for i in range(len(view))] == want


def test_repeat_view_is_one_value_over_a_shape() -> None:
    view = HL._RepeatedElems("z", 5)

    assert len(view) == 5
    assert list(view) == ["z"] * 5
    assert view[0] == "z" and view[4] == "z" and view[-1] == "z"
    with pytest.raises(IndexError):
        view[5]


def test_layout_only_ops_are_free_and_agree_with_the_handlers() -> None:
    # Every name here must be handled by a view in the emitter, or the estimate admits a program
    # the emitter cannot finish. The reverse (charged here, free there) refuses compilable models.
    assert HL._LAYOUT_ONLY_OPS == frozenset(
        {
            "linalg.transpose",
            "tensor.expand_shape",
            "tensor.collapse_shape",
            "tensor.reshape",
            "tensor.empty",
            "tensor.splat",
        }
    )


class _Ty:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _Val:
    def __init__(self, shape):
        self.type = _Ty(shape)


class _Op:
    def __init__(self, name, results, operands):
        self.name = name
        self.results = [_Val(s) for s in results]
        self.operands = [_Val(s) for s in operands]


def test_cost_charges_results_not_operands(monkeypatch) -> None:
    monkeypatch.setattr(HL, "TensorType", _Ty)
    # A generic writing 256,000 elements while READING a 32000x2048 weight: the reader pays for
    # what it writes. Charging the operand billed 65,536,000, a 256x overcharge.
    reader = _Op("linalg.generic", [(8, 32000)], [(32000, 2048)])
    assert HL.estimate_cost([reader]) == 8 * 32000

    # ...and the layout-only ops contribute nothing at all, whatever their extents.
    free = [_Op(name, [(32000, 2048)], [(32000, 2048)]) for name in sorted(HL._LAYOUT_ONLY_OPS)]
    assert HL.estimate_cost(free) == 0
    assert HL.estimate_cost([reader, *free]) == 8 * 32000
