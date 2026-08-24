"""A layer too big for the on-chip working set must be BLOCKED, not declined.

Measured on lstmnetvit/gemmini: K and N are each fine alone and together they are not, so two of 37
matmul layers fell back to the host and the whole model failed its must_accelerate gate at 35/37. The
tiler that fixes this already existed and was wired into the tile CERTIFICATION path only -- so the tile
record reported "runs at this shape" about a shape the model itself never got to run.
"""
from __future__ import annotations

import pytest

from merlin.compile_cli import _capacity_fit_tile, _operand_store_capacity_elems

# (M, K, N, does the whole layer fit on chip?) — measured against the real gemmini oracle: every shape
# marked True ran, every shape marked False was declined by the mesh.
_OBSERVED = [(1, 16, 16, True), (1, 16, 512, True), (1, 512, 16, True),
             (1, 512, 512, False), (1, 528, 512, False), (1, 4608, 512, False)]


def _cap():
    cap = _operand_store_capacity_elems("gemmini", "int8")
    if not cap:
        pytest.skip("gemmini RTL facts declare no scratchpad capacity")
    return cap


@pytest.mark.parametrize("m,k,n,fits", _OBSERVED)
def test_the_capacity_model_predicts_what_the_mesh_declined(m, k, n, fits):
    """The derived capacity must explain the observed declines, or the blocking decision is a guess."""
    mt, kt, nt, _n = _capacity_fit_tile(m, k, n, 16, _cap())
    assert ((mt, kt, nt) == (m, k, n)) is fits, \
        f"{m}x{k}x{n}: capacity model says fits={(mt, kt, nt) == (m, k, n)}, mesh said fits={fits}"


def test_a_blocked_layer_covers_every_element_exactly_once():
    """The blocking loop's tile walk must tile the output, with no gap and no double-add -- a double-add
    on a K split is silent (it just inflates the sum) and would look like a numeric defect."""
    cap = _cap()
    M, K, N = 4, 4608, 512
    mt, kt, nt, _ = _capacity_fit_tile(M, K, N, 16, cap)
    assert (mt, kt, nt) != (M, K, N), "this shape must actually need blocking for the test to mean anything"
    seen: dict[tuple[int, int], int] = {}
    for m0 in range(0, M, mt):
        for n0 in range(0, N, nt):
            for k0 in range(0, K, kt):
                for i in range(m0, min(m0 + mt, M)):
                    for j in range(n0, min(n0 + nt, N)):
                        seen[(i, j)] = seen.get((i, j), 0) + 1
    k_blocks = -(-K // kt)
    assert len(seen) == M * N, "every output element must be produced"
    assert set(seen.values()) == {k_blocks}, \
        f"each element must accumulate exactly {k_blocks} K-blocks, saw {sorted(set(seen.values()))}"


def test_an_accumulator_epilogue_is_not_split_across_k():
    """An acc_scale requant must see the whole accumulation. Splitting K under an epilogue would requant
    each partial sum, which is silently wrong rather than merely imprecise."""
    import inspect

    from merlin import compile_cli
    src = inspect.getsource(compile_cli.run_matmul_on_mesh)
    assert "if epilogue:" in src.split("_capacity_fit_tile")[-1], \
        "the blocking path must refuse an epilogue rather than split it"
    assert "cannot be split across K blocks" in src, "and must say why it refused"
