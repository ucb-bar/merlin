"""The default tile this target's recipe starts from, and the capacity model it shares with legality.

WHY THIS EXISTS. `choose_tiles` is the other half of the hand-written binding that `sched/AGENT.md`
directs be replaced, and like the legality checkers it was measured to be unfalsified: delete it and
nothing in the suite noticed, because the twelve pinned shapes exercise it only through a digest that
would change for any reason at all. A search whose only test is "the answer did not change" cannot be
replaced by a better search.

THE COUPLING IS THE POINT. `choose_tiles` searches the tiles that FIT, and the legality checker refuses
the tiles that do not. Those are the same constraint written twice, in two functions, from the same two
facts -- one as `ti * tj <= acc_rows_per_loop // dim`, the other as `I * J * dim > acc_rows_per_loop`.
Nothing held them together. If the search's model is looser than the checker's, the recipe emits
schedules its own binding rejects; if it is tighter, the recipe silently leaves capacity unused and
every measured ratio is against a handicapped baseline. Both directions are asserted here, behaviourally
-- the chosen tile is accepted, and one step past it in either capacity is refused.

WHAT IS NOT PINNED. Not the tiles themselves. The docstring is explicit that this "only picks a starting
point; measured alternatives replace it", so pinning the answers would make improving the search a
test-editing exercise. What is pinned is that the answer is ADMISSIBLE, that nothing admissible beats it
on the objective the function declares, and that the search refuses rather than guesses when nothing
fits.
"""

from __future__ import annotations

import importlib
from functools import cache

import pytest

from merlin.runtime.backends import base as _bk
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import TensorArg
from merlin.sched.isa import IsaError

pytestmark = pytest.mark.target("gemmini")

#: ResNet-50 matmul-shaped layers plus the degenerate ends -- a single block, a k of 1, an m of 1.
SHAPES = [
    (1, 1, 1),
    (16, 16, 16),
    (64, 64, 64),
    (100, 100, 100),
    (128, 512, 512),
    (512, 512, 4096),
    (1000, 1024, 1),
    (1, 1000, 2048),
    (3136, 64, 64),
    (3136, 64, 576),
    (196, 256, 1152),
    (196, 512, 256),
    (49, 512, 2304),
]


@cache
def _backend():
    return _bk.get_backend("gemmini")


@cache
def _sched():
    """The binding module itself; `choose_tiles` is not reachable through the backend's hook list."""
    return importlib.import_module(_backend().sched_matmul_reference.__module__.rsplit(".", 1)[0] + ".gemmini_sched")


@cache
def _facts():
    return _backend().sched_instruction_set().facts


def _caps() -> tuple[int, int]:
    """(operand, output) capacity in DIM blocks, as the LEGALITY CHECKER states them.

    Deliberately re-derived from the two capacity facts rather than read off `choose_tiles`, so this is
    a second opinion about the constraint and not a restatement of the first.
    """
    f = _facts()
    return f["spad_rows_per_loop"] // f["dim"], f["acc_rows_per_loop"] // f["dim"]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _kernel(m: int, n: int, k: int, tiles=None):
    operands = {
        "a": TensorArg("A", (m, k), "i8", "read"),
        "b": TensorArg("B", (k, n), "i8", "read"),
        "c": TensorArg("C", (m, n), "i8", "write"),
        "d": TensorArg("D", (n,), "i32", "read"),
    }
    return _backend().sched_matmul_reference(
        name="mm", m=m, n=n, k=k, operands=operands, relu=False, scale=1.0, tiles=tiles
    )


# -- the tile fits, and the recipe built from it is legal ---------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_the_chosen_tile_fits_the_capacities_legality_enforces(shape):
    spad_cap, acc_cap = _caps()
    ti, tj, tk = _sched().choose_tiles(*shape, _facts())
    assert min(ti, tj, tk) >= 1, f"a tile dimension of zero is not a tile: {(ti, tj, tk)}"
    assert ti * tj <= acc_cap, f"output tile {ti}x{tj} exceeds the {acc_cap}-block accumulator budget"
    assert (ti + tj) * tk <= spad_cap, f"operand tiles need {(ti + tj) * tk} blocks > {spad_cap}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_the_recipe_built_from_the_chosen_tile_is_legal(shape):
    """Arithmetic agreement is not the claim; the claim is that the binding accepts what its own search
    produced. This runs the target's declared semantics over every dynamic instance."""
    errors = check_kernel(_kernel(*shape), _backend().sched_instruction_set())
    assert not errors, f"{shape}: the recipe's own default tile is illegal on this target: {errors[:3]}"


# -- the search's capacity model is the checker's, in both directions ---------------------------------


#: A shape whose chosen tile is bound by ONE capacity only, with the step that crosses it and the
#: phrase the checker must use. Isolated deliberately: on a shape that saturates both, a step past the
#: tile trips whichever bound the checker happens to report first, and the test would pass without ever
#: exercising the other one. Measured -- (1000, 1024, 1) fills the accumulator with twelve blocks of
#: scratchpad to spare, and (16, 16, 65536) fills the scratchpad with one block in the accumulator.
_BINDING_CAPACITY = [
    ((1000, 1024, 1), "tj", "accumulator rows"),
    ((16, 16, 65536), "tk", "scratchpad rows"),
]


@pytest.mark.parametrize("shape,grow,cited", _BINDING_CAPACITY, ids=["accumulator", "scratchpad"])
def test_one_step_past_the_chosen_tile_crosses_the_capacity_that_bounds_it(shape, grow, cited):
    """The tile the search stops at is the largest legality admits, not a conservative guess.

    A search whose model is TIGHTER than the checker's leaves capacity unused and quietly handicaps
    every measurement taken against it -- which no correctness test can see, because a smaller tile is
    perfectly legal and passes every check. So the falsifier has to be on the other side: step one block
    past the chosen tile and the checker must refuse, naming the capacity that bounds this shape. If it
    accepts, the search stopped early.
    """
    ti, tj, tk = _sched().choose_tiles(*shape, _facts())
    bigger = {"tj": (ti, tj + 1, tk), "tk": (ti, tj, tk + 1)}[grow]
    iset = _backend().sched_instruction_set()
    assert not check_kernel(_kernel(*shape, tiles=(ti, tj, tk)), iset), (
        "the chosen tile is itself refused, so refusing the next one up proves nothing"
    )
    errors = check_kernel(_kernel(*shape, tiles=bigger), iset)
    assert any(cited in e for e in errors), (
        f"{shape}: tile {bigger} could have been used and the {cited.split()[0]} would still have held, "
        f"so the search stopped short of the capacity. Errors: {errors or 'none'}"
    )


@pytest.mark.parametrize("shape,grow,cited", _BINDING_CAPACITY, ids=["accumulator", "scratchpad"])
def test_each_shape_is_bounded_by_the_capacity_it_is_paired_with(shape, grow, cited):
    """The control. Without it the test above would pass for a shape stopped by the OTHER capacity, and
    one of the two models would go unexercised while the pair read as covering both."""
    spad_cap, acc_cap = _caps()
    ti, tj, tk = _sched().choose_tiles(*shape, _facts())
    bigger = {"tj": (ti, tj + 1, tk), "tk": (ti, tj, tk + 1)}[grow]
    over = {
        "accumulator rows": bigger[0] * bigger[1] > acc_cap and (bigger[0] + bigger[1]) * bigger[2] <= spad_cap,
        "scratchpad rows": (bigger[0] + bigger[1]) * bigger[2] > spad_cap and bigger[0] * bigger[1] <= acc_cap,
    }[cited]
    assert over, f"{shape} stepped to {bigger} does not cross the {cited} bound alone"


# -- nothing admissible beats the answer on the objective the function declares ------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_no_admissible_tile_scores_better_on_the_declared_objective(shape):
    """The search is exhaustive, so its answer must be optimal for the objective in its docstring:
    minimise re-read traffic ``|A|*J0 + |B|*I0``, then loop instructions, then prefer the larger tile.

    Re-enumerated here rather than re-implemented: the candidate set and the score are stated
    independently, and a search that skipped a region -- a `break` that should have been a `continue`,
    an off-by-one in a bound -- is caught by a better candidate existing.
    """
    m, n, k = shape
    f = _facts()
    dim, eb = f["dim"], f["elem_bytes"]
    spad_cap, acc_cap = _caps()
    ib, jb, kb = _ceil(m, dim), _ceil(n, dim), _ceil(k, dim)

    def score(ti, tj, tk):
        i0, j0, k0 = _ceil(ib, ti), _ceil(jb, tj), _ceil(kb, tk)
        return (m * k * eb * j0 + k * n * eb * i0, i0 * j0 * k0, -ti * tj)

    chosen = _sched().choose_tiles(m, n, k, f)
    best = score(*chosen)
    for ti in range(1, ib + 1):
        for tj in range(1, jb + 1):
            if ti * tj > acc_cap:
                continue
            for tk in range(1, kb + 1):
                if (ti + tj) * tk > spad_cap:
                    continue
                assert score(ti, tj, tk) >= best, (
                    f"{shape}: tile {(ti, tj, tk)} scores {score(ti, tj, tk)}, better than the chosen "
                    f"{chosen} at {best} -- the search missed it"
                )


def test_the_objective_discriminates_between_shapes():
    """The mutation. Every assertion above would pass for a function that always returned (1, 1, 1)."""
    answers = {s: _sched().choose_tiles(*s, _facts()) for s in SHAPES}
    assert len(set(answers.values())) > 1, f"the search returns one tile for every shape: {answers}"


# -- when nothing fits, it refuses ---------------------------------------------------------------------


def test_a_machine_too_small_for_any_tile_is_refused_not_rounded_down():
    """The error path, which the real facts cannot reach.

    With this target's capacities a 1x1x1 tile always fits, so the refusal is unreachable through
    `derive`-shaped facts and would read as dead code. It is not dead: `choose_tiles` takes facts as a
    parameter precisely so another target's can be passed, and a target whose per-loop operand budget
    cannot hold one block of A beside one of B has no tile at all. Rounding down to zero there would
    emit a loop over an empty tile.
    """
    tiny = dict(_facts()) | {"spad_rows_per_loop": _facts()["dim"], "acc_rows_per_loop": _facts()["dim"]}
    with pytest.raises(IsaError, match="no tile fits"):
        _sched().choose_tiles(64, 64, 64, tiny)


def test_a_shape_with_no_extent_has_no_tile():
    """A zero dimension yields an empty candidate set rather than a degenerate tile."""
    with pytest.raises(IsaError, match="no tile fits"):
        _sched().choose_tiles(0, 64, 64, _facts())
