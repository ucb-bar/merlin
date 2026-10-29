"""Host policy/producer regressions; no inference of Lean or native acceptance."""

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.verify.compiler_transforms import ALLOCATOR, Options


@pytest.mark.parametrize("tile", [2, 16, 32, 64, 128, 4096, 65536])
def test_private_tile_policy_accepts_ordinary_and_comfortably_large_sizes(tile):
    assert Options(buffer_allocator=ALLOCATOR, tile_size=tile).tile_size == tile


def test_private_tile_policy_boundary_is_explicit_and_separate_from_other_loop_profiles():
    with pytest.raises(ValueError, match="resource policy"):
        Options(buffer_allocator=ALLOCATOR, tile_size=65537)
    assert Options(tile_size=65537).tile_size == 65537
    assert Options(buffer_allocator=ALLOCATOR).tile_size is None


@pytest.mark.parametrize("tile", [True, False, 0, 1, 2.0, "128"])
def test_invalid_private_tile_types_and_small_values_stay_refused(tile):
    with pytest.raises(ValueError):
        Options(buffer_allocator=ALLOCATOR, tile_size=tile)


@pytest.mark.parametrize("tile", [32, 65536])
def test_larger_policy_retains_actual_producer_signed_index64_checks(tile):
    maximum = (1 << 63) - 1

    def module(upper):
        return parse_ir(f"""builtin.module {{func.func @run() {{
          %lower=arith.constant 0:index %upper=arith.constant {upper}:index %step=arith.constant 1:index
          scf.for %i=%lower to %upper step %step {{scf.yield}}
          func.return
        }}}}""")

    safe = module(maximum - tile)
    assert len(segment_loops(safe, (tile,), index_bits=64)) == 1
    unsafe = module(maximum - tile + 1)
    before = ir_text(unsafe)
    with pytest.raises(ValueError, match="signed index width"):
        segment_loops(unsafe, (tile,), index_bits=64)
    assert ir_text(unsafe) == before
