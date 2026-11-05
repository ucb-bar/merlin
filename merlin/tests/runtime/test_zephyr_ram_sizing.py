"""RAM-region sizing for a Zephyr image: does the region provision the model's real working set?

The region holds the linked image -- weights, code, stacks and the static activation arena -- and the
heap Zephyr's malloc claims from what is left. These pin the region rule (`zephyr_memory.region_for`):
a model whose working set dwarfs its weights gets a region for it, and a small one stays at the port's
stock region, which is the only size FireSim boots a whole-model image at reliably.
"""

from merlin.common.mlir_query import activation_peak_bytes
from merlin.common.paths import repo_root
from merlin.runtime.backends import zephyr_memory as zmem

MB = 1024 * 1024
DEFAULT_RAM_BYTES = 256 * MB  # selected example port's device-tree default


def _budget(dram=4096 * MB, link_limit=None):
    return zmem.MemoryBudget(board="b", dram_bytes=dram, policy="static-arena", weights_bytes=0,
                             link_limit_bytes=link_limit)  # fmt: skip


def test_a_big_working_set_grows_the_region_past_the_default():
    """whisper_tiny int8: 117 MB of weights and a 210 MB working set need more than the stock region."""
    need = 117 * MB + 210 * MB
    region = zmem.region_for(_budget(), need, DEFAULT_RAM_BYTES)
    assert region >= need and region > DEFAULT_RAM_BYTES


def test_a_small_working_set_leaves_the_default_region_alone():
    """Models that fit the stock region must stay at it: growing it for a model that does not need it is
    a regression on FireSim, not a safety margin."""
    for need_mb in (14, 19, 15, 255):
        assert zmem.region_for(_budget(), need_mb * MB, DEFAULT_RAM_BYTES) == DEFAULT_RAM_BYTES


def test_the_region_is_rounded_to_its_granule_but_never_past_what_the_board_can_link():
    """Rounding keeps the overlay stable across images a few KiB apart; it must not turn an image that
    fits the board into one that does not."""
    granule = zmem.REGION_GRANULE_BYTES
    assert zmem.region_for(_budget(), 300 * MB + 1, DEFAULT_RAM_BYTES) == 300 * MB + granule - (300 * MB % granule)
    tight = _budget(dram=511 * MB)
    assert zmem.region_for(tight, 510 * MB, DEFAULT_RAM_BYTES) == 511 * MB


def test_peak_is_measured_from_a_real_bundle_when_one_is_present():
    """Guards the accessor against IR drift: a captured model must yield a positive peak."""
    import pytest

    bundle = repo_root() / "out/artifacts/recaptures/deepjscc_int8_full/model.mlir"
    if not bundle.is_file():
        pytest.skip("deepjscc bundle not captured in this clone")
    peak = activation_peak_bytes(bundle)
    assert peak is not None and peak > 0
    # arguments (weights/inputs) are NOT activations: deepjscc's are under 1 MB, its live set ~13 MB
    assert peak > MB


def test_a_module_without_the_named_function_is_unmeasurable_not_an_error():
    assert activation_peak_bytes("module { }", func_name="forward") is None
