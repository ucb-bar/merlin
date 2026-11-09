"""RAM-region sizing for a Zephyr image: does it provision the model's real working set?

The region has to hold the weights blob (linked into ``.data``) AND the activation arena Zephyr's
malloc claims from the leftover. Sizing the headroom from the weight bytes alone assumes activations
are small relative to weights, which is false for an encoder whose attention matrices dwarf its
parameters — the case that motivated measuring the peak instead of assuming it.
"""

from merlin.common.mlir_query import activation_peak_bytes
from merlin.common.paths import repo_root
from merlin.compiler.build.zephyr.memory import ram_for_weights

MB = 1024 * 1024
DEFAULT_RAM_BYTES = 256 * MB  # selected example port's device-tree default


def _sized(weights, peak=None):
    return ram_for_weights(weights, peak, default_ram_bytes=DEFAULT_RAM_BYTES)


def test_measured_activations_never_shrink_the_region():
    """Passing a peak may only GROW the region: no image that boots today may get less RAM."""
    for weights_mb in (1, 16, 128, 512, 2048):
        w = weights_mb * MB
        base = _sized(w)
        for peak_mb in (0, 1, 64, 256, 4096):
            assert _sized(w, peak_mb * MB) >= base


def test_a_big_working_set_grows_the_region_past_the_weight_scaled_guess():
    """116 MB of weights + a 210 MB working set needs more than the weight-scaled formula gives.

    Measured (whisper_tiny int8): the weight-scaled region is 288 MB, the image takes ~125 MB of it,
    leaving a 163 MB arena for a 210 MB peak — an allocation failure on a board with enough DRAM.
    """
    w, peak = 117 * MB, 210 * MB
    assert _sized(w) < w + peak
    assert _sized(w, peak) >= w + peak


def test_a_small_working_set_leaves_the_default_region_alone():
    """Models whose activations fit the existing headroom must stay at the 256 MB default.

    FireSim only boots the whole-model image reliably at the stock ``ram0`` size, so growing the
    region for a model that does not need it is a regression, not a safety margin.
    """
    for weights_mb, peak_mb in ((1, 13), (7, 12), (11, 4)):
        assert _sized(weights_mb * MB, peak_mb * MB) == DEFAULT_RAM_BYTES


def test_an_unmeasurable_peak_falls_back_to_the_weight_scaled_size():
    """``activation_peak_bytes`` returns None on IR it cannot measure; sizing must not break."""
    w = 64 * MB
    assert _sized(w, None) == _sized(w)


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
