"""The runtime backend registry classifies backends by target CLASS (CPU/GPU/NPU), not instance."""
from __future__ import annotations

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendKind, TargetClass


#: The backends this repo ships. An OUT-OF-TREE package may register more via `plugin.backend`, which
#: is the seam working as designed — so these assertions pin the in-tree set and require anything extra
#: to actually be out-of-tree, rather than pinning a global total that any discovery breaks. Asserting
#: equality made this test fail for a target package that had merely been loaded earlier in the run,
#: which reports as "the taxonomy is wrong" when nothing about the taxonomy changed.
IN_TREE = {"spike", "saturn_vec", "gemmini", "muon", "spike_model", "zephyr_model",
           "xnnpack_board", "openblas_board", "ours_board", "xnnpack_host"}


def _in_tree(names):
    """Registered names implemented inside this repo (OOT packages load under merlin._oot_backends)."""
    return {n for n in names if base.info(n).module.startswith("merlin.runtime.backends.")}


def test_registry_taxonomy():
    registered = set(base.list_backends())
    assert IN_TREE <= registered, "an in-tree backend stopped registering"
    assert _in_tree(registered) == IN_TREE, "an unexpected in-tree backend appeared"
    # class taxonomy: address backends by CPU/GPU/NPU, not by silicon instance
    assert base.class_of("gemmini") is TargetClass.NPU
    assert base.class_of("muon") is TargetClass.GPU
    assert base.class_of("spike") is TargetClass.CPU
    # NPU=gemmini, GPU=muon; everything else (RVV/host CPU kernels, whole-model, matmul-route) is CPU
    assert _in_tree(base.backends_of_class(TargetClass.NPU)) == {"gemmini"}
    assert _in_tree(base.backends_of_class(TargetClass.GPU)) == {"muon"}
    assert _in_tree(base.backends_of_class(TargetClass.CPU)) == {
        "spike", "saturn_vec", "spike_model", "zephyr_model",
        "xnnpack_board", "openblas_board", "ours_board", "xnnpack_host"}


def test_backend_kinds():
    assert base.info("gemmini").kind is BackendKind.KERNEL
    assert base.info("zephyr_model").kind is BackendKind.WHOLE_MODEL
    assert base.info("xnnpack_board").kind is BackendKind.MATMUL_ROUTE
    assert _in_tree(b for b in base.list_backends()
                    if base.info(b).kind is BackendKind.MATMUL_ROUTE) == {
        "xnnpack_board", "openblas_board", "ours_board", "xnnpack_host"}


def test_get_backend_lazy_import():
    # the registry resolves to the real module (spike has no heavy import-time deps)
    spike = base.get_backend("spike")
    assert hasattr(spike, "run_command_buffer") and hasattr(spike, "available")


def test_parse_console_shared_protocol():
    # the OUT/METRIC/DONE parser shared by the backends (spike/gemmini delegate to it)
    outs, raw = base.parse_console("OUT Y0 2 2 1 2 3 4\nMETRIC cycles 100\nDONE\n")
    assert outs == {"Y0": [[1, 2], [3, 4]]} and raw == {"cycles": 100}
    # strip_warnings drops Verilator fragments; tolerant_metric skips malformed METRIC (gemmini flags)
    outs, raw = base.parse_console("OUT Y0 1 1 7\n%Warning: junk\nMETRIC broken\nMETRIC cycles 5\nDONE\n",
                                   strip_warnings=True, tolerant_metric=True)
    assert outs == {"Y0": [[7]]} and raw == {"cycles": 5}
    # error_cls + DONE requirement + length check
    import pytest
    with pytest.raises(ValueError):
        base.parse_console("OUT Y0 1 1 5\n", error_cls=ValueError)  # no DONE
    with pytest.raises(ValueError):
        base.parse_console("OUT Y0 2 2 1 2\nDONE\n", error_cls=ValueError)  # wrong value count
    # value_parser=float for fp SIMT targets
    outs, _ = base.parse_console("OUT Y 1 1 1.5\nDONE\n", value_parser=float)
    assert outs == {"Y": [[1.5]]}


def test_spike_gemmini_delegate_to_parse_console():
    from merlin.runtime.backends import base as _bk, spike
    gemmini = _bk.get_backend("gemmini")
    t = "OUT Y0 1 2 3 4\nMETRIC cycles 9\nDONE\n"
    assert spike.parse_output(t) == ({"Y0": [[3, 4]]}, {"cycles": 9})
    assert gemmini.parse_output(t) == ({"Y0": [[3, 4]]}, {"cycles": 9})
