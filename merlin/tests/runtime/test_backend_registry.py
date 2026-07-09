"""The runtime backend registry classifies backends by target CLASS (CPU/GPU/NPU), not instance."""
from __future__ import annotations

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendKind, TargetClass


def test_registry_taxonomy():
    assert set(base.list_backends()) == {"spike", "saturn_vec", "gemmini", "muon",
                                         "spike_model", "zephyr_model"}
    # class taxonomy: address backends by CPU/GPU/NPU, not by silicon instance
    assert base.class_of("gemmini") is TargetClass.NPU
    assert base.class_of("muon") is TargetClass.GPU
    assert base.class_of("spike") is TargetClass.CPU
    assert set(base.backends_of_class(TargetClass.CPU)) == {"spike", "saturn_vec",
                                                            "spike_model", "zephyr_model"}
    assert base.backends_of_class(TargetClass.NPU) == ["gemmini"]
    assert base.backends_of_class(TargetClass.GPU) == ["muon"]


def test_backend_kinds():
    assert base.info("gemmini").kind is BackendKind.KERNEL
    assert base.info("zephyr_model").kind is BackendKind.WHOLE_MODEL


def test_get_backend_lazy_import():
    # the registry resolves to the real module (spike has no heavy import-time deps)
    spike = base.get_backend("spike")
    assert hasattr(spike, "run_command_buffer") and hasattr(spike, "available")
