"""Backend taxonomy + registry — address runtime backends by target CLASS, not instance.

The runtime backends are per-instance modules (``spike``, ``saturn_vec``, ``gemmini``, ``muon``,
``spike_model``, ``zephyr_model``) that share one shape — a ``Backend``: resolve toolchain →
``compile_command_buffer`` (→ ELF) → ``run_elf`` → ``parse_output`` (→ outputs+metrics) →
``run_command_buffer`` (compile+run+parse, gated on the reference oracle). Historically each was
imported by name; this module classifies them by **target class** (CPU / GPU / NPU) so callers and
tooling can reason about "the CPU/RVV backend" or "the NPU/systolic backend" rather than a specific
silicon instance — the same instance→class generalization the dialect layer got via
``xdsl_dialects.targets.factory``.

Scope (step 1): the taxonomy + a registry (name → module + class) + the shared ``Backend`` Protocol.
The per-instance modules keep their current behavior; collapsing their copy-pasted plumbing
(toolchain resolve / ``OUT/METRIC/DONE`` parse / reference gate) into a shared base is the follow-up
(and must re-certify the frozen gemmini path byte-for-byte).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class TargetClass(str, Enum):
    """The hardware class a backend targets (not the specific silicon instance)."""

    CPU = "cpu"     # scalar/vector CPU — RVV baremetal + whole-model (spike, saturn_vec, *_model)
    GPU = "gpu"     # SIMT — muon
    NPU = "npu"     # systolic-array / tensor accelerator — gemmini


class BackendKind(str, Enum):
    KERNEL = "kernel"          # compiles+runs one command buffer (spike, gemmini, muon, saturn_vec)
    WHOLE_MODEL = "whole_model"  # runs a whole captured model (spike_model, zephyr_model)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    target_class: TargetClass
    kind: BackendKind
    module: str                  # dotted import path, loaded lazily via get_backend()


# The registry lives here (central) rather than as a constant edited into each backend module, so a
# new backend is one line and the class taxonomy is readable in one place.
_REGISTRY: dict[str, BackendInfo] = {
    "spike":        BackendInfo("spike", TargetClass.CPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.spike"),
    "saturn_vec":   BackendInfo("saturn_vec", TargetClass.CPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.saturn_vec"),
    "gemmini":      BackendInfo("gemmini", TargetClass.NPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.gemmini"),
    "muon":         BackendInfo("muon", TargetClass.GPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.muon"),
    "spike_model":  BackendInfo("spike_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.spike_model"),
    "zephyr_model": BackendInfo("zephyr_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.zephyr_model"),
}


@runtime_checkable
class Backend(Protocol):
    """The shape every kernel backend module exposes (module-level functions)."""

    def available(self) -> bool: ...
    def compile_command_buffer(self, cb: dict[str, Any], workdir: Any, **kw: Any) -> Any: ...
    def run_elf(self, elf: Any, **kw: Any) -> str: ...
    def parse_output(self, text: str) -> tuple[dict, dict]: ...
    def run_command_buffer(self, cb: dict[str, Any], **kw: Any) -> dict: ...


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def info(name: str) -> BackendInfo:
    return _REGISTRY[name]


def class_of(name: str) -> TargetClass:
    return _REGISTRY[name].target_class


def backends_of_class(target_class: TargetClass) -> list[str]:
    return sorted(n for n, b in _REGISTRY.items() if b.target_class == target_class)


def get_backend(name: str):
    """Lazily import + return the backend module for ``name`` (raises KeyError if unregistered)."""
    return importlib.import_module(_REGISTRY[name].module)
