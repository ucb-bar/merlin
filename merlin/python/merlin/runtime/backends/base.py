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
import pkgutil
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
    MATMUL_ROUTE = "matmul_route"  # routes matmuls to an external/hand GEMM for attribution
                                   # (xnnpack/openblas/ours on a board; xnnpack on the host)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    target_class: TargetClass
    kind: BackendKind
    module: str                  # dotted import path, loaded lazily via get_backend()


# The registry is populated two ways so the CORE never hardcodes a shipped-accelerator name:
#  * the generic ISA-class instances (RVV/host CPU kernels, whole-model runners, matmul-route
#    attribution) are seeded here directly — their identifiers are ISA-class/tool names, not silicon
#    instances, so naming them keeps the core target-agnostic;
#  * the silicon-specific reference instances (the NPU/GPU accelerator backends) SELF-REGISTER from
#    their own module via ``register(...)`` and are discovered lazily by ``_ensure_discovered()`` — so
#    a new accelerator backend is one ``register`` line in ITS OWN module (or an out-of-tree package
#    reached via ``MERLIN_TARGET_PATH``), and the core carries no name -> module map for it.
_REGISTRY: dict[str, BackendInfo] = {
    "spike":        BackendInfo("spike", TargetClass.CPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.spike"),
    "saturn_vec":   BackendInfo("saturn_vec", TargetClass.CPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.saturn_vec"),
    "spike_model":  BackendInfo("spike_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.spike_model"),
    "zephyr_model": BackendInfo("zephyr_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.zephyr_model"),
    # matmul-routing attribution backends (CPU-class): route routable matmuls to an external/hand
    # GEMM — the RVV board variants (xnnpack/openblas/ours) + the x86 host xnnpack reference.
    "xnnpack_board":  BackendInfo("xnnpack_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.xnnpack_board"),
    "openblas_board": BackendInfo("openblas_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.openblas_board"),
    "ours_board":     BackendInfo("ours_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.ours_board"),
    "xnnpack_host":   BackendInfo("xnnpack_host", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.xnnpack_host"),
}

# Names seeded above (the generic ISA-class instances). Discovery imports every OTHER submodule of this
# package so any accelerator backend that self-registers is picked up — without the core naming it.
_SEEDED: frozenset[str] = frozenset(_REGISTRY)
_discovered = False


def register(info: BackendInfo) -> None:
    """Register a backend (idempotent per name). A reference accelerator backend calls this from its
    OWN module at import time, so the core never carries a name -> module map for a specific silicon."""
    _REGISTRY[info.name] = info


def _ensure_discovered() -> None:
    """Import the not-yet-seeded submodules of this package so their ``register(...)`` calls run. Guarded
    per module: a backend whose optional deps are missing simply fails to register instead of breaking
    the registry. Runs at most once (idempotent), on first registry query — never at ``import base``,
    so importing the registry does NOT eagerly pull in the accelerator backends."""
    global _discovered
    if _discovered:
        return
    _discovered = True  # set first: a backend importing base during its own load must not re-enter
    pkg = importlib.import_module(__name__.rsplit(".", 1)[0])
    for mod in pkgutil.iter_modules(pkg.__path__):
        leaf = mod.name
        if leaf.startswith("_") or leaf == __name__.rsplit(".", 1)[1] or leaf in _SEEDED:
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{leaf}")
        except Exception:  # noqa: BLE001 — a backend with missing optional deps just does not register
            continue


@runtime_checkable
class Backend(Protocol):
    """The shape every kernel backend module exposes (module-level functions)."""

    def available(self) -> bool: ...
    def compile_command_buffer(self, cb: dict[str, Any], workdir: Any, **kw: Any) -> Any: ...
    def run_elf(self, elf: Any, **kw: Any) -> str: ...
    def parse_output(self, text: str) -> tuple[dict, dict]: ...
    def run_command_buffer(self, cb: dict[str, Any], **kw: Any) -> dict: ...


def list_backends() -> list[str]:
    _ensure_discovered()
    return sorted(_REGISTRY)


def info(name: str) -> BackendInfo:
    _ensure_discovered()
    return _REGISTRY[name]


def class_of(name: str) -> TargetClass:
    _ensure_discovered()
    return _REGISTRY[name].target_class


def backends_of_class(target_class: TargetClass) -> list[str]:
    _ensure_discovered()
    return sorted(n for n, b in _REGISTRY.items() if b.target_class == target_class)


def get_backend(name: str):
    """Lazily import + return the backend module for ``name`` (raises KeyError if unregistered)."""
    _ensure_discovered()
    return importlib.import_module(_REGISTRY[name].module)


# --- shared backend plumbing (the copy-pasted console protocol, collapsed) --------------------------
def _strip_warning_fragments(text: str) -> str:
    """Drop each line's ``%Warning:``/``Warning:`` fragment onward (stray Verilator noise), keeping
    any text before it — the structural equivalent of the old ``%?Warning:[^\\n]*`` substitution."""
    out = []
    for line in text.split("\n"):
        idx = line.find("Warning:")
        if idx != -1:
            if idx > 0 and line[idx - 1] == "%":
                idx -= 1
            line = line[:idx]
        out.append(line)
    return "\n".join(out)


def parse_console(text: str, *, error_cls: type[Exception] = RuntimeError,
                  strip_warnings: bool = False, tolerant_metric: bool = False,
                  value_parser=int) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the shared ``OUT``/``METRIC``/``DONE`` backend console protocol into (outputs, raw_metrics).

    Every backend prints results the same way — ``OUT <name> <rows> <cols> v...`` /
    ``METRIC <name> <int>`` / ``DONE`` — so the parser is shared; the small per-backend variations are
    flags: ``strip_warnings`` drops Verilator ``%Warning:`` fragments (gemmini/verilator),
    ``tolerant_metric`` skips malformed METRIC lines instead of raising (gemmini), ``value_parser`` is
    ``int`` for int8/systolic/CPU targets and ``float`` for fp SIMT targets. ``error_cls`` is the
    backend's own exception type (so messages/raises are unchanged from the hand-written versions)."""
    if strip_warnings:
        text = _strip_warning_fragments(text)
    outputs: dict[str, list] = {}
    raw: dict[str, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "OUT":
            name, rows, cols = parts[1], int(parts[2]), int(parts[3])
            vals = [value_parser(v) for v in parts[4:]]
            if len(vals) != rows * cols:
                raise error_cls(f"OUT {name}: expected {rows * cols} values, got {len(vals)}")
            outputs[name] = [vals[r * cols:(r + 1) * cols] for r in range(rows)]
        elif parts[0] == "METRIC":
            if tolerant_metric:
                try:
                    raw[parts[1]] = int(parts[2])
                except (IndexError, ValueError):
                    pass
            else:
                raw[parts[1]] = int(parts[2])
        elif parts[0] == "DONE":
            done = True
    if not done:
        raise error_cls(f"run did not reach DONE; output was:\n{text[:2000]}")
    return outputs, raw
