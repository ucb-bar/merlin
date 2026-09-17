"""Plumbing for running a layer through a target's out-of-tree backend package.

The per-process package build cache, simulator selection for the OOT certification path, the
diagnostic record of why the last mesh attempt declined (``_MESH_REFUSAL``), and the content-addressed
ids that give each mesh layer and invocation its own artifact directory.
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

_MESH_REFUSAL: dict = {}  # diagnostic only: why the last mesh attempt returned None


def _refuse(reason: str):
    """Record why a mesh attempt is giving up, then give up. Diagnostic only -- callers still
    just see None, which stays the fail-closed contract."""
    _MESH_REFUSAL["reason"] = reason
    return None


_MESH_RUN_SEQ = itertools.count()  # one run dir per mesh-layer invocation
_MESH_PKG_CACHE: dict[str, object] = {}
_MESH_PKG_LOCK = threading.Lock()


def _built_mesh_package(pkg_dir: str, timeout: int):
    """The loaded+scanned+built package for ``pkg_dir``, built at most once per process.

    The whole-model mesh path calls this once per matmul layer. Rebuilding per layer is pure waste and
    turns one broken build into a per-layer storm of identical failures.
    """
    key = str(Path(pkg_dir).resolve())
    with _MESH_PKG_LOCK:
        hit = _MESH_PKG_CACHE.get(key)
    if hit is not None:
        return hit
    from ..targetgen.oot_runner import build_package, integrity_scan, load_package

    obj = load_package(pkg_dir, contract=None)  # the same sequence run_entrypoints does when pkg is None
    integrity_scan(obj)
    build_package(obj)
    with _MESH_PKG_LOCK:
        _MESH_PKG_CACHE[key] = obj
    return obj


def _requested_mesh_simulator(simulator: str | None = None) -> str | None:
    """Return the caller/legacy mesh-simulator request, without inventing a default."""
    import os

    requested = simulator if simulator is not None else os.environ.get("MERLIN_MESH_SIM")
    return str(requested).strip() if requested is not None and str(requested).strip() else None


def _resolve_oot_mesh_simulator(target: str, simulator: str | None = None) -> str:
    """Resolve the simulator for a real OOT mesh invocation through the L3 policy.

    ``MERLIN_MESH_SIM`` predates dynamic elaborated-RTL engine selection and remains an explicit
    functional-bootstrap override (notably ``spike``).  It must never override a campaign-wide
    ``MERLIN_REQUIRED_RTL_ENGINE`` pin.  With no explicit request, the shared Chipyard L3 policy chooses
    an available equal-fidelity engine; this is intentionally not a hidden Verilator fallback.
    """
    import os

    requested = _requested_mesh_simulator(simulator)
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None
    if required is not None and requested is not None and requested != required:
        raise RuntimeError(f"required RTL engine {required!r} conflicts with requested mesh simulator {requested!r}")

    # An unpinned explicit request is deliberate (e.g. Spike bootstrap), so preserve it.  A required
    # engine, even when repeated in MERLIN_MESH_SIM, is still availability-checked by the central policy.
    if required is None and requested is not None:
        return requested

    from ..targetgen.capsule_runner import chipyard_l3_selection

    selected = chipyard_l3_selection(target)
    engine = str(selected.get("engine") or "").strip()
    if not engine:
        raise RuntimeError(f"{target}: chipyard L3 policy returned no RTL engine")
    if required is not None and engine != required:
        raise RuntimeError(f"{target}: selected RTL engine {engine!r} differs from required RTL engine {required!r}")
    return engine


def _mesh_layer_id(m: int, k: int, n: int, binding, epilogue: list | None, acc_scale: float | None) -> str:
    """A per-layer artifact identity for the mesh run directory.

    Every mesh layer used the SAME run_id ("mesh_layer"), so all of a model's layers wrote to and read
    from one artifact directory and clobbered each other. The failure was not subtle once seen: two
    layers of two different models each received the OTHER's interface --
    ``(8,344)@(344,128)`` was handed ``(288,96)`` and ``(64,288)@(288,96)`` was handed ``(352,128)`` --
    and three further layers all expected one stale ``(128,128)``. A layer that fails this way runs
    perfectly in isolation, which is why it read as a shape bug rather than a collision.

    The id is CONTENT-ADDRESSED on what actually changes the emitted kernel: extent, operand and
    accumulator dtype, and the epilogue (an acc_scale requant emits a different kernel). Two genuinely
    identical layers therefore still share one directory -- small_llama repeats the same extent eight
    times and should compile once -- while two different layers can never collide."""
    parts = [f"{m}x{k}x{n}", binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)]
    if epilogue:
        parts.append("-".join(str(e) for e in epilogue))
    if acc_scale is not None:
        parts.append(f"s{float(acc_scale):.6g}")
    return "mesh_layer_" + "_".join(parts).replace(".", "p").replace("-", "_")


def _mesh_invocation_id(layer_id: str, A: list, W: list) -> str:
    """A stable run directory for one exact real-operand mesh invocation.

    ``layer_id`` identifies emitted code shape, not an execution.  Repeated layers of the same shape
    can carry different activations/weights and therefore produce different injected interfaces, ELFs
    and traces.  Sharing the shape directory overwrote the first call while its now-unrecoverable digest
    remained in the model ledger.  Bind the exact operands into the id; identical calls may safely reuse
    one directory because their complete compiler/oracle inputs are identical.
    """
    import hashlib

    payload = json.dumps({"A": A, "W": W}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"{layer_id}_input_{hashlib.sha256(payload).hexdigest()}"
