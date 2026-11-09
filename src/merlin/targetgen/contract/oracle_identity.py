"""Observed simulator identity and the selected execution deadline ceiling."""

from __future__ import annotations

from typing import Any


def simulator_provenance(backend, simulator: str) -> dict[str, Any] | None:
    """Identify the simulator BINARY an engine ran on: its path, its digest, and its recorded lineage.

    A result that claims a hardware verdict must record which hardware revision it came from. A cert
    record already pins the ELF, the RTL commits and the toolchain — and said nothing at all about the
    prebuilt simulator that produced the numbers, which for GSIM is an out-of-tree build whose bytes are
    the only thing tying the verdict to an elaboration. A stale or mis-provenanced emulator was therefore
    undetectable after the fact, which is the same failure mode the binary provenance stamp was added to
    catch on the ELF side.

    Derived from the backend, not from a table: it is asked for ``<engine>_status()`` (a sentence about
    how the engine resolved) and ``<engine>_path()`` (where its binary is), both DERIVED attribute names,
    so an engine or a backend that does not publish them contributes nothing rather than a fabricated
    entry. Never raises — provenance that cannot be established is recorded as absent.
    """
    from pathlib import Path as _Path

    rec: dict[str, Any] = {"engine": simulator}
    try:
        getter = getattr(backend, f"{simulator}_path", None)
        if callable(getter):
            binary = _Path(str(getter()))
            rec["binary"] = str(binary)
            if binary.is_file():
                from merlin.common import provenance as _prov

                rec["sha256"] = _prov.file_digest(binary)
        status = getattr(backend, f"{simulator}_status", None)
        if callable(status):
            _ok, _why = status()
            rec["resolution"] = str(_why)
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec if len(rec) > 1 else None


def _selected_run_timeout(backend, simulator, requested, deadline):
    """Keep the job's remaining budget within the selected process ceiling."""
    import math

    remaining = min(requested, deadline.remaining())
    selected_limit = getattr(backend, "execution_timeout_limit", None)
    if selected_limit is not None:
        limit = selected_limit(simulator)
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("selected execution deadline ceiling must be finite and positive")
        remaining = min(remaining, limit)
    return remaining


# Preserve historical callable and record pickle lookup identities.
simulator_provenance.__module__ = "merlin.targetgen.contract.compile"
_selected_run_timeout.__module__ = "merlin.targetgen.contract.compile"
