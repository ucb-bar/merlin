"""Muon cyclotron sim-oracle plugin — the OUT-OF-TREE home of the bespoke SIMT grading oracle.

Declared by the muon target contract as ``plugin.sim_oracle`` and loaded by
:func:`merlin.targetgen.capsule_runner._ensure_sim_oracles_discovered` (as ``merlin._oot_sim_oracles.muon``)
through the SAME reference/OOT plugin discovery the runtime backends use. Importing this module registers
the ``cyclotron`` bespoke-sim oracle so a target that declares ``toolchain.sim_via == cyclotron`` (a
self-hosted SIMT core, e.g. radiance) routes to it — WITHOUT capsule_runner carrying a hardcoded
``_SIM_ORACLES["cyclotron"]`` entry.

Behaviorally IDENTICAL to the former hardcoded ``_cyclotron_adapters`` / ``_cyclotron_available`` in
capsule_runner: the adapters resolve to the (relocated) ``muon_oracles.default_adapters()`` and the
availability probe requires the cyclotron oracle via the muon backend's ``available("cyclotron")``, failing
closed (the mlc arc command-buffer adapter grades the wrong artifact for a SIMT target and is NOT a valid
fallback). This is a LEAF module (no submodule search), so it reaches the relocated muon package only via
the registry (``get_backend("muon")``).
"""
from __future__ import annotations

from merlin.targetgen.capsule_runner import register_sim_oracle


def _cyclotron_adapters(target: str) -> dict:
    """A self-hosted SIMT target graded on its emitted kernel ELF by the bespoke cyclotron/VCS oracle
    (``muon_oracles``) — NOT the arc command-buffer path, which grades the wrong artifact for a SIMT
    kernel. The muon adapters fail closed (MuonUnavailable) when the MERLIN_MUON_* toolchain env is
    unset, so an unwired target degrades honestly, never mis-grades."""
    from merlin.runtime.backends.base import get_backend
    return get_backend("muon").muon_oracles.default_adapters(target)


def _cyclotron_available(target: str) -> tuple[bool, str]:
    """cyclotron (SIMT): require the cyclotron oracle; the mlc arc command-buffer adapter grades the wrong
    artifact for a SIMT kernel, so it is NOT a valid fallback (that was a false-green). Fail closed."""
    try:
        from merlin.runtime.backends.base import get_backend
        if get_backend("muon").available("cyclotron"):
            return True, f"{target!r}: cyclotron SIMT oracle available"
    except Exception:  # noqa: BLE001 — an unimportable backend is honestly unavailable
        pass
    return False, (f"{target!r}: cyclotron SIMT oracle unavailable (set the MERLIN_MUON_* env); the mlc "
                   "arc command-buffer adapter grades the wrong artifact for a SIMT target and is not "
                   "a valid fallback")


# Register the bespoke cyclotron oracle (exclusive: replaces the arc/program default for a self-hosted
# SIMT core). Idempotent; runs at plugin import, which discovery triggers before oracle routing.
def _cyclotron_l3_selection(target: str) -> dict:
    """Which elaborated-RTL engine this sim's L3 cert tier resolved to, and what it was chosen over.

    Contributed so the shared reporter (`capsule_runner.describe_l3_engine`) can ASK rather than reach
    into a backend it names — a target routed through an exclusive bespoke sim never enters the chipyard
    selection path, so before this the engine policy simply did not apply to it and nothing anywhere said
    which simulator its cert ran on.
    """
    from merlin.runtime.backends.base import get_backend
    return get_backend("muon").muon_oracles.l3_selection(target)


register_sim_oracle("cyclotron", adapters=_cyclotron_adapters,
                    available=_cyclotron_available, exclusive=True,
                    l3_selection=_cyclotron_l3_selection)
