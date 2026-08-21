"""Vortex target-package backend — the ``plugin.backend`` module ``base._oot_backend_modules`` loads.

Its only job is REGISTRATION: it plugs the Vortex pure-SIMT target into the two public seams the
target-generalization framework exposes, so the shared dispatch never learns the name ``vortex``.

  * G2 — ``mlc_bridge.register_simt_introspect``: the RTL-fact introspect (opcode/CSR discovery walked
    out of the CIRCT HW-dialect import); geometry is the descriptor-frozen ``VX_CFG_*`` build parameter.
  * G3 — ``capsule_runner.register_sim_oracle``: the simx (L2, functional) + rtlsim (L3, cycle-exact)
    oracle, registered EXCLUSIVE under ``sim_via: vortex`` so a self-hosted SIMT core is graded on its
    own emitted kernel ELF — never the arc command-buffer / program-oracle default, which would grade
    the wrong artifact.

Both registrations run at import time. No runtime ``BackendInfo`` is registered: grading routes through
the sim-oracle seam, not through ``get_backend`` — the kernel is built and run by ``vortex_oracle``, and
the compiler under test is the AGENT's out-of-tree package, not this shim.
"""
from __future__ import annotations

from merlin.targetgen import capsule_runner as _CR
from merlin.targetgen import vortex_oracle as _VO
from merlin.targetgen.rtl import mlc_bridge as _MB
from merlin.targetgen.rtl import vortex_introspect as _VI


def _vortex_available(target: str) -> tuple[bool, str]:
    """(ok, reason) pre-spend probe for the simx/rtlsim oracle — fail closed, never a silent pass."""
    if _VO.available("L2"):
        return True, f"{target!r}: vortex simx/rtlsim oracle available (curated harness staged)"
    return False, (f"{target!r}: vortex oracle unavailable — set MERLIN_EXT_VORTEX and stage the curated "
                   "harness (build_harness.sh). The arc/program default grades the wrong artifact for a "
                   "SIMT kernel and is not a valid fallback.")


# G3: simx (L2) + rtlsim (L3), exclusive so it replaces the program-oracle default for this SIMT core.
_CR.register_sim_oracle("vortex", adapters=lambda target: _VO.adapters(),
                        available=_vortex_available, exclusive=True)

# G2: the RTL-fact introspect, keyed by _VI.TARGET == "vortex".
_MB.register_simt_introspect(_VI)
