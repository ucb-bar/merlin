"""Bound prepared host declaration lookup to one unchanged program selection."""

from __future__ import annotations

from functools import wraps

from merlin.development.targetgen.execution.host.admission import PreparedHostCapabilities

# ---- private_source_admission ------------------------------------------------------------------


def prepared_host_admission(function):
    """Retain no state across programs, and never reuse source-body decisions."""

    @wraps(function)
    def admit(capture, target, software, capability, host, *args, **kwargs):
        prepared = PreparedHostCapabilities(host)
        result = function(capture, target, software, capability, prepared, *args, **kwargs)
        prepared.require_current(host)
        return result

    return admit
