"""Bound prepared host declaration lookup to one unchanged program selection."""

from functools import wraps

from merlin.targetgen.host_admission_preparation import PreparedHostCapabilities


def prepared_host_admission(function):
    """Retain no state across programs, and never reuse source-body decisions."""

    @wraps(function)
    def admit(capture, target, software, capability, host, *args, **kwargs):
        prepared = PreparedHostCapabilities(host)
        result = function(capture, target, software, capability, prepared, *args, **kwargs)
        prepared.require_current(host)
        return result

    return admit
