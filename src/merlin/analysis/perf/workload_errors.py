"""Errors shared by workload fact derivation, encoding, and reference numerics."""


class WorkloadError(RuntimeError):
    """A required machine fact or faithfully encodable workload is unavailable.

    Refuse rather than guessing geometry, format, or control-flow behavior: a
    guessed program can run and return a plausible but invalid measurement.
    """
