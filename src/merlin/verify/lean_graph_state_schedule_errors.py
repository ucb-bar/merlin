"""Distinct scheduling input-domain refusals and identity/runtime unavailability.

The public host layer accepts only this explicit domain class; arbitrary
ValueError, identity changes, author bugs and process faults are unavailable.
"""


class ScheduleDomainRefusal(ValueError):
    """A specifically checked input, contract, schema or permutation refusal."""


class ScheduleUnavailable(RuntimeError):
    """A source/identity/runtime/author/transport failure; never a semantic refusal."""


class ScheduleIdentityUnavailable(ScheduleUnavailable):
    """Changed/unconfirmed live identity; hold the selected context."""


def require(condition, reason):
    if not condition:
        raise ScheduleDomainRefusal(reason)
