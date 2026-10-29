"""Original-source admission for bounded compiler verification profiles.

Admission selects a proof model; it grants no verification or delivery
authority. Only an explicit exclusion of the original source may bypass a
compiler proof. Errors after admission keep their ordinary blocking behavior.
"""

from __future__ import annotations


class SourceOutsideProfile(ValueError):
    """A known, documented original-source form is outside a fixed model.

    Runtime wrappers record this as UNVERIFIED and retain ordinary compiler
    behavior. They must not catch arbitrary parser, tool, resource, custody or
    proof failures as this exception. A source predicate must never inspect a
    proposed target to decide whether the original source is supported.
    """

    def __init__(self, profile_id: str, code: str, reason: str):
        if any(type(value) is not str or not value for value in (profile_id, code, reason)):
            raise TypeError("source exclusions require a fixed profile, reason code and explanation")
        self.profile_id = profile_id
        self.code = code
        self.reason = reason
        super().__init__(reason)
