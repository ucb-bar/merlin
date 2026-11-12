"""Pure original borrowed-call effects; no checker, discovery or runtime access.

Callers qualify these premises separately. Types and identities cannot establish
that an implementation obeys them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BorrowedBufferEffects:
    """Explicit original implementation premises, not a derived qualification."""

    qualification_identity: str
    no_retained_aliases: bool
    no_releases: bool
    complete_before_return: bool

    def __post_init__(self):
        if (
            type(self.qualification_identity) is not str
            or not self.qualification_identity.strip()
            or len(self.qualification_identity) > 4096
            or any(
                value is not True
                for value in (
                    self.no_retained_aliases,
                    self.no_releases,
                    self.complete_before_return,
                )
            )
        ):
            raise ValueError("borrowed ABI requires explicit noescape, no-release and synchronous effects")


def validate_effects(effects):
    if effects is not None:
        if type(effects) is not BorrowedBufferEffects:
            raise ValueError("explicit typed borrowed ABI implementation effects required")
        effects.__post_init__()
