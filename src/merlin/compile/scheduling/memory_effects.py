"""Exact decoded byte effects supplied by an instruction implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedAccess:
    """A half-open byte interval in an explicitly identified physical space."""

    space: str
    start: int
    size: int
    writes: bool = False

    def __post_init__(self):
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("address windows require an explicit space")
        if type(self.start) is not int or self.start < 0 or type(self.size) is not int or self.size <= 0:
            raise ValueError("address windows require a nonnegative base and positive byte size")
        if type(self.writes) is not bool:
            raise ValueError("encoded access direction must be explicit")
