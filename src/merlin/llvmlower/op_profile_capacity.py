"""Bound operation-profiler storage by the complete generated ID table."""

from collections.abc import Mapping, Sequence

DEFAULT_CAPACITY = 8192
MAX_CAPACITY = 1 << 20


def runtime_capacity(table: Sequence[Mapping]) -> int:
    """Include the closing sentinel without folding valid IDs into overflow.

    The fixed ceiling bounds static runtime storage and dump work. It is a
    generic instrumentation resource limit, unrelated to hardware geometry.
    """
    if not table or len(table) >= MAX_CAPACITY:
        raise ValueError("operation profile table exceeds the supported ID budget")
    if any(
        not isinstance(row, Mapping) or type(row.get("id")) is not int or row["id"] != i for i, row in enumerate(table)
    ):
        raise ValueError("operation profile IDs must be contiguous from zero")
    return max(DEFAULT_CAPACITY, len(table) + 1)
