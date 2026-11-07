"""Measured RVV register-block caps offered to per-operation preparation."""

from __future__ import annotations

import os

_PEROP_NR_CAP = 16
_PEROP_NR_CAP_REF_VLEN = 256
_PEROP_MR_CAP = 4
_PEROP_MR_CAP_ENV = "MERLIN_PEROP_MR_CAP"
_PEROP_KC = 16


def perop_nr_cap(vlen: int | None) -> int:
    """Scale N lanes up with wider VLEN, retaining the measured cap at VLEN <= 256."""
    if not vlen or vlen <= _PEROP_NR_CAP_REF_VLEN:
        return _PEROP_NR_CAP
    return _PEROP_NR_CAP * (int(vlen) // _PEROP_NR_CAP_REF_VLEN)


def perop_mr_cap() -> int:
    """Offer four accumulator rows unless a positive integer env override is set."""
    raw = os.environ.get(_PEROP_MR_CAP_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _PEROP_MR_CAP


def default_perop_caps(vlen: int | None) -> tuple[int, int, int]:
    """Return the (MR, NR, KC) caps used by the current RVV recipe."""
    return perop_mr_cap(), perop_nr_cap(vlen), _PEROP_KC
