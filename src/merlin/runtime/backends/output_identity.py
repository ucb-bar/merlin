"""What a simulated run's answer IS, in a few bytes: compare two runs without re-running either."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def output_identity(res: dict[str, Any]) -> dict[str, Any]:
    """What a run's answer IS, in a few bytes, so two runs can be compared without re-running either.

    ``out_hash`` is the image's own whole-tensor ``HASH`` line (None when the image printed none);
    ``prefix_sha256`` digests the dumped console prefix bit for bit. Two runs computed the same answer
    when both match: the prefix exactly, the rest of the tensor through the hash. A caller asking
    "is the 2-hart output bit-identical to the 1-hart one" or "does the shipped image match its twin"
    otherwise has to keep the arrays or simulate again, and a whole-model simulation costs minutes.
    """
    prefix = res.get("prefix")
    out: dict[str, Any] = {"out_hash": res.get("out_hash")}
    if prefix is not None:
        arr = np.ascontiguousarray(np.asarray(prefix), dtype="<f4")
        out["prefix_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
        out["prefix_elems"] = int(arr.size)
    return out


def same_output(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    """Bit-identity of two :func:`output_identity` records: True/False, or None when it cannot be told.

    Unknown is not equal: a record without a prefix digest, or with a whole-tensor hash on one side
    only, cannot certify identity, and saying "identical" for it would be a verdict nobody measured.
    """
    if not a or not b or not a.get("prefix_sha256") or not b.get("prefix_sha256"):
        return None
    if a["prefix_sha256"] != b["prefix_sha256"] or a.get("prefix_elems") != b.get("prefix_elems"):
        return False
    ha, hb = a.get("out_hash"), b.get("out_hash")
    return None if ha is None or hb is None else ha == hb
