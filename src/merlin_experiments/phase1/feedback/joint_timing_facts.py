"""Reopen explicit operator mappings from extracted facts to timing model slots.

Selectors and semantic roles come from the target's operator intake. This reader
checks arithmetic consistency; it cannot turn unsupported prices or queue-role
substitutions into physical qualification.
"""

import math
from fractions import Fraction


def _at(document, path):
    if not isinstance(path, list) or not 1 <= len(path) <= 32:
        raise ValueError("fact/model selector must be a bounded nonempty path")
    for key in path:
        if type(key) not in (str, int):
            raise ValueError("fact/model selector has an unsupported key")
        document = document[key]
    return document


def verify(alignment, facts, model):
    """Refuse inconsistent direct geometry/credit mappings; never price traffic."""
    if alignment.get("schema") != "merlin.joint_timing_fact_alignment.v1":
        raise ValueError("unsupported timing fact alignment")
    rows = alignment.get("mappings")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("timing fact alignment needs explicit bounded mappings")
    checked = []
    for row in rows:
        selectors = row["fact_paths"]
        if not isinstance(selectors, list) or not 1 <= len(selectors) <= 8:
            raise ValueError("timing mapping has invalid factor roster")
        values = [_at(facts, path) for path in selectors]
        if any(type(v) is not int or not 0 < v < 2**63 for v in values):
            raise ValueError("timing mapping needs positive extracted integer facts")
        divisor = row.get("unit_divisor", 1)
        if type(divisor) is not int or divisor not in (1, 8):
            raise ValueError("timing mapping only supports identity or bit-to-byte conversion")
        expected = Fraction(math.prod(values), divisor)
        actual = _at(model, row["model_path"])
        if type(actual) not in (int, float) or actual != expected:
            raise ValueError("selected timing model contradicts extracted facts at " + str(row["model_path"]))
        checked.append(dict(model_path=row["model_path"], value=actual, role=row["role"]))
    return dict(status="consistent", checked=checked, timing_qualified=False,
                unsupported=alignment.get("unsupported", []),
                fit_from_measurement=alignment.get("fit_from_measurement", []))
