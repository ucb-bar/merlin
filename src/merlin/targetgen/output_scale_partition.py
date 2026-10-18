"""Bind a logical column-scale obligation without assigning a physical schedule.

Independent contraction output columns may use different scalar readout gains.
The hardware vector width still describes the elements sharing one physical
gain. This record preserves the source gain vector and that width; target
packing, commands, addresses, legality and execution remain compiler obligations.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real

SCHEMA = "logical_output_column_partition_v1"


def obligation(gains, physical_columns):
    """A source-arithmetic obligation for gains varying within a physical block.

    Return None for an already block-uniform vector. Callers establish that
    these are independent source output columns; this function grants no such
    independence, hardware admission, implementation or timing authority.
    """
    if type(physical_columns) is not int or physical_columns < 1:
        raise ValueError("the physical scale-vector width must be a positive integer")
    if len(gains) == 0:
        raise ValueError("a column-scale vector must be nonempty")
    digest = hashlib.sha256()
    values = []
    for value in gains:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("column gains must be positive finite numeric values")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("column gains must be positive finite numeric values")
        # The caller retains the selected numeric format in the complete proof.
        # Bind its exact already-typed values without imposing another format.
        digest.update(value.hex().encode("ascii") + b"\n")
        values.append(value)
    if all(len(set(values[start:start + physical_columns])) == 1
           for start in range(0, len(values), physical_columns)):
        return None
    return {
        "schema": SCHEMA,
        "scope": "source_arithmetic_only",
        "columns": len(values),
        "physical_columns_per_scale": physical_columns,
        "gains_sha256": digest.hexdigest(),
        "requires": "independent_output_column_implementation",
    }
