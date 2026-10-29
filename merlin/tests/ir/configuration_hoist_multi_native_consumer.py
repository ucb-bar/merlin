"""Finite complete-state observations across sibling scopes and failures.

The ordinary C emitter consumes the delivered actual Kernel. Setter counts are
cost instrumentation, excluded by the pure-setter premise. These comparisons
supply no universal C/encoder or hardware interpretation theorem.
"""

import ctypes

from configuration_hoist_prefix_native_consumer import library


def compare(source, delivered, directory):
    _, before = library(source, directory / "original")
    _, after = library(delivered, directory / "delivered")
    rows = []
    for x, y in ((-9, 17), (0, 0), (31, -11)):
        for fail_at in (-1, 0, 1, 3, 6, 7, 9, 13, 20):
            left, right = (ctypes.c_int64 * 8)(), (ctypes.c_int64 * 8)()
            before(x, y, fail_at, left)
            after(x, y, fail_at, right)
            assert list(left)[:7] == list(right)[:7]
            assert right[7] <= left[7]
            rows.append(
                {
                    "initial": [x, y],
                    "fail_at": fail_at,
                    "observation": list(left)[:7],
                    "setters_original": left[7],
                    "setters_delivered": right[7],
                }
            )
    return rows
