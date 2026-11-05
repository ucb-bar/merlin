"""Public own-artifact feature projection and training-only feature selection.

Empirical coefficients predict the original measured call; they are never
physical service prices. Missing or unidentifiable terms are not priced zero.
The independent held qualification determines whether this projection suffices.
"""

import numpy as np

from merlin.common.digest import is_sha256
from merlin.perf import dynamic_census as C
from merlin.perf.candidate_features import FEATURE_KEYS

LEGACY_FEATURES = ("/compiled/emitted/action_sites", "/compiled/linked/executable_bytes")
FEATURES = (*LEGACY_FEATURES, *("/candidate/" + key for key in FEATURE_KEYS), *C.FEATURES)


def project(compiled):
    values = dict.fromkeys(FEATURES)
    values.update({key: compiled.get("features", {}).get(key) for key in LEGACY_FEATURES})
    own = compiled.get("candidate_diagnostics", {}).get("candidate_features", {})
    if own.get("schema") == "merlin.candidate_features.v1":
        values.update({"/candidate/" + key: own.get("features", {}).get(key) for key in FEATURE_KEYS})
    census = compiled.get("dynamic_census")
    if not isinstance(census, dict):
        return values
    identity, features, complete = (census.get(key) for key in ("identity", "candidate_features", "completeness"))
    if (compiled.get("status") != "observed" or census.get("schema") != C.SCHEMA
            or census.get("status") != "observed" or not isinstance(identity, dict)
            or not is_sha256(compiled.get("elf_sha256"))
            or identity.get("elf_sha256") != compiled["elf_sha256"]
            or not isinstance(features, dict) or features.get("status") != "descriptive"
            or not isinstance(features.get("features"), dict) or not isinstance(complete, dict)
            or complete.get("stream_eof") is not True or complete.get("truncated") is not False
            or complete.get("descendants_reaped") is not True):
        return values
    for key in C.FEATURES:
        count = features["features"].get(key)
        # Counts are bounded nonnegative integers from the original observer.
        # Missing/unsupported terms never become measured zero or ISS timing.
        if type(count) is int and count >= 0:
            values[key] = count
    return values


def identifiable_pointers(rows):
    """Use training activity only; never select columns from held cycle labels."""
    if not rows:
        raise ValueError("screen feature selection needs independent training observations")
    matrix = np.ones((len(rows), 1))
    pointers = []
    for key in FEATURES:
        values = [row.features.get(key) for row in rows]
        if any(value is None for value in values) or min(values) == max(values):
            continue
        column = np.array(values, dtype=float)
        column /= max(float(max(column)), 1.0)
        candidate = np.column_stack((matrix, column))
        if np.linalg.matrix_rank(candidate) > matrix.shape[1]:
            matrix = candidate
            pointers.append(key)
    if not pointers:
        raise ValueError("public training features cannot distinguish executable alternatives")
    return tuple(pointers)
