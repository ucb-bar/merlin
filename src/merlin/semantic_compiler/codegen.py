"""Invocation-scoped code generation choices and measured variant selection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

FACTORS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CodegenOptions:
    reduction_unroll: str = "auto"
    program_size_limit_bytes: int | None = None
    tuning_profile: dict | None = None
    io_memory_bytes: int | None = None

    def __post_init__(self):
        if self.reduction_unroll not in ("auto", *(str(i) for i in FACTORS)):
            raise ValueError("reduction unroll must be auto, 1, 2, 4 or 8")
        size = self.program_size_limit_bytes
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("program size budget must be a positive integer")
        if self.tuning_profile is not None:
            check_profile(self.tuning_profile)
        if self.io_memory_bytes is not None and (type(self.io_memory_bytes) is not int or self.io_memory_bytes <= 0):
            raise ValueError("device I/O memory window must have a positive byte size")


def check_profile(profile):
    if not isinstance(profile, dict) or profile.get("schema") != "merlin.native_tuning_profile.v1":
        raise ValueError("unsupported tuning profile")
    if not isinstance(profile.get("key"), dict) or profile.get("objective") not in ("device_cycles", "warm_host_ticks"):
        raise ValueError("tuning profile needs its compilation key and measurement objective")
    if not isinstance(profile.get("variants"), list):
        raise ValueError("tuning profile needs measured variants")
    seen = set()
    for row in profile["variants"]:
        factor = row.get("factor")
        if type(factor) is not int or factor not in FACTORS or factor in seen:
            raise ValueError("tuning factors must be distinct members of 1, 2, 4, 8")
        seen.add(factor)
        if row.get("status") != "passed":
            continue
        digest = row.get("program_sha256")
        if digest is not None and (
            not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("measured program identity must be a SHA256 hex digest")
        samples = row.get(profile["objective"])
        if (
            row.get("correctness") != "passed"
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not isinstance(samples, list)
            or len(samples) != 5
            or any(type(v) is not int or v < 0 for v in samples)
        ):
            raise ValueError("eligible tuning variant needs correctness and five measurements")
    if seen != set(FACTORS):
        raise ValueError("tuning profile must account for all four factors")


_OPTIONS = ContextVar("native_codegen_options", default=CodegenOptions())
_MATERIALIZATIONS = ContextVar("native_materializations", default=None)


@contextmanager
def codegen_options(options: CodegenOptions):
    """Share immutable options through nested target-region compilations."""
    token = _OPTIONS.set(options)
    cache_token = _MATERIALIZATIONS.set({})
    try:
        yield
    finally:
        _MATERIALIZATIONS.reset(cache_token)
        _OPTIONS.reset(token)


def current_codegen_options():
    return _OPTIONS.get()


def materialization_cache():
    """Optional invocation-local cache; providers own exact keys and replay checks."""
    return _MATERIALIZATIONS.get()


def options_from_args(args):
    path = getattr(args, "tuning_profile", None)
    return CodegenOptions(
        getattr(args, "reduction_unroll", "auto"),
        getattr(args, "program_size_limit_bytes", None),
        json.loads(Path(path).read_text()) if path else None,
        getattr(args, "io_memory_bytes", None),
    )


def choose_reduction_unroll(key: dict):
    options = current_codegen_options()
    profile = options.tuning_profile
    factor, reason = 1, "no_matching_profile"
    if options.reduction_unroll != "auto":
        factor, reason = int(options.reduction_unroll), "explicit_factor"
    elif profile is not None and profile["key"] == key:
        eligible = [row for row in profile["variants"] if row["status"] == "passed"]
        if not eligible:
            raise ValueError("matching tuning profile has no checked variant")
        # Different requested factors can emit identical code, for example a
        # contraction that fits one hardware tile. Timing noise must not select
        # a fictitious optimization. Keep the smallest factor for each image.
        distinct = {}
        for row in sorted(eligible, key=lambda row: row["factor"]):
            identity = row.get("program_sha256", ("factor", row["factor"]))
            distinct.setdefault(identity, row)
        eligible = list(distinct.values())
        if eligible:
            objective = profile["objective"]
            best = min(eligible, key=lambda row: (sorted(row[objective])[2], row["bytes"], row["factor"]))
            factor, reason = best["factor"], "measured_profile"
    record = {
        "reduction_unroll": factor,
        "selection": reason,
        "program_size_limit_bytes": options.program_size_limit_bytes,
    }
    if profile is not None:
        record["tuning_profile_sha256"] = sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return factor, record
