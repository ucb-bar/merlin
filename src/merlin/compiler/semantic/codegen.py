"""Invocation-scoped code generation choices and measured variant selection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

FACTORS = (1, 2, 4, 8)
APPLICATION_BUDGET_OPTIONS = ("program_size_limit_bytes", "execution_instruction_limit", "exact_float_unroll")


def application_budget_options(manifest, requested):
    """Read published choices, refusing conflicting operator restatements."""
    if not isinstance(manifest, dict):
        raise ValueError("application manifest must be an object")
    published = {name: manifest[name] for name in APPLICATION_BUDGET_OPTIONS if name in manifest}
    if not published:  # Legacy application without published budgets.
        return {name: value for name, value in requested.items() if value is not None}
    if set(published) != set(APPLICATION_BUDGET_OPTIONS):
        raise ValueError("incomplete application budget publication")
    CodegenOptions(**published)
    for name, value in requested.items():
        if value is not None and json.dumps(value) != json.dumps(published[name]):
            raise ValueError("selected budget differs from compiled application: " + name)
    return published


IO_MEMORY_OPTIONS = (
    "io_memory_bytes",
    "io_memory_base",
    "io_memory_alignment",
    "io_memory_capacity_base",
    "io_memory_capacity_bytes",
)

@dataclass(frozen=True)
class CodegenOptions:
    reduction_unroll: str = "auto"
    program_size_limit_bytes: int | None = None
    tuning_profile: dict | None = None
    io_memory_bytes: int | None = None
    exact_float_unroll: bool = False
    execution_instruction_limit: int | None = None
    application_buffer_views: bool = False
    io_memory_base: int | None = None
    io_memory_alignment: int | None = None
    io_memory_capacity_base: int | None = None
    io_memory_capacity_bytes: int | None = None
    application_residency: bool = False
    resident_bindings: dict | None = None
    # Linked bytes of this application's whole on-chip host (code, data, heap, and stack), from the caller's link.
    measured_host_bytes: int | None = None

    def __post_init__(self):
        if type(self.application_residency) is not bool:
            raise ValueError("application residency must be Boolean")
        if self.application_residency and (self.io_memory_capacity_base is None or self.io_memory_capacity_bytes is None):
            raise ValueError("application residency requires declared physical capacity")
        if self.reduction_unroll not in ("auto", *(str(i) for i in FACTORS)):
            raise ValueError("reduction unroll must be auto, 1, 2, 4 or 8")
        if type(self.exact_float_unroll) is not bool:
            raise ValueError("exact float unroll must be Boolean")
        if type(self.application_buffer_views) is not bool:
            raise ValueError("application buffer views must be Boolean")
        if self.execution_instruction_limit is not None and (
            type(self.execution_instruction_limit) is not int or self.execution_instruction_limit <= 0
        ):
            raise ValueError("execution instruction limit must be a positive integer")
        size = self.program_size_limit_bytes
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("program size budget must be a positive integer")
        if self.tuning_profile is not None:
            check_profile(self.tuning_profile)
        if self.io_memory_bytes is not None and (type(self.io_memory_bytes) is not int or self.io_memory_bytes <= 0):
            raise ValueError("device I/O memory window must have a positive byte size")
        for name in ("io_memory_base", "io_memory_capacity_base"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("io_memory_alignment", "io_memory_capacity_bytes", "measured_host_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    def identity_record(self):
        """Record active recipe choices without changing historical default identities."""
        record = asdict(self)
        if not self.application_residency:
            del record["application_residency"]
        if self.resident_bindings is None:
            del record["resident_bindings"]
        if not self.exact_float_unroll:
            del record["exact_float_unroll"]
        if self.execution_instruction_limit is None:
            del record["execution_instruction_limit"]
        if not self.application_buffer_views:
            del record["application_buffer_views"]
        if self.measured_host_bytes is None:
            del record["measured_host_bytes"]
        for name in IO_MEMORY_OPTIONS[1:]:
            if record[name] is None:
                del record[name]
        return record


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
    budget = getattr(args, "memory_budget", None)
    if budget is not None:
        # The device window and its capacity are one free range of the caller's derived memory budget.
        from copy import copy
        from merlin.compiler.build.memory_budget import budget_window
        if getattr(args, "io_memory_bytes", None) is not None:
            raise ValueError("a memory budget selects the device window; omit --io-memory-bytes")
        args = copy(args)
        args.io_memory_base, args.io_memory_bytes = budget_window(json.loads(Path(budget).read_text()))
        args.io_memory_capacity_base, args.io_memory_capacity_bytes = args.io_memory_base, args.io_memory_bytes
    factory = getattr(args, "_codegen_options_type", CodegenOptions)
    prepare = getattr(factory, "prepare_args", None)
    if prepare is not None:
        from copy import copy
        args = copy(args)
        prepare(args)
    path = getattr(args, "tuning_profile", None)
    generic = CodegenOptions(
        getattr(args, "reduction_unroll", "auto"),
        getattr(args, "program_size_limit_bytes", None),
        json.loads(Path(path).read_text()) if path else None,
        getattr(args, "io_memory_bytes", None),
        getattr(args, "exact_float_unroll", False),
        getattr(args, "execution_instruction_limit", None),
        getattr(args, "application_buffer_views", False),
        io_memory_base=getattr(args, "io_memory_base", None),
        io_memory_alignment=getattr(args, "io_memory_alignment", None),
        io_memory_capacity_base=getattr(args, "io_memory_capacity_base", None),
        io_memory_capacity_bytes=getattr(args, "io_memory_capacity_bytes", None),
        application_residency=getattr(args, "application_residency", False),
        measured_host_bytes=getattr(args, "measured_host_bytes", None),
    )
    factory = getattr(args, "_codegen_options_type", CodegenOptions)
    if factory is CodegenOptions:
        return generic
    from dataclasses import fields
    supplied = asdict(generic)
    supplied.update({field.name: getattr(args, field.name) for field in fields(factory)
                     if field.name not in supplied and hasattr(args, field.name)})
    return factory(**supplied)



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
