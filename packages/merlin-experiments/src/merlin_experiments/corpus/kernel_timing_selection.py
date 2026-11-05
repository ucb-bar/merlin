"""Stage an explicitly pinned additive kernel-counter selection for a release."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from merlin.common.digest import sha256_file
from merlin.runtime.kernel_call_timing import from_contract
from merlin_experiments.spec import SpecError


def select(original: dict, selection: tuple[Path, str]) -> tuple[dict, dict]:
    path, digest = selection
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise SpecError("selected kernel timing contract is absent or changed")
    candidate = yaml.safe_load(path.read_text())
    if not isinstance(candidate, dict):
        raise SpecError("selected kernel timing contract must be a mapping")
    expected = copy.deepcopy(original)
    counters = candidate.get("counter_semantics") or {}
    logical = candidate.get("logical_harness") or {}
    if not isinstance(counters, dict) or not isinstance(logical, dict):
        raise SpecError("selected kernel timing declarations must be mappings")
    for key, value in counters.items():
        current = expected.setdefault("counter_semantics", {})
        if key in current and current[key] != value:
            raise SpecError("kernel timing selection changes existing counter semantics")
        current[key] = value
    for key in ("cycle_counter_asm", "kernel_call_timing"):
        if key in logical:
            current = expected.setdefault("logical_harness", {})
            if key in current and current[key] != logical[key]:
                raise SpecError("kernel timing selection changes an existing timing declaration")
            current[key] = logical[key]
    if expected != candidate:
        raise SpecError("kernel timing selection changes non-timing contract fields")
    try:
        timing = from_contract(candidate)
        if timing is None or logical.get("complete_call_timing") is not None:
            raise ValueError("one explicit kernel timing selection is required")
        timing.verify()
    except (ValueError, OSError) as error:
        raise SpecError(str(error)) from error
    if sha256_file(path) != digest:
        raise SpecError("selected kernel timing contract changed during selection")
    return candidate, {"path": str(path), "sha256": digest, **timing.record()}
