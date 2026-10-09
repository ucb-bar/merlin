"""Explicit functional execution transport without target-backend discovery.

The caller owns independent runtime qualification. Source pins and selected
callbacks here provide attribution and drift checks, not ISA semantics, hardware
timing or correctness authority. This transport cannot label counters as RTL.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType

from .build_service import file_digest


def _callback_selection(callback):
    """Shallow actual callable selection; reachable state and imports are unproved."""
    original, wrappers = callback, []
    while type(callback) is partial:
        if any(type(key) is not str for key in callback.keywords):
            return None
        wrappers.append((callback, callback.func, callback.args, tuple(sorted(callback.keywords.items()))))
        callback = callback.func
    if type(callback) not in (FunctionType, MethodType):
        return None
    method = type(callback) is MethodType
    function = callback.__func__ if method else callback
    return original, function, callback.__self__ if method else None, function.__code__, tuple(wrappers)


def _same_callback(current, selected):
    if current is None or selected is None or any(a is not b for a, b in zip(current[:4], selected[:4], strict=True)):
        return False
    if len(current[4]) != len(selected[4]):
        return False
    for actual, original in zip(current[4], selected[4], strict=True):
        if any(a is not b for a, b in zip(actual[:3], original[:3], strict=True)):
            return False
        if len(actual[3]) != len(original[3]):
            return False
        if any(
            key != old_key or value is not old_value
            for (key, value), (old_key, old_value) in zip(actual[3], original[3], strict=True)
        ):
            return False
    return True


@dataclass(frozen=True)
class FunctionalExecutionService:
    target: str
    simulator: str
    runner: Callable
    parser: Callable
    source_pins: tuple[tuple[str, str], ...]
    engine_json: str
    _callback_selections: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Invalid callback forms remain constructible for ordinary validation
        # diagnostics; no such callback can pass verify or be invoked here.
        object.__setattr__(
            self,
            "_callback_selections",
            tuple(_callback_selection(callback) for callback in (self.runner, self.parser)),
        )

    def verify(self, target: str, simulator: str) -> dict:
        if (
            type(self) is not FunctionalExecutionService
            or target != self.target
            or simulator != self.simulator
            or not isinstance(self.source_pins, tuple)
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("functional execution requires an exact target/engine transport")
        for path, expected in self.source_pins:
            member = Path(path)
            if (
                not member.is_absolute()
                or any(parent.is_symlink() for parent in (member, *member.parents))
                or not member.is_file()
                or file_digest(member) != expected
            ):
                raise ValueError("functional execution source/tool pin changed: " + str(path))
        for callback, selected in zip((self.runner, self.parser), self._callback_selections, strict=True):
            current = _callback_selection(callback)
            if current is None:
                raise ValueError("functional execution callback must be an actual Python function or bound method")
            if not _same_callback(current, selected):
                raise ValueError("functional execution selected callback implementation or partial bindings changed")
            owner = inspect.getsourcefile(current[1])
            if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in self.source_pins:
                raise ValueError("functional execution callback has no pinned inspected source owner")
        from merlin.common.strict_json import loads

        engine = loads(self.engine_json)
        if type(engine) is not dict or not engine:
            raise ValueError("functional execution requires an explicit selected-engine citation")
        return {
            "target": target,
            "simulator": simulator,
            "engine": engine,
            "source_pins": [{"path": path, "sha256": digest} for path, digest in self.source_pins],
            "scope": "functional transport only; ISA semantics and hardware/cost authority unqualified",
        }

    def run_elf(self, elf, *, simulator, timeout, **kwargs):
        before = self.verify(self.target, simulator)
        result = self.runner(elf, simulator=simulator, timeout=timeout, **kwargs)
        if self.verify(self.target, simulator) != before:
            raise ValueError("functional execution selection changed during invocation")
        return result

    def parse_output(self, console):
        before = self.verify(self.target, self.simulator)
        result = self.parser(console)
        if self.verify(self.target, self.simulator) != before:
            raise ValueError("functional execution selection changed during parsing")
        return result

    @property
    def oracle(self) -> dict:
        return {
            "kind": "functional_diagnostic",
            "derived_from_rtl": False,
            "hardware_timing": "unqualified",
            "engine": json.loads(self.engine_json),
        }
