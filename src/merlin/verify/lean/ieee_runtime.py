"""Retain the first failed integrity observation for the new IEEE qualification route.

Historical receipts keep their runtime implementation. Diagnostics never enter the runtime
identity and never cause a mismatching observation to be retried or accepted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NoReturn

from merlin.verify.lean import interpreted
from merlin.verify.lean.check import _file_ref


def _stat(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        return {
            "resolved": str(path.resolve()),
            "device": info.st_dev,
            "inode": info.st_ino,
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
    except OSError as error:
        return {"error": f"{type(error).__name__}: {error}"}


@dataclass(frozen=True)
class Runtime(interpreted.Runtime):
    diagnostics: Path

    def _reject(self, reason: str, observation: dict[str, Any]) -> NoReturn:
        self.diagnostics.mkdir(parents=True, exist_ok=True)
        path = self.diagnostics / f"mismatch-{time.time_ns()}.json"
        path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "runtime_sha256": self.digest,
                    "entry": str(self.entry),
                    **observation,
                },
                indent=2,
            )
            + "\n"
        )
        raise ValueError(f"{reason}; first mismatch retained in {path}")

    def assert_current(self) -> None:
        observed = tuple(interpreted._modules(self.search))
        if observed != self.modules:
            self._reject(
                "Lean proof artifact inventory changed during verification",
                {"expected_modules": list(map(str, self.modules)), "actual_modules": list(map(str, observed))},
            )
        for expected in self.records:
            path = Path(expected["path"])
            before = _stat(path)
            try:
                actual = _file_ref(path)
            except OSError as error:
                self._reject(
                    "Lean interpreter, entry or proof artifact became unreadable during verification",
                    {"expected": expected, "error": str(error), "stat_before": before, "stat_after": _stat(path)},
                )
            if actual != expected:
                self._reject(
                    "Lean interpreter, entry or proof artifact changed during verification",
                    {"expected": expected, "actual": actual, "stat_before": before, "stat_after": _stat(path)},
                )


def capture(entry: Path, *, required_module: str, diagnostics: Path) -> Runtime:
    """Keep the existing capture checks and add diagnostics to subsequent observations."""
    original = interpreted.capture(entry, required_module=required_module)
    runtime = Runtime(
        **{field.name: getattr(original, field.name) for field in fields(interpreted.Runtime)}, diagnostics=diagnostics
    )
    runtime.assert_current()
    return runtime
