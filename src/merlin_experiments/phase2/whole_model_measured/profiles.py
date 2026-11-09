"""Launch profiles: the authoring driver, model and budgets a measured run is launched with, as data.

A profile replaces a hand-written launch script.  Two checks run before a profile launches anything,
because each failed silently once:

* **The model resolves to itself.**  The driver's own alias resolver falls through to its default for a
  name it does not recognise; a stale ``model`` field once ran 40 rounds under a model nobody asked for.
* **The model is priced.**  The telemetry preflight refuses an unpriced model, so a profile naming one
  would fail at its first round instead of at launch.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase2.whole_model_measured.launch_profiles.v1"
REQUIRED = (
    "driver",
    "model",
    "effort",
    "max_sessions",
    "round_seconds",
    "total_authoring_seconds",
    "iteration_seconds",
    "max_tool_calls",
)


class ProfileError(ValueError):
    """The profile is unknown, malformed, or names a model that would not run as named."""


def _document() -> dict[str, Any]:
    source = files("merlin_experiments").joinpath("resources", "whole_model_measured_profiles.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ProfileError(f"the bundled launch profiles do not declare {SCHEMA}")
    return document


def names() -> list[str]:
    return sorted(_document()["profiles"])


def load(name: str) -> dict[str, Any]:
    profiles = _document()["profiles"]
    if name not in profiles:
        raise ProfileError(f"no launch profile {name!r} ({sorted(profiles)})")
    profile = dict(profiles[name])
    missing = [key for key in REQUIRED if key not in profile]
    if missing:
        raise ProfileError(f"launch profile {name!r} lacks {missing}")
    for key in REQUIRED[3:]:
        if isinstance(profile[key], bool) or not isinstance(profile[key], int) or profile[key] <= 0:
            raise ProfileError(f"launch profile {name!r}: {key} must be a positive integer")
    return {"name": name, **profile}


def check(profile: dict[str, Any], *, price_table: Path) -> dict[str, Any]:
    """Refuse a profile whose model would not run as named or is unpriced; return what was checked."""
    model = str(profile["model"])
    resolved = model
    if profile["driver"] == "codex":
        from merlin_experiments.phase1.providers import codex_agent

        resolved = str(codex_agent.resolve_model(model) or "")
        if resolved != model:
            raise ProfileError(
                f"the codex driver resolves {model!r} to {resolved!r}: the run would not use the model it names"
            )
    from merlin_experiments.phase2.telemetry import _declared_price_rate

    rate = _declared_price_rate(Path(price_table), resolved)
    if rate is None:
        raise ProfileError(
            f"{price_table} prices no model matching {resolved!r}; the telemetry preflight would refuse it"
        )
    return {"model": model, "resolved_model": resolved, "price_table": str(price_table), "rate": list(rate)}


__all__ = ["ProfileError", "SCHEMA", "check", "load", "names"]
