"""Environment selection shared by frozen experiment launches and inspection."""

from __future__ import annotations

import os
from collections.abc import Mapping

from merlin.experiments.definition import SpecError

FROZEN_PHASE0_ENV_POLICY = "merlin.phase0.selected_environment.v1"
UNSEALED_BOARD_CATALOG_ENV = "MERLIN_BOARD_CATALOG"


def without_unsealed_board_catalog(environment: Mapping[str, str]) -> dict[str, str]:
    """Keep ambient board facts out of verified children until they have a frozen owner.

    A path in the host environment is not a source receipt: it may alias another
    location or change bytes between freeze and execution. This policy binds
    absence; a later declared-input implementation can replace it with a sealed
    catalog path, digest and alias check.
    """
    selected = dict(environment)
    selected.pop(UNSEALED_BOARD_CATALOG_ENV, None)
    return selected


def execution_environment(command: dict) -> dict[str, str]:
    """Use the frozen installed roots even if the invoking ambient selection changed."""
    if command.get("module") and UNSEALED_BOARD_CATALOG_ENV in command.get("env", {}):
        raise SpecError("installed stage has an unsealed board catalog selection; freeze a declared input")
    if command.get("adapter") == "capsule_derivation" and command.get("source_snapshot"):
        if command.get("phase0_environment_policy") != FROZEN_PHASE0_ENV_POLICY:
            raise SpecError("frozen test generation launch predates selected-only environment; freeze a new run")
        # A frozen derivation may execute source generators. Inheriting a live
        # Model2MLIR interpreter, checkout, or simulator makes its output depend
        # on inputs absent from the selected source receipt and changes resume.
        # Optional generators remain unavailable until their owners are
        # explicitly selected and frozen by test generation.
        return {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            **command["env"],
        }
    environment = dict(os.environ, **command["env"])
    if command.get("module"):
        environment = without_unsealed_board_catalog(environment)
    return environment
