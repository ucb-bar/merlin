"""FireSim workload handoff; infrastructure execution belongs to the operator.

Legacy runner selection remains an explicit migration error. This module never
loads launch plugins, submits queues, provisions machines or runs a simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from merlin.compiler.artifact import Artifact, Deployment
from merlin.runtime.external import ExternalObservation, PreparedRun, collect, prepare


class FireSimRunner(Protocol):
    """Historical launch signature retained only for import compatibility."""

    def __call__(self, elf: str, **options) -> str: ...


class FireSimRunnerError(ValueError):
    """An unsupported launch request or incomplete external observation."""


def select_runner(name: str | None = None) -> FireSimRunner:
    """Refuse the old launch route before discovering or importing a plugin."""
    raise FireSimRunnerError(
        "FireSim infrastructure execution is external: prepare an image with an explicit "
        "Deployment, then collect the operator's result package; installed runner selection "
        "and MERLIN_FIRESIM_RUNNER no longer launch work from Merlin"
    )


def prepare_image(
    elf: str | Path, *, output: str | Path, deployment: Deployment, inputs: dict[str, Path] | None = None
) -> PreparedRun:
    """Retain the exact ELF and caller-selected inputs in a portable request."""
    return prepare(
        Artifact.from_file(elf, kind="elf", role="entry"), output=output, deployment=deployment, inputs=inputs
    )


def collect_uart(prepared: PreparedRun | str | Path, *, results: str | Path) -> tuple[str, ExternalObservation]:
    """Retain hash-bound observations and return exactly one explicit UART stream.

    Matching file identity does not authenticate the external execution claim.
    The caller still owns parsing and independent numerical admission.
    """
    observed = collect(prepared, results=results)
    streams = [item for item in observed.artifacts if item["role"] == "uart"]
    if len(streams) != 1:
        raise FireSimRunnerError("external results must identify exactly one artifact with role 'uart'")
    if observed.reported_exit_code != 0:
        raise FireSimRunnerError(
            f"external execution reported exit code {observed.reported_exit_code}; observations retained"
        )
    return (observed.directory / streams[0]["path"]).read_text(encoding="utf-8"), observed
