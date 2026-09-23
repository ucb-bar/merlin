"""The store-path readout gate: a whole-model pass that left absorbable work on the host.

A model capsule's verdict says the program computed the right numbers. It has never said how much of
the program ran where, and a whole-model emission can be numerically perfect while asking the unit
for nothing but the contraction -- which is the measured shape of the slowest deployable ResNet-50
this repo has: 53 convolutions, not one stage of readout requested, the accelerator idle 96.84% of
the window, and a verdict that called it a pass.

This applies :mod:`merlin.perf.epilogue_oracle` to each model row from the artifacts the run already
left, records the two numbers on the row, and -- only at the phase its declaration puts it in --
fails a row whose emission DECIDEDLY left licensed readout on the host.

WHAT COUNTS AS A DECIDED FAILURE, and why the list is short. ``gap`` is decided: the target licenses
the stage, the grouping admits it at that site, and the buffer did not ask. Everything else is
``incomplete`` and blocks nothing at any phase -- including the case that matters most here, a
capture whose activation scale is computed at run time so no site admits anything. That capture
produces a buffer with no readout on any command, which is byte-for-byte what a lazy compiler
produces, and a two-valued gate would fail the compiler for its input.

Nothing here names a target; the target is a parameter and every licensed stage comes from its own
derived readout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import epilogue_oracle as EO
from merlin.perf import gate_phase

#: The name this gate is declared under in ``merlin/contract/gate_phases.yaml``. There is no default
#: phase; a rename that orphans the declaration raises rather than silently never blocking.
GATE = "epilogue_store_path"
PLANE = "epilogue_store_path"
CATEGORY = "ABSORBABLE_WORK_LEFT_ON_HOST"
#: The one DECIDED failure. `incomplete` is a status, not a phase, and never blocks.
FAILING = (EO.VERDICT_GAP,)


def _buffer(suite_runs: Path, name: str) -> dict[str, Any] | None:
    path = suite_runs / name / "generated" / "command_buffer.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _source(capsule: Mapping[str, Any]) -> Path | None:
    directory = capsule.get("__dir__")
    if not directory:
        return None
    path = Path(directory) / str(capsule.get("interface_mlir") or "capsule.interface.mlir")
    return path if path.is_file() else None


def assess(command_buffer: Mapping[str, Any], source: Path, target: str) -> dict[str, Any]:
    """The gate's verdict for one emission, or an ``incomplete`` naming what stopped it."""
    from merlin.common import mlir_query as mq

    try:
        module = mq.parse(source.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 -- an unparseable source decides nothing about a compiler
        return EO.incomplete(f"the capsule's source could not be parsed: {type(error).__name__}", (), (), {})
    try:
        return EO.gap(command_buffer, module, target)
    except Exception as error:  # noqa: BLE001 -- likewise; the gate reports, it never crashes a grade
        return EO.incomplete(f"the comparison could not be made: {type(error).__name__}: {error}", (), (), {})


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    """The row-sized record. The two numbers, the verdict, and never a score it did not earn."""
    return {
        "schema": EO.SCHEMA,
        "status": report["verdict"],
        "why": report["why"],
        "epilogue_share_on_store_path": report["epilogue_share_on_store_path"],
        "silent_fallbacks": list(report["silent_fallbacks"]),
        "n_silent_fallbacks": len(report["silent_fallbacks"]),
        "stages": report["stages"],
        "licensed_stages": report["licensed_stages"],
        "sites_with_omissions": len(report["omitted_sites"]),
    }


def apply_gate(
    results: Sequence[dict[str, Any]],
    capsules: Sequence[Mapping[str, Any]],
    suite_runs: Path,
    *,
    target: str,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    """Attach the store-path verdict to every model row, and fail it at the declared phase.

    Returns the rows it judged (not only the ones it failed), so a caller can report a gate that ran
    and found nothing separately from one that did not run -- the distinction this repo keeps paying
    for when it is missing.
    """
    phase = gate_phase.configured_phase(GATE) if phase is None else phase
    by_name = {str(capsule.get("name")): capsule for capsule in capsules}
    judged: list[dict[str, Any]] = []
    for result in results:
        if result.get("kind") != "model":
            continue
        name = str(result.get("capsule") or result.get("name") or "")
        capsule, buffer = by_name.get(name), _buffer(suite_runs, name)
        source = _source(capsule) if capsule else None
        if buffer is None or source is None:
            missing = "its command buffer" if buffer is None else "its source MLIR"
            report = EO.incomplete(f"the run left no artifact to read: {missing} is absent", (), (), {})
        else:
            report = assess(buffer, source, target)
        record = summarize(report)
        result["epilogue_store_path"] = record
        judged.append({"capsule": name, **record})
        if gate_phase.blocks(phase, record["status"], failing=FAILING):
            result["status"] = "fail"
            result["failure"] = {
                "plane": PLANE,
                "category": CATEGORY,
                "detail": (
                    f"{record['sites_with_omissions']} store-path site(s) carry readout "
                    f"{target} is derived to apply and this emission did not ask for "
                    f"({record['stages']['asked']} of {record['stages']['admissible']} licensed stages requested)"
                ),
            }
    return judged


def score_rows(judged: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The score-level aggregate, with ``None`` wherever nothing was measured.

    A model corpus whose every row came back ``incomplete`` reports ``None`` for the share, not 0.0:
    the difference between "no readout was requested" and "nothing could be compared" is the whole
    reason this gate ships at ``report`` first.
    """
    decided = [row for row in judged if row["epilogue_share_on_store_path"] is not None]
    return {
        "n_model_rows": len(judged),
        "n_decided": len(decided),
        "n_incomplete": len(judged) - len(decided),
        "share": (
            round(sum(row["epilogue_share_on_store_path"] for row in decided) / len(decided), 6) if decided else None
        ),
        "silent_fallbacks": sum(row["n_silent_fallbacks"] for row in decided),
        "by_capsule": {row["capsule"]: row["status"] for row in judged},
    }
