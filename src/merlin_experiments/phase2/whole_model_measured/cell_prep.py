"""A CELL run's objective config, composed from a measured run's own config.

    groups = form_cell_groups(objective_forms, [1])                     # or name the groups
    held = held_out_rows(objective_forms, groups, {"other model": (capsule, its_forms)})
    document = cell_objective_config(loop_config, cell_id="c1", groups=groups, held_out=held, ...)

A cell run is this mode pointed at a cell machine (:mod:`.cells`, measured by
:class:`.group_capsules.GroupProgramMeasurer`) instead of a whole-model machine.  Everything a cell
measurement is attributed through comes from the LOOP's own config, never restated: the builder, the
store base, the environment, and the package arm's build options are the loop's CERTIFIER's (the
emulator the loop certifies on), so a cell number and the loop's certification are the same machine,
header, corpus binding and instruction rule.  What the cell adds is data: its groups, the held-out
groups of the same form, the collateral representatives of every other form, and (optionally) a form
capsule screen run before any emulator time.

HELD-OUT GROUPS ARE FOUND BY FORM, never listed: a held-out capture's group is a member when its form
(:func:`.forms.matching_groups`, the capsule corpus's own key) equals one of the cell groups' forms --
one per distinct (form, shape), since repeats of one problem add emulator time and no transfer evidence.
Nothing here names a model, a group or a shape.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import cells as CELLS
from . import forms as FORMS

HELD_OUT_RULE = "the same form (the capsule corpus's form key) in each held-out capture, one per distinct shape"
CELL_MEASURER = "merlin_experiments.phase2.whole_model_measured.group_capsules:cell_measurer"


def _distinct(groups: Sequence[int], forms: Sequence[Mapping[str, Any]]) -> list[int]:
    shapes = {str(r["group"]): (r["form_text"], repr(sorted((r.get("shape") or {}).items()))) for r in forms}
    seen: set[Any] = set()
    out = []
    for g in groups:
        key = shapes.get(str(g), (str(g), ""))
        if key not in seen:
            seen.add(key)
            out.append(int(g))
    return out


def held_out_rows(
    objective_forms: Sequence[Mapping[str, Any]],
    groups: Sequence[int],
    captures: Mapping[str, tuple[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """``[{label, model_capsule, groups}]``: each held-out capture's groups of the cell groups' forms.
    ``captures`` maps a label to ``(model_capsule, forms)``; a capture with no group of the cell's forms
    contributes no row (it has nothing to transfer to)."""
    text = {str(r["group"]): r["form_text"] for r in objective_forms}
    missing = [g for g in groups if str(g) not in text]
    if missing:
        raise CELLS.CellError(f"cell group(s) {missing} are not groups of the objective capture")
    rows = []
    for label, (capsule, forms) in captures.items():
        matched = sorted({g for want in groups for g in FORMS.matching_groups(text[str(want)], forms)})
        matched = _distinct(matched, forms)
        if matched:
            rows.append({"label": str(label), "model_capsule": str(capsule), "groups": matched})
    return rows


def form_cell_groups(objective_forms: Sequence[Mapping[str, Any]], anchors: Sequence[int]) -> list[int]:
    """A cell over whole FORMS: for the form of each anchor group, one group per distinct shape (the first
    in program order).  Members that share a form and a shape differ only in their weights and inputs,
    so timing one of each prices the form's every distinct problem without paying for repeats."""
    text = {str(r["group"]): r["form_text"] for r in objective_forms}
    missing = [g for g in anchors if str(g) not in text]
    if missing:
        raise CELLS.CellError(f"anchor group(s) {missing} are not groups of the objective capture")
    wanted = {text[str(g)] for g in anchors}
    ordered = [
        int(r["group"]) for r in sorted(objective_forms, key=lambda r: int(r["group"])) if r["form_text"] in wanted
    ]
    return _distinct(ordered, objective_forms)


def cell_machine_spec(
    *,
    target: str,
    registry_machine: str,
    cell_id: str,
    model_capsule: str,
    groups: Sequence[int],
    held_out: Sequence[Mapping[str, Any]],
    reference_build_options: Mapping[str, Any],
    max_parallel: int = 2,
    max_cycles: int | None = None,
    collateral: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The cell machine a cell run's screen names: the real measurer, the cell, and the reference arm's
    recipe (the bar's build options, measured by the same path)."""
    timing: dict[str, Any] = {"registry_machine": registry_machine, "max_parallel": int(max_parallel)}
    if max_cycles:
        timing["max_cycles"] = int(max_cycles)
    return {
        "kind": "cell",
        "target": target,
        "measurer": CELL_MEASURER,
        "timing": timing,
        "reference_build_options": {k: v for k, v in dict(reference_build_options).items() if k != "prohibited_roles"},
        "cell": {
            "id": cell_id,
            "model_capsule": str(model_capsule),
            "groups": [int(g) for g in groups],
            "held_out": [dict(r) for r in held_out],
            "held_out_rule": HELD_OUT_RULE,
            **({CELLS.COLLATERAL: dict(collateral)} if collateral else {}),
            **({CELLS.BASELINE: dict(baseline)} if baseline else {}),
            **({CELLS.DIAGNOSTICS: dict(diagnostics)} if diagnostics else {}),
        },
    }


def cell_diagnostics(
    groups: Sequence[int],
    *,
    model_capsule: str | Path,
    target: str,
    functional_model: Mapping[str, Any] | None,
    measured: Mapping[str, Sequence[tuple[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The cell's ``diagnostics`` block: each cell group's derived roofline (:mod:`.roofline`, confronted
    with any ``measured`` ``{group: [(label, cycles)]}``) and the functional model its programs'
    efficiency census runs on. Diagnostic only: a roofline that cannot be derived says why."""
    from . import roofline as R

    machine = R.roofline_machine(target)
    shapes = R.group_shapes(model_capsule, target=target)
    rooflines = {}
    for group in groups:
        shape = shapes.get(str(group))
        if shape is not None:
            rooflines[str(group)] = R.confront(R.group_roofline(shape, machine), (measured or {}).get(str(group)) or ())
    return {
        "functional_model": dict(functional_model) if functional_model else None,
        "rooflines": rooflines,
        "machine": {k: machine[k] for k in ("array_rows", "array_cols", "provenance", "unresolved")},
    }


def screen_check(loop_check: Mapping[str, Any], *, capsules: str, label: str) -> dict[str, Any]:
    """The loop's own pre-measure check, restricted to ``capsules`` (a form capsule screen)."""
    check = copy.deepcopy(dict(loop_check))
    check.update(capsules=capsules, label=label, required=True)
    return check


def cell_objective_config(
    loop: Mapping[str, Any],
    *,
    machine: Mapping[str, Any],
    reference: str | Path,
    notice: str,
    screen: Mapping[str, Any] | None = None,
    slots: int = 2,
    timeout_seconds: float = 7200,
) -> dict[str, Any]:
    """The loop's objective config with its screen replaced by the cell ``machine`` and no certifier:
    the package arm is the loop's certifier recipe, the store is the loop's store base's ``cells``, and
    the instruction rule is the loop's -- its roles AND the sealed Phase 0 policy they are held to,
    refused here when the loop carries none it can enforce."""
    from . import config as CFG

    certifier = dict((loop.get("certifier") or {}).get("build_options") or {})
    if not certifier.get("machine") or not certifier.get("header"):
        raise CELLS.CellError("the loop config's certifier names no machine or header for the package arm")
    roles = list(loop.get("prohibited_instruction_roles") or ())
    sealed = loop.get(CFG.SEALED_POLICY)
    document = {
        "schema": loop["schema"],
        "builder": loop.get("builder"),
        "store": str(Path(str(loop["store"])) / "cells"),
        "python": loop.get("python"),
        "environment": dict(loop.get("environment") or {}),
        "prohibited_instruction_roles": roles,
        **({CFG.SEALED_POLICY: copy.deepcopy(dict(sealed))} if sealed is not None else {}),
        "screen": {
            "machine": dict(machine),
            "build_options": certifier,
            "reference": str(reference),
            "slots": int(slots),
            "max_pending": 4,
            "timeout_seconds": float(timeout_seconds),
        },
        "repeats_on_best": 1,
        "retain": loop.get("retain"),
        "stop_at_bar": False,
        "plateau_hours": loop.get("plateau_hours", 6),
        "plateau_min_sessions": loop.get("plateau_min_sessions", 8),
        "harness_notices": [notice],
    }
    if screen is not None:
        document["pre_measure_check"] = dict(screen)
    try:
        CFG.check_policy(document)
    except CFG.ConfigError as exc:
        raise CELLS.CellError(f"the cell cannot inherit the loop's instruction rule: {exc}") from exc
    return document


NOTICE = (
    "CELL MODE. This run's objective is the sum of the cycles of cell {cell} ({groups} of the model), each "
    "built as its own one-group program on the model's own inputs at model shapes and timed alone on the "
    "elaborated-RTL emulator. Correctness is exact per group, and each group's WHOLE program is scanned for "
    "prohibited instructions. A cell group answered by the library or host code refuses the candidate. The "
    "bar is the reference arm's cycles for the same group(s), measured the same way. Held-out models' groups "
    "of the same form are built and graded the same way: each must be correct; their cycles are reported and "
    "never summed. One representative group of every OTHER form of the model is also built and timed: a "
    "candidate that makes any of them slower than its baseline, wrong, or no longer answered by the package "
    "is refused. A cell number ranks; a best is confirmed on the board and offered to the whole-model loop."
)


def prepare(
    loop: Mapping[str, Any],
    *,
    target: str,
    cell_id: str,
    groups: Sequence[int],
    held_out_capsules: Sequence[str | Path] = (),
    collateral_share: Mapping[str, Any] | None = None,
    baseline_package: Path | None = None,
    collateral_tolerance: float = 0.01,
    out: Path,
    max_cycles: int | None = None,
    functional_model: Mapping[str, Any] | None = None,
    screen_capsules: str | None = None,
) -> dict[str, Any]:
    """Compose a cell run's objective config from ``loop``: find the held-out groups by form, measure the
    collateral baseline on ``baseline_package`` (a VERIFIED package, e.g. the loop's best -- an unverified
    seed that already broke a form would make the broken form its own bar) along with the cell's own
    groups on the same package (the ``baseline`` a candidate's emulator bound is derived from), measure
    the cell's reference arm once, and return ``{"config", "reference", "collateral", "baseline",
    "diagnostics", "held_out"}`` (all also under ``out``). ``diagnostics`` is each cell group's derived
    roofline (confronted with the baseline's own cycles) and ``functional_model``, the machine a
    candidate's efficiency census runs on -- never a gate; a diagnostics failure is recorded, and the
    cell is prepared without one.  ``screen_capsules`` (comma-separated) restricts the loop's own
    pre-measure check to the cell's form capsules (:func:`screen_check`), run before any emulator time."""
    import json

    screen = None
    if screen_capsules:
        if not loop.get("pre_measure_check"):
            raise CELLS.CellError("a form capsule screen restricts the loop's pre-measure check, and it declares none")
        screen = screen_check(
            loop["pre_measure_check"], capsules=str(screen_capsules), label=f"cell {cell_id} form capsule screen"
        )

    certifier = dict((loop.get("certifier") or {}).get("build_options") or {})
    capsule = str(certifier["model_capsule"])
    objective = FORMS.statement_forms(capsule, target=target)
    held: list[dict[str, Any]] = []
    if held_out_capsules:
        captures = {Path(c).name: (str(c), FORMS.statement_forms(c, target=target)) for c in held_out_capsules}
        held = held_out_rows(objective, groups, captures)
    reference_options = {k: v for k, v in certifier.items() if k != "prohibited_roles"}
    collateral = baseline = None
    if collateral_share is not None and baseline_package is None:
        raise CELLS.CellError("a collateral baseline is measured on a named, verified baseline package")
    spec = cell_machine_spec(
        target=target,
        registry_machine=str(certifier["machine"]),
        cell_id=cell_id,
        model_capsule=capsule,
        groups=groups,
        held_out=held,
        reference_build_options=reference_options,
        max_cycles=max_cycles,
    )
    if baseline_package is not None:
        # The cell's OWN groups on the same verified package: a candidate's programs are bounded from
        # these (cells.objective_bound), so a deadlocked candidate is refused at a few times its
        # baseline instead of at the flat bound.
        baseline = CELLS.measure_cell_baseline(
            {**spec, "build_options": certifier},
            baseline_package=Path(baseline_package),
            target=target,
            out=Path(out) / "cell_baseline",
        )
    if collateral_share is not None:
        representatives = CELLS.collateral_representatives(objective, groups, collateral_share)
        collateral = CELLS.measure_collateral_baseline(
            {**spec, "build_options": certifier},
            representatives,
            baseline_package=Path(baseline_package),
            target=target,
            out=Path(out) / "collateral_baseline",
            tolerance=collateral_tolerance,
        )
    measured = {str(r["group"]): [("baseline", r["cycles"])] for r in (baseline or {}).get("groups") or ()}
    try:
        diagnostics = cell_diagnostics(
            groups, model_capsule=capsule, target=target, functional_model=functional_model, measured=measured
        )
    except Exception as error:  # noqa: BLE001 -- diagnostic only: recorded, never stops a cell being prepared
        diagnostics = {"refusal": f"{type(error).__name__}: {error}"}
    machine = cell_machine_spec(
        target=target,
        registry_machine=str(certifier["machine"]),
        cell_id=cell_id,
        model_capsule=capsule,
        groups=groups,
        held_out=held,
        reference_build_options=reference_options,
        max_cycles=max_cycles,
        collateral=collateral,
        baseline=baseline,
        diagnostics=diagnostics if diagnostics.get("rooflines") else None,
    )
    reference = CELLS.reference_cell(machine, target=target, out=Path(out) / "reference")
    notice = NOTICE.format(cell=cell_id, groups=", ".join(f"g{g}" for g in groups))
    config = cell_objective_config(
        loop, machine=machine, reference=Path(out) / "reference" / "result.json", notice=notice, screen=screen
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "cell_objective_config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    return {
        "config": config,
        "reference": reference,
        "collateral": collateral,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "held_out": held,
    }


__all__ = [
    "CELL_MEASURER",
    "cell_diagnostics",
    "HELD_OUT_RULE",
    "cell_machine_spec",
    "cell_objective_config",
    "form_cell_groups",
    "held_out_rows",
    "prepare",
    "screen_check",
]
