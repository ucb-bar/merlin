"""A shape panel: one ordinary self-check of a small representative set per shape class of a family.

``python agent_selfcheck.py --panel FAMILY [--sim gsim|spike] [--wait-budget N]`` sends one ordinary
request whose capsule selector is ``panel:FAMILY``. This module answers it inside the ordinary
self-check child, with the same broker, budgets and grader:

1. it reads the declarations of the selected public roster (core, plus the scalability suite for a
   Spike panel) and keeps the cases of FAMILY (:func:`.shape_classes.describe`; a family is derived
   from each declaration, never from a name);
2. per shape class (:data:`.shape_classes.CLASSES`, computed against the tile edge the target's RTL
   facts state) it picks one representative: a case whose newest grade passed, preferring a case this
   request can measure with cycle evidence and one that already has same-kind cycle evidence to compare
   with, then the largest declared work, then the name (deterministic); the ``model_layer`` class takes
   a model-derived case not already picked for another class (a held-out layer);
3. it grades them as ordinary subsets -- the core selection through the ordinary self-check, a Spike
   panel's scalability cases through the ordinary scalability grade -- and adds a ``panel`` section:
   per class the case, its cycles with their evidence kind, the fewest earlier cycles of that case of
   the SAME kind in this run (``qa/perf_history.jsonl``), the ratio and a verdict such as
   ``regressed 1.43x vs best (square)``.

EVIDENCE IS NEVER MIXED. Functional Spike counts are not target cycle estimates.
Complete provisional model observations can guide exploration when model, facts,
scope and capsule identities match. They remain provisional and cannot establish
physical speedups or calibration accuracy. UNKNOWN observations stay unranked.
The panel defaults to gSIM for the core roster; use ``--sim spike`` to include
large scalability cases and use the selected cycle model without large gSIM runs.

FEEDS A SHAPE POLICY. Each graded class also carries a ``shape_policy`` record in the vocabulary of
:mod:`merlin.transforms.shape_policy` (``ExplorationGeometry`` fields and ``CompletedObservation``
fields, with the identity pins the row states). The pins that describe the author's own pipeline
session are listed as ``author_supplies`` and never invented; :func:`completed_observation` builds
the typed observation once they are given. Only cycle evidence is ``timing_status: complete``.

Advisory only: no gate, quota or threshold; the panel's per-case rows are ordinary self-check rows.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import tempfile
from pathlib import Path

from . import case_declarations, shape_classes

PREFIX = "panel:"
SCHEMA = "merlin.selfcheck_shape_panel.v1"
RULE = (
    "Compare the same case and evidence kind; modeled comparisons also require matching model/facts/scope "
    "identities and complete accounting. Provisional estimates guide exploration but do not establish "
    "physical speedups. Functional Spike counts and UNKNOWN timing remain unranked. Compare affected "
    "shape classes before changing their defaults."
)


def requested_family(capsules) -> str | None:
    """The family named by a ``panel:FAMILY`` selector, or None for any other selector."""
    if not isinstance(capsules, str) or not capsules.startswith(PREFIX):
        return None
    return shape_classes.canonical_family(capsules[len(PREFIX) :])


def candidates(rosters, *, family: str, edge: int) -> list[dict]:
    """Every described case of ``family`` in ``rosters`` (``[(suite, [(name, declaration)])]``)."""
    out, seen = [], set()
    for suite, rows in rosters:
        for name, declaration in rows:
            description = shape_classes.describe(declaration)
            if name in seen or family not in shape_classes.families(description):
                continue
            seen.add(name)
            out.append(
                {
                    "case": name,
                    "suite": suite,
                    "extents": description["extents"],
                    "work": description["work"],
                    "classes": shape_classes.classes(description, edge),
                    "operand_shapes": [
                        list(t["shape"]) for t in declaration.get("inputs") or () if isinstance(t, dict)
                        and isinstance(t.get("shape"), list) and all(type(d) is int and d > 0 for d in t["shape"])
                    ],
                }
            )
    return out


def available_families(rosters) -> list[str]:
    found: set[str] = set()
    for _, rows in rosters:
        for _, declaration in rows:
            found.update(shape_classes.families(shape_classes.describe(declaration)))
    return sorted(found)


def select(cases, *, status: dict, best: dict, sim: str, measurable=None) -> dict:
    """``{class: chosen case row or None}`` (module docstring, step 2).

    ``status`` is ``{case: newest loop-tier status}`` (empty: no grade yet, every case is eligible);
    ``best`` is :func:`.shape_classes.history_best`; ``measurable(row)`` says whether this request can
    produce cycle evidence for the case (default: a gSIM panel's core cases).
    """
    eligible = [c for c in cases if not status or status.get(c["case"]) == "pass"]
    measurable = measurable or (lambda row: sim == "gsim" and row["suite"] == "core")

    def prior(row) -> bool:
        return any((row["case"], kind) in best for kind in shape_classes.CYCLE_ACCURATE)

    def key(row):
        return (not measurable(row), not prior(row), -row["work"], row["case"])

    picks: dict[str, dict | None] = {}
    used: set[str] = set()
    for klass in shape_classes.CLASSES:
        pool = sorted((c for c in eligible if klass in c["classes"]), key=key)
        fresh = [c for c in pool if c["case"] not in used]
        # A held-out model layer is a case no other class already stands for.
        chosen = (fresh or ([] if klass == "model_layer" else pool) or [None])[0]
        picks[klass] = chosen
        if chosen is not None:
            used.add(chosen["case"])
    return picks


def evidence(row: dict) -> tuple:
    """``(cycles, kind, label)`` of one graded row, or ``(None, None, None)`` without cycles."""
    from .performance_brief import _evidence

    if not isinstance(row, dict):
        return None, None, None
    found = _evidence(row)
    if found is not None:
        return found[0], shape_classes.evidence_kind(found[2]), found[2]
    cycles = row.get("barrier_cycles")
    if type(cycles) in (int, float) and cycles > 0 and row.get("pass") is True:
        label = f"spike count: {row.get('barrier_tier') or 'loop'} tier simulator cycles (not cycle-accurate)"
        return cycles, "spike_count", label
    return None, None, None


#: ``ExplorationIdentity`` fields a self-check row cannot state: they describe the author's own pipeline
#: session (its source, ABI, inputs, reference, provider, ROI and context pins), so the author supplies them.
AUTHOR_PINS = ("source_sha256", "abi_sha256", "inputs_sha256", "reference_sha256", "library_sha256",
               "provider_sha256", "roi_sha256", "context_sha256", "artifact_format")
#: ``ExplorationIdentity.basis`` of each evidence kind (a Spike count states none; only cycle evidence is complete).
BASIS = {"gsim_measured": "measured_roi", "calibrated_model": "modeled_roi", "provisional_model": "modeled_roi"}
FEED = (
    "Each class's `shape_policy` record maps onto merlin.transforms.shape_policy inputs: "
    "ExplorationGeometry(**geometry) and CompletedObservation(ExplorationIdentity(**identity, <your session's "
    "author_supplies pins>), **observation), paired per alternative in ShapeCaseFeedback. Only cycle evidence "
    "carries timing_status 'complete', including identity-bound complete provisional modeled estimates. "
    "Functional Spike counts and incomplete estimates stay unknown and cannot replace the incumbent."
)


def _pin(value) -> str | None:
    hexadecimal = isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    return value if hexadecimal else None


def policy_record(pick: dict, row: dict, kind, cycles, *, family: str) -> dict:
    """One class's observation in the vocabulary of :mod:`merlin.transforms.shape_policy` (see :data:`FEED`)."""
    import hashlib

    performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
    tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
    loop = tiers.get("L2")
    loop = loop.get("status") if isinstance(loop, dict) else loop
    passed = row.get("pass") is True or row.get("status") == "pass" or loop == "pass"
    failed = row.get("pass") is False or row.get("status") == "fail" or loop == "fail"
    timed = kind in shape_classes.RANKABLE and type(cycles) in (int, float) and cycles > 0
    accounted = performance.get("complete_event_accounting") is True
    complete = kind == "gsim_measured" or (
        kind in ("calibrated_model", "provisional_model") and accounted
        and shape_classes.model_domain(performance) is not None)
    return {
        "geometry": {"family": family, "operand_shapes": pick.get("operand_shapes") or []},
        "identity": {"basis": BASIS.get(kind), "facts_sha256": _pin(performance.get("facts_sha256")),
                     "scope_sha256": _pin(performance.get("scope_sha256"))},
        "author_supplies": list(AUTHOR_PINS),
        "observation": {
            "program_sha256": _pin(row.get("execution_digest")) or _pin(performance.get("elf_sha256")),
            "feedback_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest(),
            "numerical_status": "pass" if passed else "fail" if failed else "unknown",
            "timing_status": "complete" if timed and complete else "unknown",
            "complete_roi": bool(timed and complete),
            "cycles": int(cycles) if timed and complete else None,
        },
    }


def completed_observation(record: dict, **pins):
    """``(ExplorationGeometry, CompletedObservation)`` from a panel record plus the author's own pins.

    Raises when a pin the record leaves to the author is missing or the record carries none of its own:
    an observation is never minted with an invented identity.
    """
    from merlin.transforms.evaluated_exploration import CompletedObservation, ExplorationIdentity
    from merlin.transforms.pass_exploration import ExplorationGeometry

    identity = {**{k: v for k, v in record["identity"].items() if v is not None}, **pins}
    missing = [f for f in (*AUTHOR_PINS, "basis", "facts_sha256", "scope_sha256") if identity.get(f) is None]
    if missing:
        raise ValueError(f"shape-policy observation lacks identity pins {missing}")
    geometry = record["geometry"]
    return (
        ExplorationGeometry(geometry["family"], tuple(tuple(shape) for shape in geometry["operand_shapes"])),
        CompletedObservation(ExplorationIdentity(**identity), **record["observation"]),
    )


def section(picks, rows, *, workspace, family, edge, edge_source, sim, refused=None, notes=()) -> dict:
    """The response's ``panel`` block from the picks and the graded rows (``{case: row}``)."""
    best = shape_classes.history_best(workspace)
    out, summary = {}, []
    for klass in shape_classes.CLASSES:
        pick = picks.get(klass)
        if pick is None:
            out[klass] = {"case": None, "verdict": f"no passing {family} case of this class in the roster"}
            summary.append(f"{klass}: no case")
            continue
        name = pick["case"]
        entry = {"case": name, "suite": pick["suite"], "extents": pick["extents"]}
        if name in (refused or {}):
            entry.update(
                cycles=None,
                evidence_kind=None,
                verdict=f"{shape_classes.NO_EVIDENCE} ({klass}): {refused[name]}",
            )
        elif rows.get(name) is None:
            entry.update(cycles=None, evidence_kind=None, verdict=f"not graded in this request ({klass})")
        else:
            cycles, kind, label = evidence(rows[name])
            entry.update(cycles=cycles, evidence_kind=kind, evidence=label)
            entry.update(shape_classes.compare(name, cycles, kind, best, klass,
                domain=shape_classes.model_domain(rows[name].get("performance"))))
            entry["shape_policy"] = policy_record(pick, rows[name], kind, cycles, family=family)
        out[klass] = entry
        summary.append(entry["verdict"])
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "family": family,
        "sim": sim,
        "tile_edge": edge,
        "tile_edge_source": edge_source,
        "classes": out,
        "summary": "; ".join(summary),
        "rule": RULE,
        "shape_policy_feed": FEED,
        **({"notes": list(notes)} if notes else {}),
    }


def _replace(argv, flag: str, value: str) -> list[str]:
    out, skip, found = [], False, False
    for token in argv:
        if skip:
            skip = False
        elif token == flag:
            skip, found = True, True
            out += [flag, value]
        elif token.startswith(flag + "="):
            found = True
            out += [flag, value]
        else:
            out.append(token)
    return out if found else out + [flag, value]


def _admit(names, stack, timeout: int):
    """``(admitted, refused, timeout)`` under the joint campaign's own per-case gSIM policy, if any."""
    from .. import joint_control

    if joint_control.selected() is None:
        return list(names), {}, timeout
    from ..joint_gsim_measurement import requested as measurement_request

    if measurement_request() is not None:
        return [], {n: "an operator measurement request names one case" for n in names}, timeout
    admitted, refused, budgets = [], {}, [timeout]
    correctness = joint_control.correctness_capsules()
    for name in names:
        manager = joint_control.physical_correctness if name in correctness else joint_control.physical_witness
        try:
            budgets.append(int(stack.enter_context(manager(name)) or timeout))
            admitted.append(name)
        except ValueError as error:
            refused[name] = f"gSIM not admitted by the campaign policy ({error})"
    return admitted, refused, min(budgets)


def _core(run, argv, names, out: Path, **kwargs) -> dict | None:
    """The ordinary self-check of ``names`` (inside the wrappers this request already entered)."""
    argv = _replace(_replace(argv, "--capsules", ",".join(names)), "--out", str(out))
    with contextlib.redirect_stdout(io.StringIO()):
        inspect.unwrap(run)(argv, **kwargs)
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _scalability(a, names, out: Path, *, context) -> dict | None:
    """The ordinary Spike scalability grade of ``names``."""
    from . import public_scalability

    args = copy.copy(a)
    args.suite, args.sim, args.tiers, args.capsules, args.out = "scalability", "spike", "", ",".join(names), str(out)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            public_scalability.cli(args, context=context)
        except (OSError, ValueError, RuntimeError) as error:
            return {"error": f"scalability grade refused ({type(error).__name__})", "all_pass": False}
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rosters(core_root, context, sim: str, notes: list):
    rosters = [("core", case_declarations.roster(core_root))]
    if sim == "gsim":
        notes.append(
            "gSIM panel: scalability cases are Spike-only by suite policy and are not picked; "
            "`--panel FAMILY --sim spike` grades them; complete identity-bound model estimates can guide "
            "provisional comparisons, while functional Spike counts remain unranked."
        )
        return rosters
    try:
        from .public_scalability import selected

        chosen = selected(context)
    except (OSError, ValueError, KeyError, TypeError) as error:
        chosen = None
        notes.append(f"scalability roster unavailable ({type(error).__name__})")
    if chosen is not None:
        root, record = chosen
        case_declarations.register(root / "cases")
        names = set(record.get("cases") or ())
        rosters.append(("scalability", [(n, d) for n, d in case_declarations.roster(root / "cases") if n in names]))
    return rosters


def _emit(a, document: dict) -> None:
    print(json.dumps(document, indent=2))
    if a.out:
        from .selfcheck import _atomic_json

        _atomic_json(Path(a.out), document)


def _merge(core_doc, scale_doc, *, sim, refused) -> tuple[dict, dict]:
    out = dict(core_doc or {"sim": sim, "per_capsule": [], "n_capsules": 0, "n_passed": 0, "all_pass": not refused})
    rows = {row.get("capsule"): row for row in out.get("per_capsule") or () if isinstance(row, dict)}
    if scale_doc is not None:
        cases = [row for row in scale_doc.get("cases") or () if isinstance(row, dict)]
        rows.update({row.get("capsule"): row for row in cases})
        out["scalability"] = {k: v for k, v in scale_doc.items() if k != "cases"}
        out["scalability"]["cases"] = [
            {k: row.get(k) for k in ("capsule", "status", "tiers", "failure_plane", "failure_category")}
            for row in cases
        ]
        out["all_pass"] = bool(out.get("all_pass")) and scale_doc.get("all_pass") is True
    if refused:
        out["all_pass"] = False
    return out, rows


def cli(argv, a, *, context, root=None, contract=None, run=None) -> int:
    """Answer a ``panel:FAMILY`` request (module docstring); ``run`` is the ordinary self-check main."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    capsules_root, contract = root, a.contract if a.contract is not None else contract
    family = requested_family(a.capsules)
    sim = "spike" if a.sim == "spike" else "gsim"
    if capsules_root is None:
        from .selfcheck import _public_capsules

        capsules_root = _public_capsules(context)
    case_declarations.register(capsules_root, target=context.target)
    edge, edge_source = shape_classes.tile_edge(context.target)
    notes: list[str] = []
    rosters = _rosters(capsules_root, context, sim, notes)
    if edge is None:
        _emit(a, {"error": f"shape panel unavailable: {edge_source}", "all_pass": False})
        return 2
    workspace = Path.cwd()
    status, best = shape_classes.history_status(workspace), shape_classes.history_best(workspace)
    if not status:
        notes.append("no completed grade recorded a status yet; every declared case was eligible")
    picks = select(candidates(rosters, family=family, edge=edge), status=status, best=best, sim=sim)
    chosen = {p["case"]: p for p in picks.values() if p is not None}
    if not chosen:
        _emit(
            a,
            {
                "error": f"no passing case of family {family!r} in the selected roster",
                "available_families": available_families(rosters),
                "all_pass": False,
            },
        )
        return 2
    core = sorted(n for n, p in chosen.items() if p["suite"] == "core")
    scale = sorted(n for n, p in chosen.items() if p["suite"] == "scalability")
    refused: dict = {}
    kwargs = {"context": context, "capsules_root": capsules_root, "contract": contract}
    with tempfile.TemporaryDirectory(prefix="selfcheck_panel_") as scratch:
        core_doc = scale_doc = None
        with contextlib.ExitStack() as stack:  # gSIM reservations return before the Spike grade
            if core and sim == "gsim":
                core, refused, timeout = _admit(core, stack, a.timeout)
                argv = _replace(list(argv), "--timeout", str(timeout))
            if core:
                core_doc = _core(run, argv, core, Path(scratch) / "core.json", **kwargs)
        if scale:
            scale_doc = _scalability(a, scale, Path(scratch) / "scalability.json", context=context)
    out, rows = _merge(core_doc, scale_doc, sim=sim, refused=refused)
    if core and core_doc is None:
        out.update(all_pass=False, error="the core panel selection returned no readable result")
    out["panel"] = section(
        picks, rows, workspace=workspace, family=family, edge=edge, edge_source=edge_source, sim=a.sim or sim,
        refused=refused, notes=notes,
    )
    _emit(a, out)
    return 0 if out.get("all_pass") else 1


__all__ = [
    "AUTHOR_PINS", "PREFIX", "RULE", "SCHEMA", "candidates", "cli", "completed_observation", "evidence",
    "policy_record", "requested_family", "section", "select",
]
