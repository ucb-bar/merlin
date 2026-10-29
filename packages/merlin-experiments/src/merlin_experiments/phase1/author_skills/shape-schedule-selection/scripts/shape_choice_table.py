"""Per-shape-class choice table for your tuning knobs, with held-out classes and capacity guards.

    python shape_choice_table.py --knobs submission/TUNING_KNOBS.json --cases cases.json \\
        --results results.json [--facts rtl_facts/facts.json] [--tile EDGE] [--knob NAME ...] [--json]
    python shape_choice_table.py --shape-policy policy_report.json     # read an assess_shape_policy report

This is the planning step before ``merlin.transforms.shape_policy``: it finds which capacity guard
separates the winners and which boundary cases the frozen panel still lacks; ``assess_shape_policy``
then makes the guarded, incumbent-retaining selection from complete paired feedback, and
``--shape-policy`` prints its report per class. Formats: the skill's ``reference/choice-table.md``.

* ``--knobs``: your ``merlin.tuning_knobs.v1`` file. Its knobs and values are the candidate space; its
  ``settings_file`` (beside it) gives the current setting (the incumbent). ``--knob`` restricts the space.
* ``--cases``: a JSON list of cases with their geometry: ``m``/``k``/``n``, or ``operand_shapes``, or
  ``conv`` parameters. Optional ``class`` overrides the derived class; ``"model": true`` marks a layer
  of a model.
* ``--results``: normalized rows (``case``, ``settings``, ``cycles``, ``basis``, ``domain``,
  ``numerics``, or an ``observation`` in ``CompletedObservation`` form) or self-check responses
  (``agent_selfcheck.py --out`` files; knob-override responses carry their own settings).

Each case gets an ``ExplorationGeometry``; aliases are merged. Classes are derived from structure,
never from a number you choose: ``model-layer`` (marked), ``m1``, ``ragged`` (an extent not a multiple of
the array edge), else ``long-m``/``long-n``/``square`` by comparing M and N tile counts. Candidates are
scored as ``assess_shape_policy`` scores them, by the geometric mean of cycle ratios paired with the
incumbent, and ranked with ``rank_observed_metrics``. For every class a ``ValidationPartition`` holds it
out: the winner on the other classes is checked on it. Incomplete pairs are never zero; they stay
unranked. With RTL facts each case gets integer capacity facts for ``PolicyFacts.resources``
(``<set>_copies``, ``<set>_stores_needed`` for the LHS, RHS, both operands, the output and one M stripe
of output), and a separating fit is reported as the two ``ResourcePredicate`` guards of its cutoff,
with the boundary values the panel still needs: derived from capacity, not from where measurements
happened to fall.

Static screening and timing estimates are evidence for a decision, not a qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import product
from pathlib import Path

SCHEMA = "merlin.author_shape_choice_table.v1"
ROW_SCHEMA = "merlin.author_shape_results.v1"
BASES = ("measured_roi", "modeled_roi", "static_emitted_ir")
#: Fit flag -> the working set whose ``_copies`` / ``_stores_needed`` resources decide it.
PREDICATES = {"lhs_fits": "operand_lhs", "rhs_fits": "operand_rhs", "operands_fit": "operand_both",
              "output_fits": "accumulator_output", "m_stripe_fits": "accumulator_m_stripe"}
_MODEL_IDENTITY = ("model_sha256", "timing_mode", "timing_epoch")


def _library(path: str | None) -> None:
    root = Path(path) if path else Path("submission/compiler_library")
    if root.is_dir():
        sys.path.insert(0, str(root.resolve()))


def _read(path) -> object:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("result_index"):
            for parent in Path(path).resolve().parents:
                helper = parent / "qa_result_details.py"
                if helper.is_file():
                    namespace = {"__name__": "_public_result_details"}
                    exec(compile(helper.read_bytes(), str(helper), "exec"), namespace)
                    return namespace["hydrate"](document, workspace=parent)
        return document
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from None


# --------------------------------------------------------------------------------------------------
# Candidate space
# --------------------------------------------------------------------------------------------------


def knob_space(path, only=()) -> dict:
    """``{"knobs": {name: values}, "current": {name: value}}`` from a tuning-knob declaration."""
    doc = _read(path)
    knobs = doc.get("knobs") if isinstance(doc, dict) else None
    if not isinstance(knobs, dict) or not knobs:
        raise SystemExit(f"{path}: no knobs declared")
    values = {}
    for name, spec in sorted(knobs.items()):
        domain = spec.get("values") if isinstance(spec, dict) else None
        if not isinstance(domain, list) or not domain:
            raise SystemExit(f"{path}: knob {name} declares no values")
        values[name] = tuple(domain)
    unknown = sorted(set(only) - set(values))
    if unknown:
        raise SystemExit(f"--knob names undeclared knobs: {unknown}")
    current = {}
    settings = doc.get("settings_file")
    if isinstance(settings, str) and (Path(path).parent / settings).is_file():
        loaded = _read(Path(path).parent / settings)
        current = {k: v for k, v in loaded.items() if k in values} if isinstance(loaded, dict) else {}
    for name, domain in values.items():
        default = (knobs[name] or {}).get("default") if isinstance(knobs[name], dict) else None
        current.setdefault(name, default if default in domain else domain[0])
    return {"knobs": values, "current": current, "selected": tuple(only) or tuple(values)}


def candidate(settings: dict, space: dict) -> tuple | None:
    """The candidate a setting denotes: the selected knobs' values; ``None`` when another knob differs or
    the setting names a knob the declaration does not have (another compiler configuration)."""
    if set(settings) - set(space["knobs"]):
        return None
    full = {**space["current"], **settings}
    if any(full[k] != space["current"][k] for k in space["knobs"] if k not in space["selected"]):
        return None
    if any(full[k] not in space["knobs"][k] for k in space["selected"]):
        return None
    return tuple((k, full[k]) for k in space["selected"])


def label(key: tuple, space: dict) -> str:
    varied = [f"{k}={v}" for k, v in key if len(space["knobs"][k]) > 1]
    text = ",".join(varied) or "(only setting)"
    return text + (" (current)" if dict(key) == {k: space["current"][k] for k in space["selected"]} else "")


# --------------------------------------------------------------------------------------------------
# Cases, geometry and classes
# --------------------------------------------------------------------------------------------------


def _conv(c: dict) -> tuple[tuple, tuple]:
    n, ch, h, w = (int(c.get(k, 1)) for k in ("n", "c", "h", "w"))
    out, r, s = int(c["k"]), int(c.get("r", 1)), int(c.get("s", c.get("r", 1)))
    stride, pad, dil = int(c.get("stride", 1)), int(c.get("pad", 0)), int(c.get("dilation", 1))
    oh = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    ow = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1
    if min(n, ch, h, w, out, r, s, oh, ow) <= 0:
        raise SystemExit(f"conv case has no output: {c}")
    # The output shape is an operand of the geometry, so stride/padding are part of its identity.
    return ((n, ch, h, w), (out, ch, r, s), (n, out, oh, ow)), (n * oh * ow, ch * r * s, out)


def load_cases(path) -> list[dict]:
    doc = _read(path)
    rows = doc.get("cases") if isinstance(doc, dict) else doc
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path}: expected a list of cases")
    cases = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case"), str):
            raise SystemExit(f"{path}: every case needs a name: {row}")
        if isinstance(row.get("conv"), dict):
            shapes, gemm = _conv(row["conv"])
            family = row.get("family", "conv")
        elif all(k in row for k in ("m", "k", "n")):
            gemm = (int(row["m"]), int(row["k"]), int(row["n"]))
            shapes, family = ((gemm[0], gemm[1]), (gemm[1], gemm[2])), row.get("family", "dense")
        elif isinstance(row.get("operand_shapes"), list):
            shapes = tuple(tuple(int(d) for d in s) for s in row["operand_shapes"])
            family = row.get("family", "dense")
            first, second = shapes[0], shapes[1] if len(shapes) > 1 else None
            gemm = (first[-2], first[-1], second[-1]) if second and len(first) >= 2 else None
        else:
            raise SystemExit(f"case {row['case']}: give m/k/n, operand_shapes or conv")
        cases.append({"case": row["case"], "family": str(family), "shapes": shapes, "gemm": gemm,
                      "class": row.get("class"), "model": row.get("model") is True})
    return cases


def shape_class(case: dict, edge: int | None) -> str:
    """Structural class: marked model layer, one LHS row, ragged extent, else the longer tile axis."""
    if case["class"]:
        return str(case["class"])
    if case["model"]:
        return "model-layer"
    if case["gemm"] is None:
        return "other"
    m, k, n = case["gemm"]
    if m == 1:
        return "m1"
    if edge and any(d % edge for d in (m, k, n)):
        return "ragged"
    unit = edge or 1
    tm, tn = math.ceil(m / unit), math.ceil(n / unit)
    return "long-m" if tm > tn else "long-n" if tn > tm else "square"


# --------------------------------------------------------------------------------------------------
# Capacity facts
# --------------------------------------------------------------------------------------------------


def capacity(facts_path: str | None) -> dict | None:
    """Array edge and the operand/accumulator stores from RTL facts, or ``None`` without a facts file."""
    path = Path(facts_path) if facts_path else Path("rtl_facts/facts.json")
    if not path.is_file():
        if facts_path:
            raise SystemExit(f"no facts file at {path}")
        return None
    from merlin.targetgen import address_space as AS

    doc = _read(path)
    name = ((doc.get("inputs") or {}).get("target") if isinstance(doc, dict) else None) or "workspace"
    space = AS.derive_address_space(name, facts=doc)
    operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
    return {"source": str(path), "edge": space.array_cols, "operand": operand.store,
            "accumulator": accumulator.store,
            "unknown": [r.reason for r in (operand, accumulator) if r.store is None]}


def resources(gemm, caps) -> dict:
    """Integer capacity facts of one contraction, for ``PolicyFacts.resources``.

    Per working set (row-granular, ``address_space.working_set_rows``): ``<set>_copies`` is how many
    whole copies its store holds (0 = does not fit, 2 = room to double-buffer) and
    ``<set>_stores_needed`` how many stores it would take (2 or more = overflows). Both are
    nonnegative, so ``ResourcePredicate(name, at_least)`` can guard either side of a capacity cutoff.
    """
    if gemm is None or caps is None:
        return {}
    from merlin.targetgen.address_space import working_set_rows

    m, k, n = gemm
    op, acc, edge = caps["operand"], caps["accumulator"], caps["edge"]
    sets = {}
    if op is not None and op.capacity_rows():
        lhs, rhs = working_set_rows(op, (m, k)), working_set_rows(op, (k, n))
        both = None if None in (lhs, rhs) else lhs + rhs
        sets.update({"operand_lhs": (lhs, op), "operand_rhs": (rhs, op), "operand_both": (both, op)})
    if acc is not None and acc.capacity_rows():
        sets.update({"accumulator_output": (working_set_rows(acc, (m, n)), acc),
                     "accumulator_m_stripe": (working_set_rows(acc, (m, min(n, edge or n))), acc)})
    out = {}
    for name, (rows, store) in sets.items():
        if rows:
            out[name + "_copies"] = store.capacity_rows() // rows
            out[name + "_stores_needed"] = math.ceil(rows / store.capacity_rows())
    if edge and op is not None and acc is not None and op.elems_per_row() and acc.elems_per_row():
        from merlin.compile.capacity import _capacity_fit_tile

        out["resident_tiles_cover"] = _capacity_fit_tile(m, k, n, edge, op.capacity_rows() * op.elems_per_row(),
                                                         acc.capacity_rows() * acc.elems_per_row())[3]
    return out


def predicates(facts: dict) -> dict:
    """The fit/does-not-fit flags the table separates winners by, read from :func:`resources`."""
    return {flag: None if facts.get(name + "_copies") is None else facts[name + "_copies"] >= 1
            for flag, name in PREDICATES.items()}


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------


def _numerics(row: dict) -> bool:
    tiers = row.get("tiers")
    if isinstance(tiers, dict) and "L2" in tiers:
        tier = tiers["L2"]
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    return row.get("pass") is True or row.get("numerical_execution_passed") is True or row.get("status") == "pass"


def _response_rows(doc: dict, settings: dict | None) -> list[dict]:
    """Normalized rows of one self-check response: measured certifying-tier cycles, else the timing model."""
    variant = doc.get("variant") if isinstance(doc.get("variant"), dict) else {}
    if settings is None:
        settings = dict(variant.get("tuning_overrides") or {}) if variant.get("kind") == "knob_overrides" else {}
        if variant and variant.get("kind") != "knob_overrides" and variant.get("path"):
            raise SystemExit(f"response of {variant.get('path')} needs explicit settings (wrap it with settings)")
    rows = []
    for row in doc.get("per_capsule") or doc.get("cases") or ():
        if not isinstance(row, dict) or not (row.get("capsule") or row.get("name")):
            continue
        case, cycles, basis, domain = str(row.get("capsule") or row.get("name")), None, None, None
        timing = row.get("barrier_timing_identity") if isinstance(row.get("barrier_timing_identity"), dict) else {}
        performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
        if row.get("barrier_tier") == "L3" and row.get("pass") is True and _positive(row.get("barrier_cycles")):
            cycles, basis = row["barrier_cycles"], "measured_roi"
            domain = "measured:" + json.dumps([timing.get("engine") or "simulator", timing.get("simulator_sha256")])
        elif _positive(performance.get("estimated_whole_soc_cycles")):
            cycles, basis = performance["estimated_whole_soc_cycles"], "modeled_roi"
            domain = "model:" + json.dumps([performance.get(k) for k in _MODEL_IDENTITY])
        rows.append({"case": case, "settings": settings, "cycles": cycles, "basis": basis, "domain": domain,
                     "numerics": _numerics(row), "program_sha256": row.get("program_sha256")})
    return rows


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


#: The ``ExplorationIdentity`` fields that make up a ``shape_policy.PerformanceDomain``.
_DOMAIN_FIELDS = ("provider_sha256", "facts_sha256", "library_sha256", "scope_sha256", "basis")


def _row(row: dict, path) -> dict:
    """One normalized row; ``observation`` may be an ``asdict(CompletedObservation)``, read by its own rules."""
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else row
    identity = observation.get("identity") if isinstance(observation.get("identity"), dict) else None
    basis = identity.get("basis") if identity else row.get("basis", "modeled_roi")
    if basis not in BASES:
        raise SystemExit(f"{path}: basis must be one of {BASES}: {row}")
    if identity:
        domain = json.dumps([identity.get(k) for k in _DOMAIN_FIELDS] + [row.get("implementation_sha256")])
    else:
        domain = str(row.get("domain") or basis)
    numerics = observation.get("numerical_status", "pass" if row.get("numerics", True) in (True, "pass") else "fail")
    # CompletedObservation.eligible: passing numerics, complete timing over the complete ROI, known cycles.
    complete = observation.get("timing_status", "complete") == "complete" and observation.get("complete_roi", True)
    cycles = observation.get("cycles")
    return {"case": row["case"], "settings": dict(row.get("settings") or {}),
            "cycles": cycles if complete and _positive(cycles) else None, "basis": basis, "domain": domain,
            "numerics": numerics == "pass", "program_sha256": observation.get("program_sha256")}


def load_results(path) -> list[dict]:
    doc = _read(path)
    if isinstance(doc, dict) and isinstance(doc.get("rows"), list):
        if any(not isinstance(row, dict) or not isinstance(row.get("case"), str) for row in doc["rows"]):
            raise SystemExit(f"{path}: every row needs a case")
        return [_row(row, path) for row in doc["rows"]]
    entries = doc.get("responses") if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected {{'rows': [...]}}, {{'responses': [...]}} or a list of responses")
    rows = []
    for entry in entries:
        settings = None
        if isinstance(entry, dict) and "response" in entry:
            settings, entry = dict(entry.get("settings") or {}), entry["response"]
        if isinstance(entry, str):
            here = Path(entry)
            entry = _read(here if here.is_file() else Path(path).parent / entry)
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: unreadable response entry")
        rows.extend(_response_rows(entry, settings))
    return rows


# --------------------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------------------


def _domain(rows) -> tuple[str, str] | None:
    """The (basis, domain) with the most known observations; measured before modeled before static."""
    counts = {}
    for row in rows:
        if row["cycles"] is not None and row["basis"]:
            key = (row["basis"], row["domain"])
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda key: (counts[key], -BASES.index(key[0]), key[1]))


def _rank(scores: dict, basis: str, domain: str) -> dict:
    """Order candidates by their paired cycle ratio to the incumbent; unknown ratios stay unranked."""
    from merlin.transforms.pass_exploration import ObservedMetrics, rank_observed_metrics

    observations = tuple((name, ObservedMetrics(basis, domain, (("cycle_ratio", value),)))
                         for name, value in sorted(scores.items()))
    if not observations:
        return {"ranked": (), "unranked_unknown": ()}
    return rank_observed_metrics(observations, objectives=("cycle_ratio",))


def _ratios(table: dict, geometries, name: str, incumbent: str) -> list | None:
    """Per geometry, candidate cycles over incumbent cycles; ``None`` when any pair is incomplete."""
    pairs = [(table.get((g.identity, name)), table.get((g.identity, incumbent))) for g in geometries]
    return None if any(None in pair for pair in pairs) else [x / y for x, y in pairs]


def _score(table: dict, geometries, names, incumbent: str) -> dict:
    """Geometric-mean paired ratio per candidate: the score ``shape_policy.assess_shape_policy`` trains on."""
    scores = {}
    for name in names:
        ratios = _ratios(table, geometries, name, incumbent)
        scores[name] = None if not ratios else math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return scores


def choice_table(space, cases, rows, caps, edge) -> dict:
    from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition

    by_geometry, aliases = {}, {}
    for case in cases:
        geometry = ExplorationGeometry(case["family"], case["shapes"])
        if geometry.identity in by_geometry:
            aliases.setdefault(by_geometry[geometry.identity]["case"], []).append(case["case"])
            continue
        facts = resources(case["gemm"], caps)
        by_geometry[geometry.identity] = {**case, "geometry": geometry, "class_name": shape_class(case, edge),
                                          "resources": facts, "predicates": predicates(facts)}
    case_geometry = {c["case"]: g for g, c in by_geometry.items()}
    for primary, names in aliases.items():
        for name in names:
            case_geometry[name] = case_geometry[primary]
    keys, skipped, unmatched, failing = {}, 0, set(), []
    for row in rows:
        key = candidate(row["settings"], space)
        if key is None:
            skipped += 1
            continue
        if row["case"] not in case_geometry:
            unmatched.add(row["case"])
            continue
        keys.setdefault(key, label(key, space))
        row["candidate"], row["geometry"] = keys[key], case_geometry[row["case"]]
        if not row["numerics"]:
            failing.append((row["case"], keys[key]))
    usable = [r for r in rows if "candidate" in r]
    chosen = _domain([r for r in usable if r["numerics"]])
    if chosen is None:
        raise SystemExit("no result carries comparable cycles for a declared candidate")
    basis, domain = chosen
    observed, programs = {}, {}
    for row in usable:
        if row["numerics"] and row["cycles"] is not None and (row["basis"], row["domain"]) == chosen:
            observed.setdefault((row["geometry"], row["candidate"]), []).append(row["cycles"])
        if row.get("program_sha256"):
            programs.setdefault(row["geometry"], {}).setdefault(row["program_sha256"], set()).add(row["candidate"])
    table = {slot: statistics.median(values) for slot, values in observed.items()}  # repeated requests
    current = label(tuple((k, space["current"][k]) for k in space["selected"]), space)
    names = sorted({*keys.values(), current})
    classes = {}
    for g in by_geometry.values():
        classes.setdefault(g["class_name"], []).append(g["geometry"])
    class_rows = []
    for name, held in sorted(classes.items()):
        ranking = _rank(_score(table, held, names, current), basis, domain)
        ranked = list(ranking["ranked"])
        best = ranked[0] if ranked else None
        others = [g for c, gs in classes.items() if c != name for g in gs]
        transfer = None
        if others:
            partition = ValidationPartition(tuple(others), tuple(held))
            trained = _rank(_score(table, partition.training, names, current), basis, domain)["ranked"]
            winner = trained[0][0] if trained else None
            values = dict(ranked)
            held_ratios = _ratios(table, partition.held_out, winner, current) if winner else None
            transfer = {"partition_sha256": partition.manifest()["sha256"], "winner": winner,
                        "holds": bool(best and winner in values and values[winner] == best[1]),
                        "held_out_max_ratio": max(held_ratios) if held_ratios else None,
                        "held_out_position": [c for c, _ in ranked].index(winner) if winner in values else None}
        totals = {c: sum(table[(g.identity, c)] for g in held) for c in names
                  if all((g.identity, c) in table for g in held)}
        class_rows.append({
            "class": name, "cases": [by_geometry[g.identity]["case"] for g in held],
            "best": best[0] if best else None, "best_ratio": best[1][0] if best else None,
            "best_cycles": totals.get(best[0]) if best else None, "current_cycles": totals.get(current),
            "ranked": [[c, v[0]] for c, v in ranked],
            "unknown": sorted(c for c, _ in ranking["unranked_unknown"]), "trained_on_other_classes": transfer,
        })
    per_case = []
    for gid, g in by_geometry.items():
        values = {c: table[(gid, c)] for c in names if (gid, c) in table}
        low = min(values.values()) if values else None
        per_case.append({"case": g["case"], "aliases": aliases.get(g["case"], []), "class": g["class_name"],
                         "gemm": g["gemm"], "predicates": g["predicates"], "resources": g["resources"],
                         "winners": sorted(c for c, v in values.items() if v == low), "cycles": low,
                         "distinct_programs": len(programs.get(gid, {})) or None, "measured": len(values)})
    return {"schema": SCHEMA, "basis": basis, "domain": domain, "edge": edge,
            "facts": caps["source"] if caps else None, "candidates": names, "current": current,
            "classes": class_rows, "cases": per_case,
            "separating_predicates": separations(per_case, keys, space, current),
            "set_aside": {"other_knob_settings": skipped, "unmatched_cases": sorted(unmatched),
                          "failing_numerics": sorted(failing),
                          "other_domains": sum(1 for r in usable if r["cycles"] is not None
                                               and (r["basis"], r["domain"]) != chosen)},
            "performance_qualification": False}


def _guards(working_set: str, incumbent: str) -> dict:
    """Both sides of one capacity cutoff as ``ResourcePredicate`` guards over :func:`resources` facts.

    ``assess_shape_policy`` keeps a class incumbent until its frozen panel holds each guard's exact
    boundary value and the adjacent outside value; ``boundary_values`` lists both.
    """
    from dataclasses import asdict

    from merlin.transforms.pipeline_policy import ResourcePredicate

    sides = {"when_true": ResourcePredicate(working_set + "_copies", 1),
             "when_false": ResourcePredicate(working_set + "_stores_needed", 2)}
    try:
        from merlin.transforms.shape_policy import ShapeScheduleClass
    except ImportError:  # a library without shape_policy: the guards stay ordinary policy predicates
        ShapeScheduleClass = None
    for side, guard in sides.items():
        if ShapeScheduleClass is not None:
            ShapeScheduleClass(side, (guard,), incumbent)  # refuses a guard shape_policy would not accept
    return {side: {"predicate": asdict(guard), "boundary_values": [guard.at_least, guard.at_least - 1]}
            for side, guard in sides.items()}


def separations(per_case, keys, space, incumbent) -> list[dict]:
    """Capacity predicates whose value determines a knob's winning value across the measured cases."""
    by_label = {text: dict(key) for key, text in keys.items()}
    found = []
    for knob in space["selected"]:
        winners = {}
        for row in per_case:
            values = {by_label[c][knob] for c in row["winners"] if c in by_label}
            if len(values) == 1:
                winners[row["case"]] = (values.pop(), row["predicates"])
        if len({value for value, _ in winners.values()}) < 2:
            continue
        for name, working_set in PREDICATES.items():
            mapping, support, consistent = {}, {True: 0, False: 0}, True
            for value, facts in winners.values():
                fact = facts.get(name)
                if fact is None or mapping.setdefault(fact, value) != value:
                    consistent = False
                    break
                support[fact] += 1
            if not (consistent and len(mapping) == 2):
                continue
            guards = _guards(working_set, incumbent)
            for side in guards.values():
                seen = {row["resources"].get(side["predicate"]["resource"]) for row in per_case}
                side["missing_boundary_values"] = [v for v in side["boundary_values"] if v not in seen]
            # Support per side: one case on a side is a hypothesis to test at the boundary.
            found.append({"knob": knob, "predicate": name, "when_true": mapping[True],
                          "when_false": mapping[False], "cases": len(winners),
                          "cases_true": support[True], "cases_false": support[False], "guards": guards})
    return found


# --------------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------------


def _pct(ratio):
    return f"{100.0 * (ratio - 1):+.1f}%" if ratio is not None else "-"


def render(report: dict) -> str:
    lines = [f"Choice table ({report['basis']}, {report['domain']}); array edge {report['edge'] or 'unknown'}"
             f"{' from ' + report['facts'] if report['facts'] else ''}",
             f"current setting: {report['current'].removesuffix(' (current)')}", ""]
    width = max([len("best in class"), *(len(str(r["best"] or "unknown")) for r in report["classes"])])
    header = f"{'class':<12} {'cases':>5}  {'best in class':<{width}} {'cycles':>12} {'vs current':>10}  holds-out"
    lines += [header, "-" * len(header)]
    for row in report["classes"]:
        transfer = row["trained_on_other_classes"]
        held = "-"
        if transfer is not None:
            worst = transfer["held_out_max_ratio"]
            regress = f"; regresses {_pct(worst)} on a case here" if worst is not None and worst > 1 else ""
            held = (f"{'yes' if transfer['holds'] else 'no'} (other classes pick "
                    f"{transfer['winner'] or 'nothing known'}{regress})")
        cycles = f"{row['best_cycles']:,.0f}" if row["best_cycles"] is not None else "unknown"
        best = str(row["best"] or "unknown")
        lines.append(f"{row['class']:<12} {len(row['cases']):>5}  {best:<{width}} {cycles:>12} "
                     f"{_pct(row['best_ratio']):>10}  {held}")
        if row["unknown"]:
            lines.append(f"{'':<20}unranked (a case lacks a passing pair with the current setting): "
                         f"{', '.join(row['unknown'])}")
    lines += ["", "Per case (capacity predicates: L=lhs R=rhs O=both operands A=output S=one M stripe fit; "
              "T=resident tiles)"]
    for row in sorted(report["cases"], key=lambda r: (r["class"], r["case"])):
        p = row["predicates"]
        flags = "".join(("-" if p.get(k) is None else "y" if p[k] else "n") for k in PREDICATES)
        cover = row["resources"].get("resident_tiles_cover", "-")
        gemm = "x".join(str(d) for d in row["gemm"]) if row["gemm"] else "-"
        win = ", ".join(row["winners"]) or "unknown"
        lines.append(f"  {row['case']:<36} {row['class']:<11} {gemm:<16} LROAS={flags} T={cover:<4} -> {win}")
    lines += ["", "Capacity predicates that separate the per-case winners"]
    for s in report["separating_predicates"]:
        lines.append(f"  {s['knob']}: {s['when_true']} where {s['predicate']} ({s['cases_true']} cases), "
                     f"{s['when_false']} where not ({s['cases_false']} cases)")
        for side, guard in s["guards"].items():
            p = guard["predicate"]
            missing = " and ".join(map(str, guard["missing_boundary_values"]))
            need = f"needs a case with {p['resource']} = {missing}" if missing else "covered"
            lines.append(f"    {side}: ResourcePredicate({p['resource']!r}, {p['at_least']}); boundary {need}")
    if not report["separating_predicates"]:
        lines.append("  none: keep a per-class choice, add cases at the class boundary, or derive another fact")
    aside = report["set_aside"]
    lines += ["", f"set aside: {aside['other_knob_settings']} rows with other knobs changed, "
              f"{len(aside['failing_numerics'])} failing numerics, {aside['other_domains']} in other timing domains, "
              f"unmatched cases {aside['unmatched_cases'] or 'none'}",
              "Evidence for a decision, not a qualification: confirm with ab-library-pass before changing a default."]
    return "\n".join(lines)


def render_shape_policy(report: dict) -> str:
    """Per class of an ``assess_shape_policy(...).report``: what was selected and what keeps the incumbent."""
    if not isinstance(report, dict) or report.get("schema") != "merlin.shape_schedule_policy.v1":
        raise SystemExit("--shape-policy expects the report of merlin.transforms.shape_policy.assess_shape_policy")
    lines = [f"shape policy {str(report.get('policy_sha256'))[:12]} (max regression ratio "
             f"{report.get('max_regression_ratio')})", ""]
    for group in report.get("classes", ()):
        lines.append(f"{group['shape_class']}: {group['status']}; selected {group['selected']} "
                     f"(incumbent {group['incumbent']})")
        for alt in group.get("alternatives", ()):
            score = alt.get("training_geometric_mean_ratio")
            ratios = alt.get("matched_cycle_ratios") or []
            why = []
            if alt.get("missing_geometries"):
                why.append(f"{len(alt['missing_geometries'])} panel geometries without feedback")
            for _index, side, edge, outside in alt.get("missing_boundaries") or ():
                why.append(f"{side} cutoff needs cases at {edge} and {outside}")
            if alt.get("reasons"):
                why.append(f"{len(alt['reasons'])} incomplete pairs")
            if alt.get("mixed_performance_domains"):
                why.append("mixed performance domains")
            if ratios and max(ratios) > report.get("max_regression_ratio", 1.0):
                why.append(f"regresses {_pct(max(ratios))} on a matched case")
            lines.append(f"  {alt['alternative']:<28} {'supported' if alt['supported'] else 'not supported':<14} "
                         f"training {_pct(score) if score is not None else 'unknown'}"
                         f"{'; ' + '; '.join(why) if why else ''}")
    winners = [w for w in report.get("exact_source_winners", ()) if w.get("winner")]
    lines += ["", f"exact-source winners (this source only, not a guard): {len(winners)} of "
              f"{len(report.get('exact_source_winners', ()))} geometries"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="Formats: the shape-schedule-selection skill, reference/choice-table.md.")
    ap.add_argument("--shape-policy", default=None,
                    help="instead: print an assess_shape_policy report (JSON of policy.report) per class")
    ap.add_argument("--knobs", help="your TUNING_KNOBS.json (merlin.tuning_knobs.v1)")
    ap.add_argument("--cases", help="JSON list of cases with geometry")
    ap.add_argument("--results", help="normalized rows, CompletedObservation rows or self-check responses (JSON)")
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument("--tile", type=int, default=None, help="array edge when no facts are available")
    ap.add_argument("--knob", action="append", default=[], help="vary only these knobs (repeatable)")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    if a.shape_policy:
        report = _read(a.shape_policy)
        print(json.dumps(report, indent=1) if a.json else render_shape_policy(report))
        return 0
    if not (a.knobs and a.cases and a.results):
        ap.error("--knobs, --cases and --results are required (or --shape-policy REPORT)")
    _library(a.library)
    space = knob_space(a.knobs, tuple(a.knob))
    if math.prod(len(space["knobs"][k]) for k in space["selected"]) > 256:
        ap.error("more than 256 candidates: restrict with --knob")
    caps = capacity(a.facts)
    edge = a.tile or (caps or {}).get("edge")
    report = choice_table(space, load_cases(a.cases), load_results(a.results), caps, edge)
    report["declared_candidates"] = math.prod(len(space["knobs"][k]) for k in space["selected"])
    report["unmeasured_candidates"] = sorted(
        label(tuple(zip(space["selected"], values)), space)
        for values in product(*(space["knobs"][k] for k in space["selected"]))
        if label(tuple(zip(space["selected"], values)), space) not in report["candidates"])
    print(json.dumps(report, indent=1, default=str) if a.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
