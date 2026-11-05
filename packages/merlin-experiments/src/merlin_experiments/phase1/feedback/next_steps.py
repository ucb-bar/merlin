"""One ranked next step per high-headroom case, for the summaries the author actually reads.

Measured on one run: the author read self-check responses 301 times through small scripts that print
only status, failure and cases; 4 of those outputs ever showed ``optimization_next``. What it did read
was the self-check's printed output, ``qa/latest_grade_summary.txt`` and ``qa/round_brief.md``, whose
"Largest headroom" lines carried the right diagnosis with the wrong fix: a fence-bound product was told
to "try" a pooled-convolution readout cache (its entry names a readout, as the fence words did) and a
pass its compiler already imported. This module picks, per case, ONE choice and the command that
applies it, for those summaries (the JSON fields stay as they are):

* the bottleneck in plain words, from what the evidence entry already carries: a stream that drains
  the device after nearly every output tile (:mod:`.fence_overlap`), operands re-read beyond what the
  local capacity allows (:mod:`.capacity_use`), the busiest modeled resource, else the roofline bound;
* candidates, ranked: for a fence-bound stream only mechanisms that address ordering (the
  synchronization, overlap, issue and movement facets, whose catalogue entry names a fence, barrier,
  drain or epoch, or the generic fence mechanisms) -- any catalogue mechanism so described is picked
  up without editing this file, and one whose workload requirement the case is known to meet (a
  convolution mechanism for a convolution) ranks ahead of the generic ones; otherwise the facets of
  the bottleneck, mechanisms naming the busy unit first. Mechanisms with an ``unsupported``
  requirement are dropped, and workload requirements the selection left ``unknown`` are scoped from
  the case's own declaration or public inventory row (a two-dimensional product without a window is
  not a convolution, a single operation has no sibling products). Among mechanisms that name the
  bottleneck, capability-confirmed ones come first; still-unconfirmed requirements are named;
* source call sites are observed separately from imports; untried applicable alternatives lead,
  while existing calls retain tuning/coverage advice. Neither imports nor calls prove execution;
* a chain case's composition route (:mod:`.composition_next`) ranks first when the host-compute audit
  flagged the case, when execution is the busiest resource, or when no mechanism remains; a model or
  program case's unrequested ``source_composition.json`` option ranks first (an owner-checked request);
* the command: the ab-library-pass ``apply_library_pass.py`` route into ``variants/`` followed by the
  variant's self-check (:func:`.library_pass.ab_command`), a knob value or a request file in a variant;
  and where to read more.

Readiness is per case (:mod:`.family_rollup`): every passing case belongs to a bundle of passing cases
sharing its operation family and shape class, whose representative gets a step at once -- its mechanism
choice with the siblings as the held-out check (``--panel``), or, without cycles yet, the command that
times it -- and each family with cases left gets a finish-correctness step naming the library route that
builds them fast from the start. :func:`_ranked` orders all of them by value (large and connected cases
first, then headroom); no step waits for another case or family. The informational family line
(correct / total, timed) closes the block.

Advisory only: nothing here grades, gates or applies, and no speedup is claimed. Only public workspace
files and the response's own evidence are read; no private reference or measurement. No ``re``.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import mechanism_hints

SCHEMA = "merlin.selfcheck_next_steps.v1"
KEY = "next_steps"
#: Steps in the printed block, and the byte budget of one line and of the whole block.
LIMIT = 5
LINE_BYTES = 1000
BLOCK_BYTES = 5600
HEADER = ("next steps (advisory; one choice per case or bundle by value, large and connected headroom first; a passing "
          "case's performance step never waits for other cases; detail: optimization_next):")
#: How a step is judged: estimates (complete timing model, else the calibrated fast estimate) decide.
DECIDE_ON_ESTIMATES = ("decide on cycle estimates (the complete timing model where it ran, else the calibrated "
                       "fast estimate, each labelled with its evidence kind): prefer the change the estimate "
                       "favors and revert if a later measurement disagrees; Spike numerics and final required "
                       "gSIM gates unchanged")
#: Step kinds that are not a mechanism choice: a passing representative to time, a family's cases left.
OTHER_KINDS = ("measure", "finish_correctness")
#: Facets that can remove a per-tile drain, and the words a catalogue entry uses for one.
FENCE_FACETS = ("synchronization", "overlap", "issue", "movement")
FENCE_TERMS = ("fence", "barrier", "drain", "epoch")
RESTREAM_WORDS = ("resident", "retain", "retention", "reuse", "panel")
_EXCLUDED_FACETS = frozenset({"legality", "feedback", "compiler_control"})
_OFF_VALUES = (False, None, "off", "none", "baseline")
_FACET_CLASS = {
    mechanism_hints._STORAGE: "local storage",
    mechanism_hints._TRANSFER: "memory transfers",
    mechanism_hints._ISSUE: "instruction issue",
    mechanism_hints._EXECUTE: "accelerator execution",
    mechanism_hints._HOST: "host CPU work",
}


# --- context --------------------------------------------------------------------------------------------


class Context:
    """Per-response inputs read once: selection, the author's imports and knobs, roles and flags."""

    def __init__(self, workspace, document=None, rows=None):
        from . import composition_next
        from .library_pass import inventory_script, skill_root

        self.workspace = Path(workspace) if workspace is not None else Path.cwd()
        from .result_details import load_row

        rows = [load_row(row, workspace=self.workspace) for row in rows or ()]
        self.selection = mechanism_hints.with_source_evidence(
            mechanism_hints.load_selection(self.workspace), self.workspace)
        self.inventory = inventory_script()
        self.imports = (self.selection.get('source_imports', {}) if self.selection
                        else self.inventory.imported_modules(self.workspace / "submission"))
        self.knobs = _knobs(self.workspace / "submission")
        from ..workload_roles import workspace_roles

        self.roles = workspace_roles(self.workspace) or {}
        self.members = _members(self.workspace)
        self.flagged = composition_next.host_flagged(document or {}, rows or ())
        self.skills = skill_root(self.workspace)
        self.rows = {
            str(row.get("capsule") or row.get("name")): row
            for row in rows or ()
            if isinstance(row, dict) and (row.get("capsule") or row.get("name"))
        }
        from . import idle_performance, idle_tuning

        self.tuning_progress = idle_tuning.matching_progress(
            idle_performance._load(self.workspace / idle_tuning.LEDGER),
            {**(document or {}), "cases": list(rows or ())})

    def imported(self, api: str) -> bool:
        module, _, symbol = str(api or "").partition(":")
        return bool(module) and bool(self.inventory.users([module], [symbol] if symbol else [], self.imports))


def _read_json(path: Path, limit: int = 4 << 20):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _knobs(submission: Path) -> dict:
    """``{knob: {values, mechanism, current}}`` from the author's ``TUNING_KNOBS.json`` and its settings."""
    declared = _read_json(submission / "TUNING_KNOBS.json")
    if not isinstance(declared, dict) or not isinstance(declared.get("knobs"), dict):
        return {}
    settings_name = str(declared.get("settings_file") or "tuning_settings.json")
    settings = _read_json(submission / settings_name) if "/" not in settings_name else None
    settings = settings if isinstance(settings, dict) else {}
    out = {}
    for name, knob in declared["knobs"].items():
        if isinstance(knob, dict) and isinstance(knob.get("values"), list) and knob["values"]:
            out[name] = {**knob, "current": settings.get(name, knob["values"][0])}
    return out


def _members(workspace: Path) -> dict:
    record = _read_json(Path(workspace) / "PUBLIC_CAPABILITIES.json", 16 << 20)
    members = record.get("members") if isinstance(record, dict) else None
    return members if isinstance(members, dict) else {}


# --- the case -------------------------------------------------------------------------------------------


def _ranks(tensors) -> list[int]:
    return [len(t["shape"]) for t in tensors or () if isinstance(t, dict) and isinstance(t.get("shape"), list)]


def workload_states(name: str, ctx: Context, signals: dict) -> dict[str, str]:
    """Workload traits the case's own public files settle (``supported``/``unsupported``); others absent."""
    from .. import operation_optimizations
    from . import case_declarations

    declaration = case_declarations.load(name, ctx.workspace)
    member = ctx.members.get(name) if isinstance(ctx.members.get(name), dict) else {}
    states: dict[str, str] = {}
    if isinstance(declaration, dict):
        families = operation_optimizations.families_of(declaration)
        attrs = (declaration.get("operation") or {}).get("attributes") or {}
        single = not operation_optimizations._nodes(declaration) and not isinstance(attrs.get("matmuls"), list)
        single = single and declaration.get("kind") != "model"
        states["workload.convolution"] = "supported" if "convolution" in families else "unsupported"
        dense = bool({"contraction", "matrix_vector"} & set(families))
        if families:
            states["workload.dense_contraction"] = "supported" if dense else "unsupported"
        if single:
            epilogue = "elementwise_epilogue" in families or "producer_epilogue" in signals
            states["workload.pointwise_epilogue"] = "supported" if epilogue else "unsupported"
            for trait in ("workload.shared_input", "workload.multiple_live_outputs", "workload.graph_fanout"):
                states[trait] = "unsupported"
        tensors = declaration.get("inputs")
    else:
        tensors = member.get("input_tensors")
        ranks = _ranks(tensors)
        if ranks and max(ranks) <= 2:
            states["workload.convolution"] = "unsupported"  # no window fits two-dimensional operands
    if "shared_input_siblings" in signals:
        for trait in ("workload.shared_input", "workload.multiple_live_outputs", "workload.graph_fanout"):
            states[trait] = "supported"
    if any(isinstance(t, dict) and t.get("role") == "weight" for t in tensors or ()):
        states["workload.immutable_weights"] = "supported"
    return states


def _plain(entry: dict) -> tuple[str, str, tuple, tuple]:
    """``(bottleneck in plain words, mode, facets, words)`` from one evidence entry."""
    fence = entry.get("fence_overlap") if isinstance(entry.get("fence_overlap"), dict) else {}
    if fence.get("fence_per_tile"):
        text = f"the device drains after nearly every output tile ({fence.get('fences')} fences"
        if fence.get("output_tiles"):
            text += f" for {fence['output_tiles']} output tiles"
        text += "), so loads cannot overlap compute"
        wait = (fence.get("dynamic") or {}).get("fence_wait_share_of_cycles")
        if isinstance(wait, int | float):
            text += f"; {round(100 * wait, 1)}% of modeled cycles wait on them"
        return text, "fence", FENCE_FACETS, FENCE_TERMS
    use = entry.get("capacity_use") if isinstance(entry.get("capacity_use"), dict) else {}
    reads = use.get("operand_reads") if isinstance(use.get("operand_reads"), dict) else {}
    ratio, allowed = reads.get("ratio"), reads.get("capacity_bound_ratio")
    if isinstance(ratio, int | float) and isinstance(allowed, int | float) and ratio > allowed * 1.2:
        text = f"operands are re-read {ratio}x the compulsory bytes (local capacity allows ~{allowed}x)"
        return text, "restream", mechanism_hints._TRANSFER, RESTREAM_WORDS
    dominant = entry.get("dominant_modeled_resource")
    name = dominant.get("name") if isinstance(dominant, dict) else dominant if isinstance(dominant, str) else None
    facets = mechanism_hints.resource_facets(name)
    if facets is not None:
        activity = dominant.get("modeled_activity") if isinstance(dominant, dict) else None
        unit = mechanism_hints.storage_unit(name)
        text = f"{_FACET_CLASS.get(facets, 'a resource')} is the busiest modeled resource ({name}"
        text += f", {round(100 * activity)}% activity)" if isinstance(activity, int | float) else ")"
        return text, "resource", facets, ((unit,) if unit else ())
    bound = entry.get("bound_by")
    facets = mechanism_hints.bound_facets(bound)
    if facets is not None:
        gap = entry.get("gap_ratio") or entry.get("gap_to_roofline")
        return f"held by the {bound} bound ({gap}x above it)", "bound", facets, ()
    return "", "", (), ()


def _names(row: dict, terms) -> int:
    """How many of a catalogue entry's fields name one of ``terms`` (a whole word starting with it)."""
    fields = [row.get("id"), row.get("mechanism"), row.get("legality"), row.get("evidence")]
    count = 0
    for text in fields:
        words = "".join(c if c.isalnum() else " " for c in str(text or "").lower()).split()
        count += any(word.startswith(term) for word in words for term in terms)
    return count


def mechanism_candidates(selection, facets, words, mode: str, states: dict) -> list[dict]:
    """Catalogue rows for a bottleneck, ranked; unsupported and wrong-workload rows are dropped."""
    out = []
    for index, row in enumerate((selection or {}).get("all_mechanisms") or ()):
        if not isinstance(row, dict) or row.get("status") == "unsupported" or row.get("family") in _EXCLUDED_FACETS:
            continue
        if row.get("family") not in facets:
            continue
        requirements = dict(row.get("requirements") or {})
        for key, state in requirements.items():
            if key.startswith("workload.") and state != "supported" and key in states:
                requirements[key] = states[key]
        if "unsupported" in requirements.values():
            continue
        named = _names(row, words) if words else 0
        direct = row.get("id") in mechanism_hints.FENCE_MECHANISMS
        if mode == "fence" and not (named or direct):
            continue  # an entry that only names a readout or a cache does not remove a drain
        unconfirmed = sorted(k for k, v in requirements.items() if not k.startswith("workload.") and v != "supported")
        workload = sorted(k for k, v in requirements.items() if k.startswith("workload.") and v != "supported")
        # Workload traits this case is known to have make a mechanism specific to it (e.g. a convolution
        # epoch mechanism for a convolution); the catalogue's own generic fence mechanisms come next, in
        # their stated order, so stream compaction (which keeps every dynamic fence) comes last.
        specific = sum(1 for k, v in requirements.items() if k.startswith("workload.") and v == "supported")
        order = mechanism_hints.FENCE_MECHANISMS
        generic = order.index(row["id"]) if row.get("id") in order else len(order)
        observed = mechanism_hints.opportunity({**row, 'requirements': requirements}, selection or {})
        from .operand_traffic import CONV_ORDER, region, route_kind

        traffic_order = 0
        if mode == "operand_traffic" and region(states) == "convolution":
            traffic_order = CONV_ORDER.get(route_kind(row, convolution=True), 2)
        elif mode == "operand_traffic":
            traffic_order = {"array_retention": 0, "wide_loads": 1}.get(route_kind(row), 2)
        rank = (
            traffic_order,
            bool(observed['source_call_sites']),
            not named,
            bool(unconfirmed),
            bool(workload),
            -specific,
            -named,
            generic,
            facets.index(row["family"]),
            index,
        )
        out.append({"row": row, "rank": rank, "unconfirmed": unconfirmed, "applies_if": workload,
                    "opportunity": observed})
    out.sort(key=lambda item: item["rank"])
    return out


def _mode_off(row: dict, knobs: dict):
    """``(knob, current, next value)`` when a declared knob names this mechanism and holds its baseline."""
    keys = {str(row.get("id")), str(row.get("api")), f"{row.get('api')}:{row.get('symbol')}", str(row.get("symbol"))}
    for name, knob in knobs.items():
        if str(knob.get("mechanism")) not in keys:
            continue
        values, current = knob["values"], knob.get("current")
        if current == values[0] or current in _OFF_VALUES:
            following = [v for v in values if v != current and v not in _OFF_VALUES]
            if following:
                return name, current, following[0]
    return None


def _suite(name: str, ctx: Context, default: str) -> str:
    member = ctx.members.get(name) if isinstance(ctx.members.get(name), dict) else {}
    suite = member.get("selfcheck_suite")
    return suite if suite in ("core", "scalability", "programs") else default


def _check(name: str, variant: str, suite: str) -> str:
    route = {"scalability": "--suite scalability", "programs": "--suite programs"}.get(suite, "--sim spike")
    return f"python agent_selfcheck.py --submission variants/{variant} --capsules {name} {route} --wait-budget 5"


def choose(entry: dict, ctx: Context, *, suite: str = "core", reuse: bool = True) -> dict | None:
    """The one next step for a ranked evidence entry, or None when nothing fits (module docstring).

    A convolution with observed read amplification leads with its selected reuse route; when its own
    census also shows a drain per output tile, the fence step (``reuse=False``) rides along as
    ``fence_step`` so both are offered, neither suppressed."""
    from . import composition_next
    from .library_pass import _variant, ab_command

    name = entry.get("test_case")
    from ..workload_roles import PRIMARY

    if not isinstance(name, str) or not name or ctx.roles.get(name) not in PRIMARY:
        return None
    bottleneck, mode, facets, words = _plain(entry)
    signals, inventory, declaration = composition_next._signals(name, ctx.workspace)
    states = workload_states(name, ctx, signals)
    from . import operand_traffic

    selection = mechanism_hints.for_case(ctx.selection, name, ctx.workspace) if ctx.selection else None
    traffic = operand_traffic.signal(entry, states, selection) if reuse else None
    conv_traffic = bool(traffic) and traffic.get("region") == "convolution"
    fence_step = choose(entry, ctx, suite=suite, reuse=False) if conv_traffic and mode == "fence" else None
    if traffic:
        bottleneck, mode = traffic["reading"], "operand_traffic"
        facets, words = ("storage", "movement", "overlap"), RESTREAM_WORDS
        if conv_traffic:  # the shared convolution consumer is catalogued with its drain/epoch family
            facets = facets + ("synchronization",)
    suite = _suite(name, ctx, suite)
    ranked = []
    for item in mechanism_candidates(selection, facets, words, mode, states) if mode else ():
        row = item["row"]
        api = f"{row.get('api')}:{row.get('symbol')}"
        knob = None
        # A declared completion knob on the consumer row changes drains, not the reuse route compared here.
        if ctx.imported(api) and not conv_traffic:
            knob = _mode_off(row, ctx.knobs)
        ranked.append({"kind": "mechanism", "row": row, "api": api, "knob": knob, **item})
    route = None
    script = composition_next.routes_script()
    for rid in script.BY_ID:
        if rid in signals and not ctx.imported(script.BY_ID[rid]["symbol"]):
            route = {"kind": "composition", "id": rid, "api": script.BY_ID[rid]["symbol"], "evidence": signals[rid][0]}
            break
    request = composition_next.request_advice(
        name, signals, inventory, declaration, row=ctx.rows.get(name) or {}, workspace=ctx.workspace
    )
    choices = list(ranked)
    if route is not None:
        first = name in ctx.flagged or (mode == "resource" and "fusion" in facets) or not ranked
        choices.insert(0 if first else min(1, len(choices)), route)
    if request is not None:
        choices.insert(0, {"kind": "source_composition", "request": request})
    if not choices:
        return None
    pick = choices[0]
    construction = None
    reading = bottleneck or "headroom above the roofline bound"
    if pick["kind"] == "mechanism":
        row = pick["row"]
        ident = str(row.get("id"))
        notes = []
        if pick["knob"]:
            knob, current, following = pick["knob"]
            notes.append(f"imported, but knob {knob}={json.dumps(current)} keeps it at its baseline")
            variant = _variant(f"{ident}-{knob}")
            apply = (
                f"set {knob} to {json.dumps(following)} in variants/{variant}/ (a copy of submission/), then "
                + _check(name, variant, suite)
            )
        else:
            notes.append("imported; runtime use unconfirmed" if ctx.imported(pick['api'])
                         else "not imported by submission/")
            notes.append(pick['opportunity']['opportunity_status'])
            apply = ab_command(name, _variant(ident), suite, mechanism=ident, workspace=ctx.workspace)
            if pick['opportunity']['source_call_sites']:
                apply = 'tune a distinct legal setting or inspect emitted-route coverage, then ' + apply
            elif pick['opportunity']['opportunity_status'] == 'applicability_unknown':
                apply = 'check source/effect prerequisites first, then ' + apply
            if not ctx.knobs and not (ctx.tuning_progress or {}).get("declared_candidates"):
                from .idle_tuning import construction_task

                construction = construction_task(ctx.workspace)
                notes.append("construct shared source/effect adapter + SAME emitter"
                             if construction["kind"] == "bind_source_effects" else
                             "declare finite legal alternatives in TUNING_KNOBS.json")
        if pick["unconfirmed"]:
            notes.append("capability unconfirmed: " + ", ".join(pick["unconfirmed"][:2]))
        if pick["applies_if"]:
            notes.append("applies if " + ", ".join(pick["applies_if"][:2]))
        read = f"{ctx.skills}/pass-library/reference/{row.get('family')}.md"
        route = next((r for r in operand_traffic.library_routes(selection, convolution=True) if r["id"] == ident),
                     None) if conv_traffic else None
        if route is not None:  # its opened capabilities stay in related_library_routes; the line is bounded
            read = f"{ctx.skills}/{route['recipe']}"
        choice = f"{ident} ({pick['api']}; {'; '.join(notes)})"
        kind, ident_out, api = "mechanism", ident, pick["api"]
    elif pick["kind"] == "composition":
        rid = pick["id"]
        choice = f"composition route {rid}: {pick['evidence']} ({pick['api']}; not imported by submission/)"
        apply = ab_command(name, _variant(rid), suite, mechanism=rid, workspace=ctx.workspace)
        read = script.next_command(script.BY_ID[rid], ctx.skills)
        kind, ident_out, api = "composition", rid, pick["api"]
    else:
        request = pick["request"]
        options = [m["option"] for m in request["options"] if m["option"] != "op_profile"] or [
            m["option"] for m in request["options"]
        ]
        variant = _variant(f"sc-{name}")
        choice = f"source_composition.json {', '.join(options)} ({request['options'][0]['why']})"
        apply = (
            f"write {json.dumps(request['example'], separators=(',', ':'))} to variants/{variant}/"
            f"source_composition.json in a copy of submission/, then " + _check(name, variant, suite)
        )
        read = request["reference"]
        kind, ident_out, api = "source_composition", ",".join(options), None
    return {
        "test_case": name,
        "bottleneck": reading,
        "kind": kind,
        "choice": ident_out,
        "api": api,
        "choice_text": choice,
        "apply": apply,
        "read": read,
        "alternatives": len(choices) - 1,
        "evaluation": DECIDE_ON_ESTIMATES,
        **(pick['opportunity'] if pick['kind'] == 'mechanism' else {}),
        **({"operand_traffic": traffic,
            "related_library_routes": operand_traffic.library_routes(selection, convolution=conv_traffic)}
           if traffic else {}),
        **({"fence_step": fence_step} if fence_step is not None else {}),
        **({"construction_task": construction} if construction is not None else {}),
    }


def _capability_steps(ctx, suite):
    """An actual public failure may name a selected route without claiming the missing law exists."""
    from ..workload_roles import PRIMARY
    from . import operand_traffic
    from .library_pass import _variant, ab_command

    for name, row in ctx.rows.items():
        refusal = operand_traffic.refusal(row.get("failure"))
        if not refusal or ctx.roles.get(name) not in PRIMARY:
            continue
        selection = mechanism_hints.for_case(ctx.selection, name, ctx.workspace)
        routes = [r for r in operand_traffic.library_routes(selection) if r["kind"] == refusal["kind"]]
        if not routes:
            continue
        route = routes[0]
        pending = [key for key, value in route["requirements"].items() if value != "supported"]
        yield dict(test_case=name, kind="mechanism", choice=route["id"], api=route["api"],
                   bottleneck="capability refusal; inspect an existing checked library route",
                   choice_text=f"{route['id']} ({route['api']}; capability/effect law still required)",
                   apply=ab_command(name, _variant(route["id"]), _suite(name, ctx, suite),
                                    mechanism=route["id"], workspace=ctx.workspace),
                   read=f"{ctx.skills}/{route['recipe']}", alternatives=len(routes) - 1,
                   capability_refusal=refusal, prerequisites=route["prerequisites"],
                   unresolved_requirements=pending,
                   evaluation="advisory route; ordinary source, numerical and timing gates unchanged")
        return  # One construction hint cannot consume the whole performance block.


def line(step: dict, index: int | None = None) -> str:
    """One line; over :data:`LINE_BYTES` the pointer to read goes first, then the bottleneck is shortened.
    The apply command is never cut."""
    head = f"{index}. " if index is not None else ""
    if step.get("kind") in OTHER_KINDS:
        text = f"{head}{step['text']} | apply: {step['apply']}"
        if len(text.encode()) > LINE_BYTES:
            text = f"{head}{step['text'][: LINE_BYTES // 2]}... | apply: {step['apply']}"
        return text
    held = f" | held-out: {step['held_out']}" if step.get("held_out") else ""
    tail = f" -> {step['choice_text']} | apply: {step['apply']}{held}"
    text = f"{head}{step['test_case']}: {step['bottleneck']}{tail} | read: {step['read']}"
    if len(text.encode()) <= LINE_BYTES:
        return text
    text = f"{head}{step['test_case']}: {step['bottleneck']}{tail}"
    spare = LINE_BYTES - len(f"{head}{step['test_case']}: ...{tail}".encode())
    if len(text.encode()) > LINE_BYTES and spare > 0:
        text = f"{head}{step['test_case']}: {step['bottleneck'][:spare]}...{tail}"
    return text


def _held_out(bundle: dict | None) -> str | None:
    """The bundle's siblings as the A/B's held-out check: the panel command and how many share the class."""
    if not bundle or bundle.get("passing", 0) < 2:
        return None
    where = bundle.get("panel") or "the siblings"
    return f"{where} ({bundle['passing'] - 1} passing {bundle['label']} siblings)"


def _measure(bundle: dict, ctx: Context) -> dict:
    """A passing representative without cycles: obtain them (the idle timing lanes also take it first)."""
    from .performance_brief import GSIM_COMMAND

    name = bundle["representative"]
    suite = _suite(name, ctx, "core")
    command = (f"python agent_selfcheck.py --suite {suite} --capsules {name}" if suite in ("scalability", "programs")
               else GSIM_COMMAND.format(name=name))
    text = (f"{name}: passes ({bundle['label']}, {bundle['passing']} passing), no cycles yet -> time it first; "
            "idle timing takes bundle representatives first")
    return {"kind": "measure", "test_case": name, "text": text, "apply": command, "bundle": bundle["label"]}


def _correctness(record: dict, ctx: Context) -> dict:
    """A family's cases left to finish, and the library route that builds them fast from the start."""
    from . import family_rollup

    left = record["left"][:family_rollup.NAMED]
    suites = {_suite(name, ctx, "core") for name in left}
    suite = suites.pop() if len(suites) == 1 else "core"
    route = f" --suite {suite}" if suite in ("scalability", "programs") else ""
    step = {"kind": "finish_correctness", "test_case": left[0] if left else record["family"],
            "family": record["family"], "text": family_rollup.correctness_text(record),
            "apply": f"python agent_selfcheck.py{route} --capsules {','.join(left)}"}
    if record.get("construction"):
        step["read"] = f"{ctx.skills}/" + record["construction"][0]["recipe"].replace("/", "/scripts/", 1)
    from . import reference_construction

    reference_construction.attach(step, ctx, suite=suite)
    return step


def _ranked(entries, families, ctx: Context, suite: str) -> list[dict]:
    """Every step by value: large and connected cases first; within each tier performance steps (by headroom,
    then representatives to time) alternate with families that have cases left (most cases left first).
    A passing case's step never waits for another case, and no family has to finish first."""
    reps = {b["representative"]: b for b in (families or {}).get("bundles") or ()}
    in_bundle = {}
    for bundle in (families or {}).get("bundles") or ():
        for name in (bundle["representative"], *(bundle.get("siblings") or ())):
            in_bundle.setdefault(name, bundle)
    items, seen = [], set()
    for entry in entries:  # this response's own primary cases with headroom
        bundle = in_bundle.get(entry["test_case"])
        items.append(((0, 0, -entry["headroom_cycles"]), entry, bundle))
        seen.add(entry["test_case"])
    for name, bundle in reps.items():
        if name in seen:
            continue
        rank = 0 if bundle.get("primary") else 1
        if bundle.get("top"):
            items.append(((rank, 0, -bundle["top"]["headroom_cycles"]), dict(bundle["top"]), bundle))
        elif bundle.get("untimed"):
            items.append(((rank, 1, -bundle["passing"]), None, bundle))
    for record in (families or {}).get("families") or ():
        if record["correct"] < record["cases"]:
            items.append(((0 if record["left_large"] else 1, 2, -(record["cases"] - record["correct"])), None, record))
    # Within each tier (large and connected first) performance and correctness steps alternate in their own
    # value order, so neither kind pushes the other out of the printed block.
    items.sort(key=lambda item: item[0])
    position: dict[tuple, int] = {}
    order = []
    for item in items:
        group = (item[0][0], item[0][1] == 2)
        position[group] = position.get(group, -1) + 1
        order.append(((item[0][0], position[group], item[0][1] == 2), item))
    items = [item for _, item in sorted(order, key=lambda pair: pair[0])]
    steps = []
    for _, entry, holder in items:
        if len(steps) >= LIMIT:
            break
        if entry is not None:
            found = choose(entry, ctx, suite=suite)
            if found is not None:
                held = _held_out(holder)
                if held:
                    found["held_out"] = held
                    found["bundle"] = holder["label"]
                fence_step = found.pop("fence_step", None)
                steps.append(found)
                if fence_step is not None:  # the case's drain evidence keeps its own line
                    steps.append(fence_step)
        elif "representative" in holder:
            steps.append(_measure(holder, ctx))
        else:
            steps.append(_correctness(holder, ctx))
    return steps


def block(document: dict, rows, brief, *, workspace=None, families=None, context=None) -> dict | None:
    """``{schema, lines, steps}``: the top :data:`LIMIT` steps by value (:func:`_ranked`) and the
    informational family line; None without any step."""
    from .library_pass import suite_of

    entries = [
        e
        for e in (brief.get("largest_headroom") or () if isinstance(brief, dict) else ())
        if isinstance(e, dict) and isinstance(e.get("headroom_cycles"), int | float) and e["headroom_cycles"] > 0
    ]
    from . import operand_traffic

    refusal = any(operand_traffic.refusal(row.get("failure")) for row in rows or () if isinstance(row, dict))
    if not entries and not (families or {}).get("bundles") and not (families or {}).get("families") and not refusal:
        return None
    ctx = context if context is not None else Context(workspace, document, rows)
    suite = suite_of(document) or "core"
    steps = (list(_capability_steps(ctx, suite)) + _ranked(entries, families, ctx, suite))[:LIMIT]
    if not steps:
        return None
    from . import reference_construction

    for step in steps:
        if "reference_construction" not in step:
            reference_construction.attach(step, ctx, suite=suite)
    progress = ctx.tuning_progress
    summary = (f" idle sweep: {progress['ready_baselines']} ready baselines, "
               f"{progress['declared_candidates']} candidates, "
               f"{progress['outstanding_candidate_cases']} candidate/case checks outstanding, "
               f"{progress['requests_in_flight']} requests in flight ({progress['status']})."
               if progress else "")
    lines = [HEADER + summary]
    for index, step in enumerate(steps, 1):
        text = line(step, index)
        if sum(len(item.encode()) + 1 for item in lines) + len(text.encode()) > BLOCK_BYTES:
            break
        lines.append(text)
    kept = steps[: len(lines) - 1]
    family = (families or {}).get("family_line")
    if family and sum(len(item.encode()) + 1 for item in lines) + len(family.encode()) <= BLOCK_BYTES:
        lines.append(family[:LINE_BYTES])
    lines = _reference_lines(lines, kept, (families or {}).get("reference_lines") or {})
    if isinstance(families, dict):
        _bundle_lines(families, kept)
    return {"schema": SCHEMA, "advisory_only": True, "lines": lines, "steps": kept,
            **({"idle_sweep_progress": progress} if progress else {})}


def _reference_lines(lines: list[str], steps: list[dict], references: dict) -> list[str]:
    """Each kept step's matching reference lines (:mod:`.reference_map`) under it, within the block budget;
    steps and the family line keep their place."""
    used = sum(len(item.encode()) + 1 for item in lines)
    out = lines[:1]
    for index, text in enumerate(lines[1:]):
        out.append(text)
        step = steps[index] if index < len(steps) else None
        extras = ((step or {}).get("reference_construction") or {}).get("lines", [])
        if extras and step.get("api"):
            references_used = step["reference_construction"]["references"]
            extras = [f"reference target: close expert mechanism/perf gap; test {step['choice']} via {step['api']} "
                      f"against {references_used[0]['id']}; derive a shape/resource rule", *extras[1:]]
        if not extras:
            extras = references.get((step or {}).get("test_case"), ()) if step is not None else ()
            if extras and step.get("api"):
                # Name the selected library experiment first. The reference map
                # is hypothesis evidence, not proof that this mechanism is missing.
                extras = [f"reference experiment: {step['choice']} via {step['api']} | {extras[0]}", *extras[1:]]
        for extra in extras:
            extra = extra.encode()[:420].decode("utf-8", errors="ignore")
            extra = "   " + extra
            if used + len(extra.encode()) + 1 <= BLOCK_BYTES:
                out.append(extra)
                used += len(extra.encode()) + 1
    return out


def _bundle_lines(families: dict, steps: list[dict]) -> None:
    """Name each bundle's chosen step on its brief line (the roll-up already sits in the brief)."""
    from . import family_rollup

    chosen = {step["test_case"]: step for step in steps if step.get("kind") not in OTHER_KINDS}
    bundles = families.get("bundles") or ()
    families["lines"] = [
        family_rollup.bundle_line(b, chosen.get(b["representative"])) for b in bundles[: family_rollup.BUNDLES]
    ]


def attach(document: dict, rows, brief, *, workspace=None, families=None, context=None) -> None:
    from ..author_profiles import modern_advice

    if not modern_advice():
        return
    """Put :data:`KEY` on ``document``; never raises (the response stays as it was)."""
    try:
        found = block(document, rows, brief, workspace=workspace, families=families, context=context)
        if found is not None:
            document[KEY] = found
    except Exception:  # noqa: BLE001 -- advisory only
        pass
    from . import reference_experiments
    reference_experiments.attach(document)


__all__ = ["KEY", "SCHEMA", "Context", "attach", "block", "choose", "line", "mechanism_candidates"]
