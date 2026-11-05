"""Name catalogue mechanisms that address a test case's measured bottleneck; advice, never a verdict.

The optimization catalogue offered every mechanism as one long menu, so the author had to map its own
measurements onto it by hand and mostly wrote specialised schedules instead of reusing the library's
general passes. This module joins two things the author already has: the bottleneck its own checks
measured for a test case (the busiest modeled resource, or which roofline bound limits it) and the
mechanisms the architecture selection in ``OPTIMIZATION_SUGGESTIONS.json`` found applicable to this
target. Nothing here is curated per model, test case or target, and nothing ranks mechanisms by what is
known to win: the mapping below goes from the timing model's generic resource vocabulary to the
catalogue's own facets, and mechanisms keep catalogue order within a facet.

Resource names are ``<domain>.<kind>[.<detail>]`` (``accelerator.storage.scratchpad.0``,
``accelerator.issue.load``, ``cpu.issue``, ``memory.l1d.probe`` ...). The mapping:

* local storage busy (``storage``) -> storage, movement, overlap mechanisms;
* transfers busy (``memory.*``, or a ``load``/``store`` controller or issue lane) -> movement, overlap, storage;
* instruction issue busy (``cpu.issue``, ``cpu.rocc_dispatch``, an accelerator ``issue`` lane) -> issue,
  synchronization, scheduling;
* accelerator execution busy (``execute``) -> scheduling, overlap, fusion;
* other host work (``cpu.*``) -> host_compute, issue, packing.

Without resource feedback, the limiting roofline bound decides: request service -> movement, storage,
overlap; compute floor -> scheduling, issue, fusion. Hints take one mechanism per facet in turn.

A busy storage resource also names its unit (``accumulator`` in ``accelerator.storage.accumulator.all``).
A facet alone pointed an accumulator-bound case at tile capacity, copy coalescing and prefetch, which
change memory traffic and leave accumulator work alone. Mechanisms whose own catalogue entry (id,
requirements, mechanism, legality, evidence) names that unit therefore come first, most mentions first,
then those in the resource's facets; the facet hints fill the remaining places.

A case whose command stream drains the whole device after nearly every output tile (limiter
``FENCE_BOUND``, read by :mod:`.fence_overlap`) is held back by that ordering rather than by one unit:
overlap, synchronization, movement, mechanisms whose entry names a fence, barrier or readout first
(:func:`fence_hints`). A fence mechanism scoped to one workload (``WORKLOAD_FENCE_MECHANISMS``) leads
only when the case's own declaration confirms that workload (:func:`for_case`), so a convolution case
is pointed at compute epochs first, a matrix product at dense panel epochs, and neither at the
other's. Mechanisms of those facets that name them but whose capability the selection could not
confirm are listed apart with the unconfirmed requirements, never as available.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

SUGGESTIONS = "OPTIMIZATION_SUGGESTIONS.json"
#: Mechanisms named per test case.
PER_CASE = 4
_MAX_BYTES = 8 << 20

_STORAGE = ("storage", "movement", "overlap")
_TRANSFER = ("movement", "overlap", "storage")
_ISSUE = ("issue", "synchronization", "scheduling")
_EXECUTE = ("scheduling", "overlap", "fusion")
_HOST = ("host_compute", "issue", "packing")
#: The limiter of a stream fenced after nearly every output tile, and the facets that address it.
FENCE_BOUND = "fences"
_SYNC = ("synchronization", "overlap", "issue", "movement")
#: Words a catalogue entry uses for what a fence-bound stream does too often (stems match; ``drain``
#: is left out because its stem also matches ``dram``).
FENCE_WORDS = ("fence", "barrier", "readout")
# Generic catalogue mechanisms, not target-specific winning configurations.
# Command loops address submission/code overhead; they do not remove dynamic fences.
#: Fence mechanisms that apply to one workload: first only when the case declares that workload.
WORKLOAD_FENCE_MECHANISMS = ("convolution_compute_epochs", "convolution_readout_epochs", "dense_panel_epochs")
FENCE_MECHANISMS = (*WORKLOAD_FENCE_MECHANISMS, "barrier_cleanup", "copy_pipeline", "command_loops")


def with_source_evidence(selection, workspace):
    """Observe current authored call sites; imports alone never establish use.

    This is bounded source discovery, not execution or pass qualification. Read
    the current tree again on each feedback request so edits refresh the advice.
    """
    if not selection:
        return selection
    from .library_pass import inventory_script

    root = Path(workspace) / "submission"
    calls, imports, keyword_calls = {}, {}, {}
    for path in inventory_script().python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError, UnicodeError):
            continue
        aliases = {}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                aliases.update((a.asname or a.name, f"{node.module}.{a.name}") for a in node.names)
                imported.update((node.module, *(f"{node.module}.{a.name}" for a in node.names)))
            elif isinstance(node, ast.Import):
                aliases.update((a.asname or a.name.split('.')[0], a.name if a.asname else a.name.split('.')[0])
                               for a in node.names)
                imported.update(a.name for a in node.names)
        for name in imported:
            if name == 'merlin' or name.startswith('merlin.'):
                imports.setdefault(name, []).append(path.relative_to(root).as_posix())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts, value = [], node.func
            while isinstance(value, ast.Attribute):
                parts.insert(0, value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id in aliases:
                name = '.'.join((aliases[value.id], *parts))
                if name.startswith('merlin.'):
                    site = f"{path.relative_to(root).as_posix()}:{node.lineno}"
                    calls.setdefault(name, []).append(site)
                    for kw in node.keywords:
                        if kw.arg and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                            keyword_calls.setdefault(name, {}).setdefault(kw.arg, []).append(site)
    adapter = root / 'tools/library_pass_adapter.py'
    return {**selection, 'source_calls': calls, 'source_imports': imports,
            'source_keyword_calls': keyword_calls,
            'library_adapter_present': adapter.is_file() and not adapter.is_symlink()}


def _prerequisites(row):
    """Use the selected mechanism descriptor and existing public API contract."""
    checks = [row.get('legality'), row.get('adapter_requirements')]
    if str(row.get('api', '')).endswith(('.dense_array_retention', '.tile_load_coalescing')):
        from merlin.compile.scheduling.api_catalog import scheduling_capabilities

        name = ('array_weight_retention' if str(row['api']).endswith('.dense_array_retention')
                else 'exact_final_wide_loads')
        capability = next(c for c in scheduling_capabilities(23)['capabilities'] if c['name'] == name)
        checks.extend(capability['requires'])
        binding = adapter_contract(row, {})
        if binding:
            checks.extend(binding['required_bindings'])
    if str(row.get('api', '')).endswith('.spatial_readout_cache'):
        from merlin.compile.scheduling.api_catalog import scheduling_capabilities

        capability = next(c for c in scheduling_capabilities(17)['capabilities']
                          if c['name'] == 'source_exact_quantized_spatial_readout_cache')
        checks.extend(capability['requires'])
        checks.append('connected staged producer/readouts in the original RollingPackedSpatialPlan')
    elif str(row.get('api', '')).endswith('.temporary_storage'):
        checks.append('StorageAdapter.project/rewrite must decode actual effects, completion and relocation')
    elif row.get('id') == 'convolution_readout_epochs':
        from merlin.compile.scheduling.api_catalog import scheduling_capabilities

        capability = next(c for c in scheduling_capabilities(23)['capabilities']
                          if c['name'] == 'source_qualified_convolution_readout_epochs')
        checks.extend(capability['requires'])
        checks.append('conv_epoch_recipe.emit_checked_readout_epoch verifies the actual consumer receipt; '
                      'missing/default/mismatched readout_epoch is refused before treating the route as applied')
    return list(dict.fromkeys(str(c) for c in checks if c))


def adapter_contract(row, selection):
    """Name existing bindings a selected route needs; source discovery proves none.

    A generic pass-adapter file, import or proposal call cannot establish that
    retained modes or wide operands reach the checked emitted program. The
    actual address/array law remains an independently selected target input.
    """
    prefix = 'merlin.compile.scheduling.'
    module = str(row.get('api', ''))
    if module == prefix + 'dense_array_retention':
        capability = 'array_weight_retention'
        consumers = ('dense_array_retention.verify_dense_array_retention',
                     'dense_array_retention.verify_dense_array_tile_loads')
        bindings = [
            'DenseLoopAdapter.encode/project must round-trip actual fresh and retained Preload/Compute modes: '
            'weight_row/weight_block=None and fresh_weights=False; an always-fresh projector cannot bind this route',
            'Bind DenseArrayRetentionAdmission from selected fresh-capture/STAY/output-descriptor semantics; '
            'opcode availability and capacity do not establish that law',
            'Use dense_retained_compute_order(order=("k", "n", "m")) to propose source-axis LoopTransforms; '
            'consume schedule.loop_transform in schedule.plan and recheck the actual pure compute run',
            'Verify against the original retained/segmented source and epoch admission, then emit proposal.kernel; '
            'checking_kernel is proof-only and must never replace the retained emitted program',
        ]
        action = ('Bind actual fresh/retained encode/project modes and the selected array law; '
                  'consume schedule.plan, verify the retained kernel, then reanalyze emitted bytes.')
    elif module == prefix + 'tile_load_coalescing':
        capability = 'exact_final_wide_loads'
        consumers = ('panel_tile_loads.admit_panel_tile_loads',
                     'dense_segmented_pipeline.admit_segmented_tile_loads',
                     'dense_array_retention.verify_dense_array_tile_loads')
        bindings = [
            'Bind TileLoadAddressLaw from selected decoded limits and addressing: tile_columns, max_rows, '
            'max_columns, element_bytes, source_pitch_bytes and local_block_stride; packet field width is not a limit',
            'Project the final actual wide call into TileLoadMapping and supply encode/encode_band round trips; '
            'retain each band source pointer, byte pitch, local row stride, tail and completion epoch',
            'Use the existing panel, segmented or array wide-load consumer for full source/lifetime/epoch checks; '
            'address mapping alone grants neither execution ordering nor completion removal',
            'Consume the original schedule.plan and emit the checked coalesced kernel, then decode actual final '
            'transfer rectangles instead of counting the primitive proof expansion as emitted commands',
        ]
        action = ('Bind final TileLoadMapping/encode_band and the selected address law; '
                  'consume schedule.plan, run the existing wide-load consumer, then reanalyze emitted bytes.')
    else:
        return None
    observed = [dict(api=prefix+name, sites=list((selection.get('source_calls') or {}).get(prefix+name) or ())[:4])
                for name in consumers if (selection.get('source_calls') or {}).get(prefix+name)]
    return dict(capability=capability, required_bindings=bindings,
        verification_consumers=[prefix+name for name in consumers], observed_consumer_call_sites=observed,
        consumer_binding='source_call_only' if observed else 'not_observed',
        target_law='unverified_by_source_discovery',
        next_action=action, gap=dict(kind='evidence_gap' if observed else 'adapter_gap',
            goals=['adapter.'+capability, 'schedule.consumed', 'evidence.actual_emission'],
            reason=('A checked consumer source call is observed; actual mode/address binding and emitted use remain '
                    'unverified' if observed else
                    'No checked consumer call observed; library/proposal availability does not bind target effects'),
            next_action=action))


def opportunity(row, selection):
    """Small advisory states; no state grants legality, application or ranking."""
    api = f"{row.get('api')}.{row.get('symbol')}"
    sites = list((selection.get('source_calls') or {}).get(api) or ())[:4]
    consumer_sites = sites
    if row.get('id') == 'convolution_readout_epochs':
        # The default call deliberately retains every store drain. Only an
        # explicit opt-in keyword is a source observation, still not execution.
        sites = list(((selection.get('source_keyword_calls') or {}).get(api) or {})
                     .get('readout_law') or ())[:4]
    missing = sorted(k for k, v in (row.get('requirements') or {}).items() if v != 'supported')
    special = (str(row.get('api', '')).endswith(('.spatial_readout_cache', '.temporary_storage'))
               or row.get('id') == 'convolution_readout_epochs')
    if (str(row.get('api', '')).endswith('.spatial_readout_cache')
            and row.get('consumer_geometry') != 'supported'):
        missing.append('original connected consumer window geometry')
    checks = _prerequisites(row)
    if missing or special and not sites:
        status = 'applicability_unknown'
        action = ('Check the original source and lowering prerequisites before proposing this pass: '
                  + '; '.join(checks or missing))
    elif sites:
        status = 'already_observed'
        action = ('Inspect emitted-route coverage and tune a distinct legal configuration; '
                  'do not repeat import/setup. Source calls do not prove execution or a performance gain.')
    elif selection.get('library_adapter_present') is False:
        status = 'missing_adapter'
        action = ('Implement tools/library_pass_adapter.py prepare/source hooks '
                  'using the existing ab-library-pass recipe.')
    else:
        status = 'candidate'
        action = 'Try a changed legal pipeline through ab-library-pass/scripts/apply_library_pass.py.'
    if special and not sites and selection.get('library_adapter_present') is False:
        action += ' Build the source/effect adapter first; library availability alone is insufficient.'
    if sites and status != 'already_observed':
        action += ' A source call is already present: inspect its coverage/refusal before adding another.'
    binding = adapter_contract(row, selection)
    if binding:
        action += ' ' + binding['next_action']
    return dict(opportunity_status=status, source_call_sites=sites,
                consumer_call_sites=consumer_sites,
                observation_basis=('source call only; runtime application unconfirmed'
                                   if sites else 'not observed in source'),
                prerequisites=checks, next_action=action,
                recipe='ab-library-pass/reference/adapter.md',
                **(dict(adapter_contract=binding) if binding else {}))


def _priority(row, selection, deprioritize):
    observed = opportunity(row, selection)
    status = observed['opportunity_status']
    return (bool(observed['source_call_sites']),
            {'candidate': 0, 'missing_adapter': 1, 'applicability_unknown': 2,
             'already_observed': 3}[status], row.get('id') in deprioritize)


def for_case(selection, name, workspace):
    """Refine workload applicability from this case's declaration, never its name.

    The initial architecture menu has no workload IR. A known dense contraction
    excludes convolution-only actions and confirms dense-only ones (a known
    convolution the reverse); unknown or mixed declarations stay unknown.
    This does not infer buffer independence, quantization legality or target effects.
    """
    if not selection:
        return selection
    from . import case_declarations, shape_classes

    declaration = case_declarations.load(name, workspace)
    description = shape_classes.describe(declaration)
    if not description:
        return selection
    convolution = "supported" if description["family"] == "conv" else "unsupported"
    scoped = {"workload.convolution": convolution,
              "workload.dense_contraction": "unsupported" if convolution == "supported" else "supported"}
    rows = []
    for row in selection.get("all_mechanisms") or ():
        if not isinstance(row, dict):
            continue
        requirements = dict(row.get("requirements") or {})
        if scoped.keys() & requirements.keys():
            requirements.update({key: state for key, state in scoped.items() if key in requirements})
            status = ("unsupported" if "unsupported" in requirements.values() else
                      "unknown" if "unknown" in requirements.values() else "supported")
            row = {**row, "requirements": requirements, "status": status}
        if str(row.get('api', '')).endswith('.spatial_readout_cache'):
            attrs = (declaration.get('operation') or {}).get('attributes') or {}
            expected = {'pool_size': (3, 3), 'pool_stride': (2, 2), 'pool_padding': (1, 1, 1, 1)}
            observed = {}
            for key, target in expected.items():
                value = attrs.get(key)
                if type(value) is int:
                    value = (value,) * len(target)
                if isinstance(value, (tuple, list)) and all(type(v) is int for v in value):
                    observed[key] = tuple(value)
            geometry = ('unsupported' if any(observed[k] != expected[k] for k in observed) else
                        'supported' if len(observed) == len(expected) else 'unknown')
            row = {**row, 'consumer_geometry': geometry}
            if geometry == 'unsupported':
                row = {**row, 'status': 'unsupported'}
        rows.append(row)
    return {**selection, "all_mechanisms": rows}


def resource_facets(name) -> tuple[str, ...] | None:
    """Catalogue facets that address a busy resource, from its generic name segments."""
    if not isinstance(name, str) or not name:
        return None
    segments = name.split(".")
    domain, rest = segments[0], segments[1:]
    if "storage" in rest:
        return _STORAGE
    if domain == "memory" or "load" in rest or "store" in rest:
        return _TRANSFER
    if (domain == "cpu" and rest[:1] in (["issue"], ["rocc_dispatch"])) or (domain != "cpu" and "issue" in rest):
        return _ISSUE
    if "execute" in rest:
        return _EXECUTE
    if domain == "cpu":
        return _HOST
    return None


def storage_unit(name) -> str | None:
    """The storage unit a busy storage resource names (the segment after ``storage``), else None."""
    segments = name.split(".") if isinstance(name, str) else []
    if "storage" not in segments[1:]:
        return None
    after = segments[segments.index("storage", 1) + 1 :]
    return after[0].lower() if after and after[0].isalpha() else None


def _words(text) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in str(text or "").lower()).split()


def _same_stem(word: str, unit: str) -> bool:
    """``accumulate``/``accumulation`` name an ``accumulator``: they share all but its last two letters."""
    stem = max(len(unit) - 2, 1)
    return len(word) >= stem and len(os.path.commonprefix([word, unit])) >= stem


def _mentions(row: dict, unit) -> int:
    """How many fields of a catalogue entry name ``unit`` (one word, or any of a tuple of words)."""
    units = (unit,) if isinstance(unit, str) else tuple(unit)
    fields = [
        row.get("id"),
        " ".join((row.get("requirements") or {}).keys()),
        row.get("mechanism"),
        row.get("legality"),
        row.get("evidence"),
    ]
    return sum(any(_same_stem(word, name) for word in _words(text) for name in units) for text in fields)


def bound_facets(bound_by) -> tuple[str, ...] | None:
    if not isinstance(bound_by, str):
        return None
    if "requests" in bound_by:
        return _TRANSFER
    if bound_by == "compute":
        return ("scheduling", "issue", "fusion")
    if bound_by == FENCE_BOUND:
        return _SYNC
    return None


def load_selection(workspace) -> dict | None:
    """The architecture selection staged for the author, or ``None`` when it was not staged."""
    path = Path(workspace) / SUGGESTIONS
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) and isinstance(document.get("all_mechanisms"), list) else None


def _available(selection: dict) -> list[dict]:
    """Mechanisms whose capability and library requirements are all supported (workload may be pending)."""
    rows = []
    for row in selection.get("all_mechanisms") or ():
        requirements = row.get("requirements") if isinstance(row, dict) else None
        if not isinstance(requirements, dict) or row.get("status") == "unsupported":
            continue
        if all(state == "supported" for key, state in requirements.items() if not key.startswith("workload.")):
            rows.append(row)
    return rows


def _workload_confirmed(row: dict) -> bool:
    """Every workload prerequisite of ``row`` is confirmed for this case (none pending or refused)."""
    states = [v for k, v in (row.get("requirements") or {}).items() if k.startswith("workload.")]
    return bool(states) and all(state == "supported" for state in states)


def fence_hints(selection: dict | None, *, why: str, limit: int = PER_CASE,
                deprioritize=()) -> tuple[list[dict], list[dict]]:
    """``(available, unconfirmed)`` mechanisms for a fence-bound stream (see the module docstring)."""
    if not selection:
        return [], []
    found = hints(selection, bound_by=FENCE_BOUND, limit=limit, words=FENCE_WORDS,
                  why=why, deprioritize=deprioritize)
    chosen = {hint["id"] for hint in found}
    unconfirmed = []
    for row in selection.get("all_mechanisms") or ():
        requirements = row.get("requirements") if isinstance(row, dict) else None
        if (
            not isinstance(requirements, dict)
            or row.get("id") in chosen
            or row.get("status") == "unsupported"
            or row.get("family") not in _SYNC
            or not (_mentions(row, FENCE_WORDS) or row.get("id") in FENCE_MECHANISMS)
        ):
            continue
        unknown = sorted(
            key for key, state in requirements.items() if not key.startswith("workload.") and state != "supported"
        )
        if unknown:
            unconfirmed.append(
                {
                    "id": row.get("id"),
                    "api": f"{row.get('api')}:{row.get('symbol')}",
                    "facet": row.get("family"),
                    "capability_unconfirmed": unknown,
                }
            )
    unconfirmed.sort(key=lambda row: (
        row.get("id") not in FENCE_MECHANISMS,
        FENCE_MECHANISMS.index(row["id"]) if row.get("id") in FENCE_MECHANISMS else 0))
    return found, unconfirmed[:limit]


def hints(
    selection: dict | None, *, dominant=None, bound_by=None, limit: int = PER_CASE,
    words=None, why=None, deprioritize=()
) -> list[dict]:
    """Up to ``limit`` mechanisms for one test case's measured bottleneck (see the module docstring).

    ``words`` (with ``why``) replaces the busy storage unit as what an entry must name to come first.
    """
    if not selection:
        return []
    name = (dominant or {}).get("name") if isinstance(dominant, dict) else None
    facets = resource_facets(name)
    if facets is not None:
        why = f"busiest modeled resource {name} (activity {dominant.get('modeled_activity')})"
    else:
        facets = bound_facets(bound_by)
        if facets is not None and why is None:
            why = f"limited by its {bound_by} roofline bound"
    if facets is None:
        return []
    available = _available(selection)
    available.sort(key=lambda row: _priority(row, selection, deprioritize))
    unit = (tuple(words) if words else storage_unit(name)) if why and (name or words) else None
    named = []
    if unit is not None:
        scored = [(_mentions(row, unit), index, row) for index, row in enumerate(available)]
        named = [
            row
            for count, _index, row in sorted(
                (item for item in scored if item[0]),
                key=lambda item: (_priority(item[2], selection, deprioritize), -item[0],
                                  item[2].get("family") not in facets, item[1]),
            )
        ]
    if bound_by == FENCE_BOUND:
        direct = [
            row
            for key in FENCE_MECHANISMS
            for row in available
            if row.get("id") == key and (key not in WORKLOAD_FENCE_MECHANISMS or _workload_confirmed(row))
        ]
        named = direct + [row for row in named if row not in direct]
    # Then one mechanism per facet in turn (catalogue order within a facet), facets the named mechanisms
    # did not cover first, so the hints span the ways to address the bottleneck.
    covered = {row.get("family") for row in named}
    order = sorted(facets, key=lambda facet: facet in covered)
    queues = [[row for row in available if row.get("family") == facet and row not in named] for facet in order]
    named_as = f"names the busy {unit} unit" if isinstance(unit, str) else f"names a {'/'.join(unit or ())}"
    direct_reasons = {
        "convolution_compute_epochs": "group computes into epochs drained only at epoch boundaries and stores "
        "where RTL ordering facts order the local rows (heavy-pass-recipes conv_epoch_recipe.py)",
        "dense_panel_epochs": "drain once per output block instead of after every K step where RTL ordering "
        "facts order loads, computes and accumulator updates (heavy-pass-recipes dense_epoch_recipe.py)",
        "convolution_readout_epochs": "bind the separate producer/readout source law, explicitly consume it "
        "and verify its applied receipt; retain the final private-block drain (conv_epoch_recipe.py)",
        "barrier_cleanup": "inspect redundant barriers under complete effect/completion laws",
        "copy_pipeline": "inspect independent copy placement and completion to enable overlap",
        "command_loops": "inspect host issue/code overhead; command compaction preserves dynamic fences",
    }
    chosen = [(row, f"{direct_reasons.get(row.get('id'), named_as) if bound_by == FENCE_BOUND else named_as}; {why}")
              for row in named]
    while any(queues):
        for queue in queues:
            if queue:
                chosen.append((queue.pop(0), why))
    def hint_priority(pair):
        row = pair[0]
        observed, *rest = _priority(row, selection, deprioritize)
        # Keep a confirmed workload's direct fence mechanism visible even when
        # its source-adapter premises need construction. This is still labeled
        # applicability_unknown; generic menus must not bury the missing seam.
        direct = (bound_by == FENCE_BOUND and row.get('id') in WORKLOAD_FENCE_MECHANISMS
                  and _workload_confirmed(row))
        return observed, not direct, *rest
    chosen.sort(key=hint_priority)
    result = []
    for row, row_why in chosen[:limit]:
        workload = {key: state for key, state in (row.get("requirements") or {}).items() if key.startswith("workload.")}
        result.append(
            {
                "id": row.get("id"),
                "api": f"{row.get('api')}:{row.get('symbol')}",
                "facet": row.get("family"),
                "why": row_why,
                "evidence_to_request": row.get("evidence"),
                **opportunity(row, selection),
                "workload_check": (
                    "satisfied"
                    if row.get("status") == "supported"
                    else "unchecked: run `python optimization_suggest.py --ir <source> --output <file>`"
                    f" for {sorted(k for k, v in workload.items() if v != 'supported')}"
                ),
            }
        )
    return result
