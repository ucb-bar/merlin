"""Bottom-up CCA extraction with occurrence-local provenance, without optimization selection."""

from __future__ import annotations

import ast
import bisect
import hashlib
import json
import struct
import time
from collections import Counter
from pathlib import Path

from .reference_database_io import store_body, write_json

ADVICE_KEYS = frozenset(
    (
        "next_action",
        "library_routes",
        "transfer_candidates",
        "transfer_questions",
        "experiment_tasks",
        "optimization_tasks",
        "knob_proposals",
    )
)


def observations_only(value):
    if isinstance(value, dict):
        return {key: observations_only(item) for key, item in value.items() if key not in ADVICE_KEYS}
    if isinstance(value, (tuple, list)):
        return [observations_only(item) for item in value]
    return value


def elf_function_symbols(elf):
    """Read original ELF function metadata; absence never creates a guessed source name."""
    functions = []
    for header in elf.headers:
        _, kind, _, _, offset, size, link, _, _, stride = header
        if kind not in (2, 11):
            continue
        if stride != 24 or size % stride or link >= len(elf.headers):
            raise ValueError("invalid ELF symbol table")
        strings = elf.headers[link]
        if strings[1] != 3:
            raise ValueError("invalid ELF symbol strings")
        names, symbols = elf.read_at(strings[4], strings[5]), elf.read_at(offset, size)
        for at in range(0, size, stride):
            name, info, _, owner, address, length = struct.unpack_from("<IBBHQQ", symbols, at)
            if info & 15 != 2 or not length or not 0 < owner < len(elf.headers):
                continue
            stop = names.find(b"\0", name, name + 1025)
            if not 0 <= name < len(names) or stop < 0:
                raise ValueError("invalid ELF symbol name")
            functions.append(dict(name=names[name:stop].decode(), address=address, size_bytes=length))
    return functions


def symbol_overlap(symbols, start, stop):
    """Name a span from overlapping ELF function symbols; partial extents stay explicit."""
    overlapping = [
        dict(symbol, overlap_bytes=min(stop, symbol["address"] + symbol["size_bytes"]) - max(start, symbol["address"]))
        for symbol in symbols
        if symbol["address"] < stop and start < symbol["address"] + symbol["size_bytes"]
    ]
    names = {symbol["name"] for symbol in overlapping}
    if not overlapping:
        return dict(symbol=None, symbol_status="none", overlapping_symbols=[])
    if len(names) != 1:
        return dict(symbol=None, symbol_status="ambiguous", overlapping_symbols=overlapping)
    exact = any(row["address"] == start and row["size_bytes"] == stop - start for row in overlapping)
    return dict(symbol=names.pop(), symbol_status="exact" if exact else "partial", overlapping_symbols=overlapping)


def independent_site_census(rows, opcode):
    """Count selected extension words over the whole listing, without consulting span inference."""
    by_funct = Counter()
    for row in rows:
        word = row.hexcode
        if word and len(word) == 8 and int(word, 16) & 127 == opcode:
            by_funct[int(word, 16) >> 25] += 1
    return dict(
        sites=sum(by_funct.values()),
        by_funct7={str(key): value for key, value in sorted(by_funct.items())},
        instructions=len(rows),
        basis="every listed 32-bit word whose major opcode equals the selected ISA's extension opcode",
    )


STREAM_MAX_STEPS = 8_000_000
STREAM_MAX_RUNS = 200_000
STREAM_MAX_ENTRIES = 1_000_000


def unproved_call_returns(counted, summaries):
    """Calls a proved execution stepped over without a known return value, with the callee's reason.

    A summary applies its ``a0`` only when the callee proves it from its entry state; a
    call assumed to return, or a summary whose return is not a known value, leaves the
    caller's return register unknown. Keyed by hex call target, from the execution's own counts.
    """
    out = {}
    for key, stepping in (("summarized_calls", "proved_summary"), ("assumed_returning_calls", "assumed_to_return")):
        for target, calls in sorted((counted.get(key) or {}).items()):
            summary = (summaries or {}).get(target) or {}
            if (summary.get("returns") or {}).get("a0") is not None:
                continue
            reason = summary.get("reason") or (
                "callee return is not a known value under its entry state (it reads state the summary does not "
                "prove, e.g. memory written before the call)"
                if summary.get("status") == "proved"
                else "no callee summary"
            )
            out[hex(target)] = dict(calls=calls, stepping=stepping, reason=reason)
    return out


def _note_unproved_returns(effects, record, context):
    """Name the unproved call returns behind loads left unresolved; observations are unchanged."""
    unproved = record.get("unproved_call_returns")
    if not unproved:
        return
    callers = ((context or {}).get("entry_context") or {}).get("callers")
    note = (
        "the proved execution steps over calls to " + ", ".join(unproved) + " whose return values are not "
        "proved; a local address computed from them stays unresolved (static callers of this region: "
        + ("unknown" if callers is None else str(callers))
        + ")"
    )
    for name in ("operand_buffering", "prefetch_distance"):
        fact = (effects.get("facts") or {}).get(name)
        if not isinstance(fact, dict):
            continue
        if fact.get("unresolved") or any("unresolved" in str(limit) for limit in fact.get("limits") or ()):
            fact["limits"] = [*(fact.get("limits") or ()), note]


def proved_stream(
    body, decoder, context=None, *, max_steps=STREAM_MAX_STEPS, max_runs=STREAM_MAX_RUNS, max_entries=STREAM_MAX_ENTRIES
):
    """The decoded issue-order stream of one region whose scalar control proves.

    The region's agreed entry state is tried first, then each per-caller entry
    context in order; the first proved execution supplies the stream. Returns
    ``(stream, arguments, truncated, record)``; ``stream`` is ``None`` when none proves.
    ``max_steps`` bounds scalar steps per attempt (about 0.3M steps/s on one core);
    ``max_runs`` bounds the affine-run record and ``max_entries`` the decoded stream.
    """
    from merlin.compile.analysis.riscv_machine import finite_control_counts
    from merlin.kernels.decoded_effects import decode_stream

    context = context or {}
    semantics = (decoder or {}).get("semantics")
    if semantics is None:
        return None, None, False, dict(status="unknown", reason="no selected operand observer")
    opcode = semantics.isa_constants((decoder or {}).get("target")).get("CUSTOM_OPCODE")
    if opcode is None:
        return None, None, False, dict(status="unknown", reason="selected ISA declares no extension opcode")
    tries = [("region", context.get("entry_state"))]
    tries += [
        (row.get("context_id"), dict(context.get("entry_state") or {}, **(row.get("registers") or {})))
        for row in context.get("contexts") or ()
    ]
    reasons = []
    for context_id, state in tries:
        try:
            summaries = {**(context.get("call_summaries") or {}), **(context.get("tail_summaries") or {})}
            counted = finite_control_counts(
                body,
                custom_opcode=opcode,
                entry_state=state or None,
                call_summaries=summaries or None,
                record_commands=True,
                max_steps=max_steps,
                max_commands=max_steps,
                max_recorded=max_runs,
                assume_returning_calls=True,
                symbolic_returns=True,
            )
        except (ValueError, TypeError) as error:
            reasons.append(dict(context_id=context_id, reason=f"entry state not applicable: {error}"))
            continue
        if counted["status"] == "proved":
            stream = decode_stream(counted["command_stream_runs"], decoder=decoder, limit=max_entries)
            arguments = {name: value for name, value in (state or {}).items() if type(value) is int}
            truncated = len(stream) < counted["command_stream_entries"]
            unproved = unproved_call_returns(counted, summaries)
            return (
                stream,
                arguments,
                truncated,
                dict(
                    **({"unproved_call_returns": unproved} if unproved else {}),
                    **(
                        {
                            "symbolic_returns": len(counted["symbolic_returns"]),
                            "symbolic_return_basis": "one fresh symbol per dynamic call instance; never assumed equal",
                        }
                        if counted.get("symbolic_returns")
                        else {}
                    ),
                    status="proved",
                    context_id=context_id,
                    entries=counted["command_stream_entries"],
                    decoded_entries=len(stream),
                    runs=len(counted["command_stream_runs"]),
                    truncated=truncated,
                    commands=counted["commands"],
                    fences=counted["fences"],
                    steps=counted["steps"],
                    max_steps=max_steps,
                    attempts=len(reasons) + 1,
                    basis="proved_scalar_control"
                    if not counted.get("assumed_returning_calls")
                    else "proved_scalar_control_assuming_listed_calls_return",
                    **(
                        {"assumed_returning_calls": {hex(k): v for k, v in counted["assumed_returning_calls"].items()}}
                        if counted.get("assumed_returning_calls")
                        else {}
                    ),
                ),
            )
        reasons.append(
            dict(context_id=context_id, **(counted.get("command_stream_status") or dict(reason=counted.get("reason"))))
        )
    return (
        None,
        None,
        False,
        dict(status="unknown", reason="no_proved_command_stream", max_steps=max_steps, attempts=reasons[:8]),
    )


def region_effects(report, body, decoder, structure, occurrence_id, context=None, *, stream_max_steps=STREAM_MAX_STEPS):
    """Pure addition: decoded-command effects for one accelerator span; CCA records are unchanged."""
    from merlin.kernels.decoded_effects import decoded_commands, derive_effects

    context = context or {}
    commands = decoded_commands(
        body, decoder=decoder, entry_state=context.get("entry_state"), call_summaries=context.get("call_summaries")
    )
    evidence_id = (report.get("evidence") or {}).get("evidence_id") or occurrence_id
    stream, arguments, truncated, record = proved_stream(body, decoder, context, max_steps=stream_max_steps)
    effects = derive_effects(
        report,
        structure,
        commands=commands,
        evidence_id=evidence_id,
        stream=stream,
        arguments=arguments,
        stream_truncated=truncated,
    )
    effects["command_stream"] = record
    _note_unproved_returns(effects, record, context)
    return effects


MAX_ENTRY_CONTEXTS = 8
ARGUMENT_REGISTERS = tuple("a" + str(index) for index in range(8))


def transfer_edges(body, start, stop, occurrence_id):
    """Static call and candidate tail transfers leaving one span, decoded from its own bytes."""
    from merlin.compile.analysis.riscv_machine import MASK, decode_cpu

    edges = []
    for index, row in enumerate(body):
        decoded = decode_cpu(row)
        if decoded.kind not in ("call", "call_indirect", "jump", "indirect"):
            continue
        transfer = "call" if decoded.kind in ("call", "call_indirect") else "tail"
        target, basis = decoded.target, "decoded_direct_" + transfer
        if decoded.kind in ("call_indirect", "indirect"):
            target, basis = None, "unresolved_indirect_" + transfer
            if index:
                previous, high = body[index - 1], decode_cpu(body[index - 1])
                if (
                    high.kind == "auipc"
                    and high.rd
                    and high.rd == decoded.rs1
                    and previous.addr + high.size == row.addr
                ):
                    target = (previous.addr + high.imm + decoded.imm) & MASK & ~1
                    basis = "adjacent_decoded_auipc_" + transfer
        if transfer == "tail" and (target is None or start <= target < stop):
            # Only control leaving this span is a candidate tail transfer; local jumps stay CFG.
            continue
        edges.append(
            dict(
                caller_id=occurrence_id,
                call_pc=row.addr,
                target=target,
                basis=basis,
                kind=transfer,
                callee_id=None,
                dynamic_multiplicity="unknown",
            )
        )
    return edges


def resolve_callees(edges, outputs):
    """Bind decoded targets to materialized spans; mid-span jumps are cross-span transfers."""
    # Maps identify code spans, not logical layers, invocation order, shapes or semantic equality.
    starts = sorted((row["address"], row["address"] + row["size_bytes"], row["id"]) for row in outputs)
    keys = [row[0] for row in starts]
    for edge in edges:
        if edge["target"] is not None:
            index = bisect.bisect_right(keys, edge["target"]) - 1
            if index >= 0 and starts[index][0] <= edge["target"] < starts[index][1]:
                edge["callee_id"] = starts[index][2]
                edge["target_offset"] = edge["target"] - starts[index][0]
    # A jump into the middle of another span is a cross-span transfer, not a tail call.
    edges = [
        dict(edge, kind="cross_span_jump") if edge["kind"] == "tail" and edge.get("target_offset") else edge
        for edge in edges
        if edge["kind"] != "tail" or edge["callee_id"] is not None
    ]
    return edges


def _known(value):
    if value is None:
        return False
    if isinstance(value, dict) and value.get("kind") == "unknown":
        return bool(value.get("known_bits") or value.get("pointer_base"))
    return True


def _canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def interprocedural_contexts(regions, edges, *, machine=None):
    """Caller-derived entry contexts for accelerator regions, from the lifted call graph.

    ``regions`` maps occurrence ids to ``(scope, body)``. Argument states come from
    each caller's lifter ``call_sites``; callee return summaries are memoized per
    callee span. Each caller is lifted under its own entry contexts (its per-caller
    contexts when it has them, else the registers all its callers agree on), so a
    constant set two or more calls up still reaches the callee's arguments; inside
    a recursive call chain the caller is lifted without an entry state.
    Contexts are evidence within one region, never new occurrences.
    Lifters without these outputs leave every context explicitly unknown.
    """
    from merlin.kernels.machine_evidence import lifter_call

    if machine is None:
        from merlin.compile.analysis import riscv_machine as machine
    summarize = getattr(machine, "summarize_function", None)
    summaries, sites, resolved, active = {}, {}, {}, set()
    entries = [edge for edge in edges if edge.get("callee_id") is not None and not edge.get("target_offset")]

    def call_summaries(caller_id):
        if summarize is None:
            return None
        bound = {}
        for edge in entries:
            if edge["caller_id"] != caller_id or edge["kind"] != "call" or edge["callee_id"] not in regions:
                continue
            if edge["callee_id"] not in summaries:
                summaries[edge["callee_id"]] = lifter_call(summarize, regions[edge["callee_id"]][1])
            bound[edge["target"]] = summaries[edge["callee_id"]]
        return bound or None

    def caller_sites(caller_id, state=None):
        key = (caller_id, _canonical(state))
        if key not in sites:
            try:
                _, analysis = lifter_call(
                    machine.register_facts,
                    regions[caller_id][1],
                    entry_state=state or None,
                    call_summaries=call_summaries(caller_id),
                )
            except (ValueError, TypeError):
                if not state:
                    raise
                sites[key] = caller_sites(caller_id)
                return sites[key]
            rows = (analysis or {}).get("call_sites")
            sites[key] = None if not isinstance(rows, list) else {row.get("pc"): row for row in rows}
        return sites[key]

    def caller_states(caller_id):
        """``(context_id, entry_state)`` pairs that together cover every static caller of ``caller_id``."""
        if caller_id in active:
            return [(None, None)]
        context = resolve(caller_id)
        agreed = context.get("entry_state")
        rows = context.get("contexts") or ()
        if rows:
            return [(row["context_id"], dict(agreed or {}, **row["registers"])) for row in rows]
        return [(None, agreed)]

    def resolve(occurrence_id):
        if occurrence_id not in resolved:
            active.add(occurrence_id)
            try:
                resolved[occurrence_id] = entry_context(occurrence_id)
            finally:
                active.discard(occurrence_id)
        return resolved[occurrence_id]

    def entry_context(occurrence_id):
        scope = regions[occurrence_id][0]
        incoming = [edge for edge in entries if edge["callee_id"] == occurrence_id and edge["caller_id"] in regions]
        callers = sorted({edge["caller_id"] for edge in incoming})
        witnesses = [
            dict(caller_id=edge["caller_id"], call_pc=edge["call_pc"], kind=edge["kind"]) for edge in incoming[:16]
        ]
        record = dict(
            callers=len(callers),
            call_edges=len(incoming),
            call_site_witnesses=witnesses,
            basis="static_call_graph_and_caller_register_lift",
        )
        context = dict(call_summaries=call_summaries(occurrence_id))
        tails = {}
        if summarize is not None and scope.get("accelerator_sites"):
            for edge in entries:
                if edge["caller_id"] == occurrence_id and edge["kind"] == "tail" and edge["callee_id"] in regions:
                    if edge["callee_id"] not in summaries:
                        summaries[edge["callee_id"]] = lifter_call(summarize, regions[edge["callee_id"]][1])
                    tails[edge["target"]] = summaries[edge["callee_id"]]
        if tails:
            context["tail_summaries"] = tails
        if not incoming:
            return dict(
                context,
                entry_context=dict(record, status="unknown", reason="no static callers in the decomposed artifact"),
            )
        tuples, unresolved = [], 0
        for edge in incoming:
            for caller_context, caller_state in caller_states(edge["caller_id"]):
                found = caller_sites(edge["caller_id"], caller_state)
                if found is None:
                    return dict(
                        context,
                        entry_context=dict(
                            record, status="unknown", reason="register lifter reports no call-site argument states"
                        ),
                    )
                state = (found.get(edge["call_pc"]) or {}).get("argument_state")
                unresolved += not isinstance(state, dict)
                state = state if isinstance(state, dict) else {}
                site = dict(caller_id=edge["caller_id"], call_pc=edge["call_pc"])
                if caller_context is not None:
                    site["caller_context"] = caller_context
                tuples.append(
                    (site, {reg: state.get(reg) if _known(state.get(reg)) else None for reg in ARGUMENT_REGISTERS})
                )
        agreed = {}
        for reg in ARGUMENT_REGISTERS:
            values = {_canonical(state[reg]) for _, state in tuples}
            if len(values) == 1 and tuples[0][1][reg] is not None:
                agreed[reg] = tuples[0][1][reg]
        distinct = {}
        for site, state in tuples:
            distinct.setdefault(_canonical(state), (state, []))[1].append(site)
        record.update(unresolved_call_sites=unresolved, distinct_argument_tuples=len(distinct), agreed_registers=agreed)
        if len(tuples) > len(incoming):
            record["caller_context_lifts"] = len(tuples)
        # Registers every caller agrees on are sound for the region's single lift.
        context["entry_state"] = agreed or None
        if len(distinct) == 1:
            status = (
                dict(status="all_callers_agree", registers=agreed)
                if agreed
                else dict(status="unknown", reason="callers pass no known argument values")
            )
        elif len(distinct) <= MAX_ENTRY_CONTEXTS:
            status = dict(status="per_caller_contexts", registers=agreed)
            context["contexts"] = [
                dict(
                    context_id="ctx-" + str(index),
                    registers={reg: value for reg, value in state.items() if value is not None},
                    call_sites=calls[:16],
                    callers=len({row["caller_id"] for row in calls}),
                )
                for index, (state, calls) in enumerate(distinct.values())
            ]
        else:
            status = dict(
                status="unknown",
                registers=agreed,
                reason=f"{len(distinct)} distinct caller argument tuples exceed {MAX_ENTRY_CONTEXTS}",
            )
        context["entry_context"] = dict(record, **status)
        return context

    return {
        scope["region_id"]: resolve(occurrence_id)
        for occurrence_id, (scope, _) in regions.items()
        if scope.get("accelerator_sites")
    }


def machine_regions(root, artifact, *, facts, decoder, structure=None, stream_max_steps=STREAM_MAX_STEPS):
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.compile.analysis.riscv_machine import region_index
    from merlin.experiments.phase1.feedback import reference_analysis as analysis
    from merlin.perf.pc_instruction_listing import normalized_text

    path = Path(root) / artifact["path"]
    # The ELF's own arch attribute wins; else the selected contract's declared host ISA.
    elf = PinnedELF(open_ordinary(path), artifact["path"], declared_isa=(decoder or {}).get("host_isa"))
    try:
        if elf.pin["sha256"] != artifact["sha256"]:
            raise ValueError("model changed after staging")
        rows = elf.instructions()
        symbols = elf_function_symbols(elf)
        table, _ = analysis._table(facts)
        opcode = table.get("custom_opcode")
        if opcode is None:
            raise ValueError("selected ISA has no derived accelerator opcode")
        indexed = region_index(rows, artifact_sha256=artifact["sha256"], custom_opcode=opcode)
        # The static call graph is decoded first so callers' argument states can
        # select each accelerator region's entry context before it is lifted.
        spans, edges = [], []
        for scope, body in indexed:
            start, stop = scope["address_range"]
            occurrence_id = artifact["id"] + ":" + scope["region_id"]
            spans.append(dict(id=occurrence_id, address=start, size_bytes=stop - start))
            edges.extend(transfer_edges(body, start, stop, occurrence_id))
        edges = resolve_callees(edges, spans)
        contexts = interprocedural_contexts(
            {artifact["id"] + ":" + scope["region_id"]: (scope, body) for scope, body in indexed}, edges
        )
        # This existing owner lifts every accelerator span, without a first-page sample.
        typed_ccas = []
        lifted = analysis._lift(
            elf,
            "elf",
            elf.pin,
            facts=facts,
            offset=0,
            limit=len(indexed),
            candidate=False,
            decoder=decoder,
            ccas=typed_ccas,
            entry_contexts=contexts,
        )
        if lifted["next_offset"] is not None:
            raise ValueError("incomplete CCA region census")
        reports = {tuple(row.get("address_range", ())): row for row in lifted["regions"]}
        cca_objects = {
            tuple(row.get("address_range", ())): cca for row, cca in zip(lifted["regions"], typed_ccas, strict=True)
        }
        outputs, bodies = [], []
        for scope, body in indexed:
            start, stop = scope["address_range"]
            (section,) = [entry for entry in elf.sections if entry[0] == scope["section"]]
            payload = elf.read_at(section[1] + start - section[3], stop - start)
            stored = store_body(root, "machine", payload, ".bin")
            bodies.append(stored)
            occurrence_id = artifact["id"] + ":" + scope["region_id"]
            directory = Path("mappings") / artifact["id"] / scope["region_id"]
            (Path(root) / directory).mkdir(parents=True, exist_ok=True)
            listing = directory / "instructions.dis"
            (Path(root) / listing).write_text(
                "Disassembly of section " + scope["section"] + ":\n" + normalized_text(body)
            )
            report = reports.get(tuple(scope["address_range"]))
            if report is None:
                report = dict(
                    kind="static_host_region",
                    listed_instructions=len(body),
                    cca=None,
                    semantic_region_status="unknown",
                    reason="no_selected_accelerator_sites; host computation semantics not reconstructed",
                )
            if scope["accelerator_sites"] and structure is not None:
                report = dict(
                    report,
                    derived_effects=region_effects(
                        report,
                        body,
                        decoder,
                        structure,
                        occurrence_id,
                        context=contexts.get(scope["region_id"]),
                        stream_max_steps=stream_max_steps,
                    ),
                )
            report_path = directory / "analysis.json"
            write_json(Path(root) / report_path, observations_only(report))
            if report.get("cca"):
                write_json(Path(root) / directory / "cca.json", report["cca"])
                from merlin.kernels.cca_mlir import to_mlir

                (Path(root) / directory / "cca.mlir").write_text(to_mlir(cca_objects[tuple(scope["address_range"])]))
            outputs.append(
                dict(
                    id=occurrence_id,
                    artifact_id=artifact["id"],
                    kernel_id=stored["id"],
                    section=scope["section"],
                    address=start,
                    size_bytes=stop - start,
                    boundary_status=scope["boundary_status"],
                    accelerator_sites=scope["accelerator_sites"],
                    listing=listing.as_posix(),
                    analysis=report_path.as_posix(),
                    source_correspondence="unknown",
                    model_layer="unknown",
                    numerical_equivalence="unknown",
                    performance="unknown",
                    containing_symbols=[
                        symbol
                        for symbol in symbols
                        if symbol["address"] <= start < stop <= symbol["address"] + symbol["size_bytes"]
                    ],
                    **symbol_overlap(symbols, start, stop),
                )
            )
        # Every selected custom issue must occur in exactly one materialized span.
        expected = sum(row["accelerator_sites"] for row in lifted["regions"])
        observed = sum(row["accelerator_sites"] for row in outputs)
        independent = independent_site_census(rows, opcode)
        if (
            observed != expected
            or observed != independent["sites"]
            or sum(len(body) for _, body in indexed) != len(rows)
        ):
            raise ValueError("accelerator decomposition coverage differs from original CCA inventory")
        return (
            bodies,
            outputs,
            edges,
            dict(
                regions=len(outputs),
                accelerator_regions=sum(row["accelerator_sites"] > 0 for row in outputs),
                accelerator_sites=observed,
                all_instructions=len(rows),
                decomposed_instructions=sum(len(body) for _, body in indexed),
                boundary_status="inferred_static_span; original ELF has no asserted source boundaries",
                complete_accelerator_site_coverage=True,
                independent_site_census=independent,
                tail_edges=sum(edge["kind"] == "tail" for edge in edges),
                entry_contexts=dict(
                    Counter(
                        (context.get("entry_context") or {}).get("status", "unknown") for context in contexts.values()
                    )
                ),
                context_observations=sum(len(context.get("contexts") or ()) for context in contexts.values()),
                call_summaries=len(
                    {target for context in contexts.values() for target in (context.get("call_summaries") or {})}
                ),
                symbol_status=dict(Counter(row["symbol_status"] for row in outputs)),
                decoder_isa=dict(elf.decoder_isa),
            ),
        )
    finally:
        elf.close()


def source_regions(root, artifact):
    from merlin.kernels.reference_inventory import inspect_source

    raw = (Path(root) / artifact["path"]).read_bytes()
    started = time.monotonic()
    inventory = inspect_source(raw, artifact["format"])
    if inventory["parser"] == "unavailable":
        raise ValueError("source parser unavailable: " + json.dumps(inventory["diagnostics"]))
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    module = ast.parse(text) if artifact["format"] == "python_schedule" else None
    definitions = {node.lineno: node for node in module.body if isinstance(node, ast.FunctionDef)} if module else {}
    bodies, outputs, edges = [], [], []
    for index, function in enumerate(inventory["functions"]):
        start = function["line"]
        node = definitions.get(start)
        if node is not None:
            start = min([start, *(decorator.lineno for decorator in node.decorator_list)])
        payload = "".join(lines[start - 1 : function["end_line"]]).encode()
        stored = store_body(
            root, artifact["format"], payload, ".py" if artifact["format"] == "python_schedule" else ".c"
        )
        bodies.append(stored)
        identity = artifact["id"] + ":source-" + str(index)
        relative = Path("mappings") / artifact["id"] / ("source-" + str(index) + ".json")
        report = dict(
            kind="source_syntax",
            parser=inventory["parser"],
            function=function,
            cca=None,
            semantic_region_status="unknown",
            machine_correspondence="unknown",
            reason="existing source parser supplies syntax; no verified source-to-command adapter for this input",
            diagnostics=inventory["diagnostics"],
            advice_generated=False,
        )
        write_json(Path(root) / relative, observations_only(report))
        outputs.append(
            dict(
                id=identity,
                artifact_id=artifact["id"],
                kernel_id=stored["id"],
                symbol=function["symbol"],
                start_line=start,
                end_line=function["end_line"],
                boundary_status="parsed_source_definition",
                accelerator_sites=None,
                analysis=relative.as_posix(),
                source_correspondence="original_exact_source_span",
                model_layer="unknown",
                numerical_equivalence="unknown",
                performance="unknown",
            )
        )
        for callee, count in function["calls"].items():
            edges.append(
                dict(
                    caller_id=identity,
                    callee_symbol=callee,
                    static_sites=count,
                    callee_id=None,
                    basis="parsed_source_call",
                    dynamic_multiplicity="unknown",
                )
            )
    exported_bindings = 0
    if module is not None:
        # Record the actual upstream factories/exports as syntax without importing Exo or choosing a schedule.
        exports = [
            dict(line=node.lineno, end_line=node.end_lineno, statement=ast.get_source_segment(text, node))
            for node in module.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        write_json(
            Path(root) / "mappings" / artifact["id"] / "module-exports.json",
            dict(kind="parsed_module_assignments", exports=exports, executed=False),
        )
        by_symbol = {row["symbol"]: row["id"] for row in outputs}

        def bindings(target, selector=()):
            if isinstance(target, ast.Name):
                yield target.id, selector
            elif isinstance(target, (ast.Tuple, ast.List)):
                for index, child in enumerate(target.elts):
                    yield from bindings(child, (*selector, index))

        for node in module.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for target in node.targets:
                for name, selector in bindings(target):
                    statement = ast.get_source_segment(text, node)
                    recipe = dict(statement=statement, result_selector=list(selector), executed=False)
                    payload = (json.dumps(recipe, sort_keys=True, indent=2) + "\n").encode()
                    stored = store_body(root, "source_export", payload, ".json")
                    bodies.append(stored)
                    identity = artifact["id"] + ":export-" + name
                    relative = Path("mappings") / artifact["id"] / ("export-" + name + ".json")
                    write_json(
                        Path(root) / relative,
                        dict(
                            kind="parsed_export_recipe",
                            recipe=recipe,
                            cca=None,
                            semantic_region_status="unknown",
                            compiled=False,
                            factory=ast.unparse(node.value.func),
                            reason="Original optimized-procedure recipe; Exo was not imported or executed.",
                        ),
                    )
                    outputs.append(
                        dict(
                            id=identity,
                            artifact_id=artifact["id"],
                            kernel_id=stored["id"],
                            symbol=name,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            boundary_status="parsed_source_export",
                            accelerator_sites=None,
                            analysis=relative.as_posix(),
                            source_correspondence="original_exact_export_recipe",
                            model_layer="unknown",
                            numerical_equivalence="unknown",
                            performance="unknown",
                        )
                    )
                    edges.append(
                        dict(
                            caller_id=identity,
                            callee_id=by_symbol.get(ast.unparse(node.value.func)),
                            callee_symbol=ast.unparse(node.value.func),
                            basis="parsed_factory_name_reference",
                            result_selector=list(selector),
                            dynamic_multiplicity="unknown",
                        )
                    )
                    exported_bindings += 1
    write_json(Path(root) / "mappings" / artifact["id"] / "source-inventory.json", observations_only(inventory))
    return (
        bodies,
        outputs,
        edges,
        dict(
            functions=len(inventory["functions"]),
            exported_bindings=exported_bindings,
            parser=inventory["parser"],
            diagnostics=observations_only(inventory["diagnostics"]),
            seconds=time.monotonic() - started,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_lift="unknown",
        ),
    )
