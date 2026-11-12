"""Verify an author's own before/after builds from their emitted artifacts; never grade or advise.

The bank arm's ``--reference-analysis verify`` reads only owned ``submission/``, ``selfcheck_out/``,
``artifacts/`` and ``variants/`` paths: the ordinary selfcheck outputs (``lowered.llvm.mlir``,
``command_buffer.json``, optional ``package_kernel.elf`` or one author-built ``*.o``) and, when the
author kept them, selfcheck responses. Proved-stream facts come from the ELF, else that object, else
the build's own ``lowered.llvm.mlir`` compiled host-side with ``-mno-relax``. Emitted LLVM is decoded
again by the host's selected ISA decoder (RTL facts plus the contract's operand protocol); instruction
roles come from the same selected roster as ``decoded_effects``. Nothing here edits a compiler, ranks
a setting or extrapolates beyond the listed builds. Each verdict is separate:

* ``numerically_correct``: the selfcheck verdict row of that capsule, when a response is present;
* ``intervention_applied``: ``applied`` only when the emitted stream differs from the before build
  and the declared mechanism's derived effect differs; an identical stream, or an unchanged declared
  effect, is ``optimization_did_not_apply_emitted_fallback``; otherwise ``unknown``. With no declared
  mechanism, a changed stream is ``applied`` only when a stream mechanism measured on both builds' proved
  streams differs (those are listed, none attributed); else ``unknown`` names each one not measured;
* ``traffic_reduced``: proved per-call executions (static sites x proved trip counts) of the
  movement-in roles per accumulate-role execution; any unproved site keeps it ``unknown``.

Storage validity reports the schedule-dialect verifier over emitted schedule records and
lexical local-address window overlaps from ``derive_effects``; neither is dynamic proof."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from merlin.development.backend.feedback import candidate_artifacts as _candidate_artifacts
from merlin.development.backend.feedback import candidate_comparison as _candidate_comparison
from merlin.development.backend.feedback.candidate_artifacts import (  # noqa: F401 -- preserve established imports at this boundary
    _COMPILE_TIMEOUT_S,
    _MAX_CASES,
    COMMANDS,
    ELF,
    LOWERED,
    OBJECT,
    OBJECTS_AMBIGUOUS,
    OWNED,
    _elf_stream,
    _files,
    _read,
    case_dirs,
    decode_lowered,
    object_recipe,
    owned_path,
)
from merlin.development.backend.feedback.candidate_comparison import (  # noqa: F401 -- preserve established imports at this boundary
    _DIFFERS,
    _EXTRA_FLAGS,
    APPLIED,
    FALLBACK,
    UNKNOWN,
    _applied,
    _baseline_entry,
    _brief,
    _digest,
    _dimensions,
    _flag,
    _identical,
    _region_by_dimensions,
    _routes,
    _stream_applied,
    _stream_fact_names,
    _undeclared_applied,
    declared_shape,
    numerical,
    regressions,
    stream_changes,
    stream_facts,
    verdict_rows,
)
from merlin.development.backend.feedback.candidate_effects import (  # noqa: F401 -- preserve established imports at this boundary
    COMPUTE,
    DATA_MOVEMENT,
    MOVEMENT_IN,
    _element_bytes,
    _ratio,
    _round,
    _storage_status,
    capacities,
    capacity_conditions,
    effects,
    mechanisms,
    receipts,
    traffic,
    verify_schedule,
    window_overlaps,
)
from merlin.development.backend.feedback.candidate_request import (  # noqa: F401 -- preserve established imports at this boundary
    _MAX_REPLY,
    _NOT_ESTABLISHED,
    GAPS,
    SCHEMA,
    _contract_document,
    _raised,
    _summary,
    bank_decoding,
    public_capsules_root,
    validate,
)


def analyze_build(files, *, decoder=None, structure=None, evidence_id="build", symbol=None, build=None):
    """One build record from its pinned emitted artifacts (all optional; missing stays unknown).

    The proved command stream comes from the first available form of the author's own build: the linked
    ``package_kernel.elf``, else one owned relocatable object in the build directory, else the build's
    own ``lowered.llvm.mlir`` compiled host-side with the selected object recipe and ``-mno-relax``
    (``build``: :func:`object_recipe`). ``symbol`` is the entry the stream is proved from."""
    pins, record = [], dict(evidence_id=evidence_id)
    command_buffer = None
    if COMMANDS in files:
        raw, pin = _read(*files[COMMANDS])
        pins.append(pin)
        try:
            command_buffer = json.loads(raw)
        except ValueError:
            record["command_buffer_status"] = "unparseable"
    for name in (ELF, OBJECT):
        if name in files:
            raw, pin = _read(*files[name])
            pins.append(pin)
            record["elf" if name == ELF else "object"] = pin
    trace, lowered = None, None
    if LOWERED in files:
        raw, pin = _read(*files[LOWERED])
        pins.append(pin)
        record["emitted_sha256"] = pin["sha256"]
        lowered = raw
        trace = decode_lowered(raw.decode("utf-8", "replace"), decoder)
    stream, reason, source = _build_stream(files, lowered, decoder, symbol, build)
    record["stream_source"] = source
    return build_record(
        trace,
        decoder=decoder,
        structure=structure,
        command_buffer=command_buffer,
        evidence_id=evidence_id,
        inputs=pins,
        extra=record,
        stream_commands=stream,
        stream_reason=reason,
    )


def _build_stream(files, lowered, decoder, symbol, build):
    """``(decoded commands, reason, source)`` of the build's first available binary form."""
    from merlin.analysis.kernels.mechanism_applicability import NO_STREAM

    if not symbol:
        return (
            None,
            NO_STREAM + ": no kernel entry symbol (candidate.symbol, or the contract's declared entry)",
            dict(kind=None, symbol=None),
        )
    for name, kind in ((ELF, "linked_elf"), (OBJECT, "owned_relocatable_object")):
        if name in files:
            stream, reason = _elf_stream(files[name], decoder, symbol)
            return stream, reason, dict(kind=kind, path=files[name][1], symbol=symbol)
    if lowered is None:
        detail = (
            "several relocatable objects; name one with --after <dir>/<file>.o"
            if OBJECTS_AMBIGUOUS in files
            else "no package_kernel.elf, relocatable object or lowered.llvm.mlir"
        )
        return None, f"{NO_STREAM}: {detail} in this build", dict(kind=None, symbol=symbol)
    return _compiled_stream(lowered, decoder, symbol, build)


def build_record(
    trace,
    *,
    decoder=None,
    structure=None,
    command_buffer=None,
    evidence_id="build",
    inputs=(),
    extra=None,
    stream_commands=None,
    stream_reason=None,
):
    """Pure core over an (already decoded) trace and command buffer; used by tests and real inputs.

    ``stream_commands``: the decoded commands of a proved dynamic execution, when one exists; width,
    buffering and overlap facts come only from it and otherwise stay unknown."""
    record = dict(extra or {}, evidence_id=evidence_id, inputs=list(inputs))
    record["stream_facts"] = stream_facts(stream_commands, structure, evidence_id, stream_reason)
    # Identity of the issued commands (class and decoded fields in issue order; code addresses excluded).
    record["proved_stream_sha256"] = (
        None if stream_commands is None else _digest([(e.get("class"), e.get("fields")) for e in stream_commands])
    )
    roles = ((decoder or {}).get("structure") or {}).get("instruction_roles")
    if trace is None:
        record.update(
            stream=dict(status="unknown", reason="emitted_ir_or_selected_decoder_unavailable"),
            traffic=None,
            effects=None,
        )
    else:
        rows = trace["instructions"]
        stream = [(row.get("class"), row.get("decoded"), row.get("executions")) for row in rows]
        record["stream"] = dict(
            status="decoded",
            sites=len(rows),
            sha256=_digest(stream),
            classes=dict(Counter(row.get("class") for row in rows)),
        )
        record["traffic"] = traffic(rows, roles) if roles else None
        derived = effects(rows, decoder or {}, structure, evidence_id) if roles else None
        record["effects"] = None if derived is None else derived["facts"]
        loops = (trace.get("summary") or {}).get("counted_loops") or {}
        record["loop_proof"] = dict(
            basis=loops.get("basis"),
            unavailable=[u for f in loops.get("functions") or () for u in f.get("unavailable") or ()][:8],
        )
    found = receipts(command_buffer)
    verifiers = [
        dict(family=r["family"], **verify_schedule(r["schedule_ir"]["text"], r["schedule_ir"]["sha256"]))
        for r in found
        if "schedule_ir" in r
    ]
    for r in found:
        r.pop("schedule_ir", None)
    record["receipts"] = found
    record["storage"] = dict(
        library_verifiers=verifiers
        or [
            dict(
                verifier="schedule_dialect.verify",
                status="not_run",
                reason="no emitted schedule record in the command buffer",
            )
        ],
        local_address_windows=window_overlaps(record.get("effects")),
    )
    if command_buffer is not None:
        record["declared_tensors"] = {
            name: dict(shape=t.get("shape"), dtype=t.get("dtype"), role=t.get("role"))
            for name, t in (command_buffer.get("tensors") or {}).items()
            if isinstance(t, dict)
        }
    return record


def compare(before, after, *, mechanism=None, verdicts=None):
    """Separate verdicts for one before/after pair of owned builds."""
    delta = {}
    if before.get("traffic") and after.get("traffic"):
        for role in sorted(set(before["traffic"]["per_role"]) | set(after["traffic"]["per_role"])):
            b, a = before["traffic"]["per_role"].get(role, {}), after["traffic"]["per_role"].get(role, {})
            d = dict(
                static_before=b.get("static_sites") if role in before["traffic"]["per_role"] else 0,
                static_after=a.get("static_sites") if role in after["traffic"]["per_role"] else 0,
                dynamic_before=b.get("dynamic") if role in before["traffic"]["per_role"] else 0,
                dynamic_after=a.get("dynamic") if role in after["traffic"]["per_role"] else 0,
            )
            d["static_delta"] = (
                d["static_after"] - d["static_before"] if None not in (d["static_before"], d["static_after"]) else None
            )
            known = None not in (d["dynamic_before"], d["dynamic_after"])
            d["dynamic_delta"] = d["dynamic_after"] - d["dynamic_before"] if known else None
            delta[role] = d
    ratios = {}
    for basis in ("static", "dynamic"):
        rb = ((before.get("traffic") or {}).get("ratios") or {}).get(basis) or {}
        ra = ((after.get("traffic") or {}).get("ratios") or {}).get(basis) or {}
        ratios[basis] = dict(
            before=rb.get("loads_per_compute"),
            after=ra.get("loads_per_compute"),
            **{
                key: dict(before=rb.get(key), after=ra.get(key))
                for key in ("data_movement_loads_per_compute", "preloads_per_compute")
            },
        )
    dyn = ratios["dynamic"]
    if None in (dyn["before"], dyn["after"]):
        reduced = dict(
            status="unknown",
            reason="dynamic loads per compute not proved on both builds",
            static_before=ratios["static"]["before"],
            static_after=ratios["static"]["after"],
        )
    else:
        reduced = dict(
            status="reduced" if dyn["after"] < dyn["before"] else "not_reduced",
            before=dyn["before"],
            after=dyn["after"],
            basis="proved dynamic loads per compute",
        )
    applied = _applied(before, after, mechanism)
    storage = dict(
        before=_storage_status(before),
        after=_storage_status(after),
        before_detail=before["storage"],
        after_detail=after["storage"],
    )
    changes = stream_changes(before, after)
    return dict(
        numerically_correct=verdicts or dict(status="unknown", reason="no selfcheck verdict row supplied"),
        intervention_applied=applied,
        traffic_reduced=reduced,
        storage_valid=storage,
        traffic_delta=dict(per_role=delta, loads_per_compute=ratios),
        stream_changes=changes,
        regressions=regressions(before, after, changes),
        route_receipts=dict(before=_routes(before), after=_routes(after)),
        independence=(
            "numerical correctness, intervention application and traffic are separate; "
            "a correct fallback build is not evidence of optimization transfer"
        ),
    )


def verify(workspace, value, *, decoder=None, structure=None, facts=None, capsules_root=None, build=None):
    """The full verify report for one validated request value.

    ``build``: :func:`object_recipe` of the selected contract (declared entry symbol and object flags);
    ``candidate.symbol`` overrides the declared entry."""
    mechanism = value.get("mechanism") or None
    if mechanism is not None and mechanism not in mechanisms():
        raise ValueError("verify_mechanism_not_a_derived_effect")
    before_cases = case_dirs(workspace, value["path"])
    after_cases = case_dirs(workspace, value["candidate"]["path"])
    baseline_cases = case_dirs(workspace, value["baseline"]) if value.get("baseline") else {}
    symbol = value["candidate"].get("symbol") or (build or {}).get("entry_symbol") or None
    symbol_source = (
        "candidate.symbol"
        if value["candidate"].get("symbol")
        else "selected contract declared entry"
        if symbol
        else None
    )
    build_recipe = build
    sweep = {}
    for item in value.get("sweep") or ():
        for capsule, entry in case_dirs(workspace, item).items():
            sweep.setdefault(capsule, entry)
    verdicts = verdict_rows(workspace, value.get("verdicts"))
    spaces = capacities(facts)
    cache = {}

    def build(entry):
        if entry[1] not in cache:
            cache[entry[1]] = analyze_build(
                _files(*entry),
                decoder=decoder,
                structure=structure,
                evidence_id=entry[1],
                symbol=symbol,
                build=build_recipe,
            )
        return cache[entry[1]]

    pairs = sorted(set(before_cases) & set(after_cases))
    if len(before_cases) == 1 and len(after_cases) == 1 and not pairs:
        pairs = [(next(iter(before_cases)), next(iter(after_cases)))]
    else:
        pairs = [(name, name) for name in pairs]
    if not pairs:
        raise ValueError("verify_before_after_share_no_capsule")
    results = []
    for left, right in pairs:
        b, a = build(before_cases[left]), build(after_cases[right])
        row = compare(b, a, mechanism=mechanism, verdicts=numerical(verdicts.get(right)))
        base = _baseline_entry(baseline_cases, right)
        if base is not None:
            row["against_baseline"] = _against(build(base), a, base[1])
        results.append(
            dict(
                capsule=right,
                before=before_cases[left][1],
                after=after_cases[right][1],
                before_inputs=b["inputs"],
                after_inputs=a["inputs"],
                fell_back_to_seed_or_before=_identical(a, b, build(base) if base else None),
                **row,
            )
        )
    report = dict(
        schema=SCHEMA,
        kind="candidate_verification",
        mechanism=mechanism,
        accepted_mechanisms=list(mechanisms()),
        pairs=results,
        entry_symbol=dict(symbol=symbol, source=symbol_source),
        decoder_inputs=list((decoder or {}).get("decoder_inputs") or []),
        not_established=list(_NOT_ESTABLISHED),
        advisory_only=True,
        qualification_issued=False,
    )
    if sweep:
        report["sweep"] = sweep_table(
            sweep,
            before_cases,
            build,
            mechanism=mechanism,
            verdicts=verdicts,
            spaces=spaces,
            capsules_root=capsules_root,
            baseline_cases=baseline_cases,
        )
    if baseline_cases:
        report["baseline"] = value["baseline"]
    return report


def _against(baseline, after, label):
    """Regression flags of ``after`` against the author-supplied seed build of the same capsule."""
    changes = stream_changes(baseline, after)
    return dict(baseline=label, regressions=regressions(baseline, after, changes), stream_changes=changes)


def _selected(contract, facts, target, descriptor):
    from merlin.development.backend.feedback import reference_analysis as A

    decoder = A._operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True)
    if not decoder or decoder.get("semantics") is None:
        return decoder, None
    from merlin.analysis.kernels.decoded_effects import selected_structure

    return decoder, selected_structure(decoder, _contract_document(contract, descriptor))


def handle(
    value,
    *,
    base,
    workspace,
    facts=None,
    contract=None,
    target=None,
    descriptor=None,
    compiler_sha256=None,
    capsules_root=None,
    bank=None,
):
    """Serve one verify request: a bounded reply plus a hash-pinned detail file in the workspace.

    ``bank``: ``(contract, facts)`` of the selected bank's pinned decoder (:func:`bank_decoding`); it
    replaces the descriptor's selection for decoding only."""
    from merlin.verification.lean_contract import document_digest

    validate(value)
    workspace = Path(workspace).resolve()
    if bank is not None:
        contract, facts, descriptor = bank[0], bank[1], None
    decoder, structure = _selected(contract, facts, target, descriptor)
    build = object_recipe(_contract_document(contract, descriptor), target)
    report = verify(
        workspace, value, decoder=decoder, structure=structure, facts=facts, capsules_root=capsules_root, build=build
    )
    report["decoder_selection"] = "cca_bank_decoder" if bank is not None else "selected_contract"
    identity = document_digest(report)
    full = dict(
        base, status="analyzed", report=report, report_identity=identity, current_compiler_sha256=compiler_sha256
    )
    from merlin.development.backend.feedback.result_details import compact_row

    detail = compact_row(dict(name="candidate_verification", **full), workspace=workspace)["detail"]
    reply = dict(
        base,
        status="analyzed",
        report=_summary(report),
        report_identity=identity,
        detail=detail,
        report_in_detail=True,
        current_compiler_sha256=compiler_sha256,
        next_action="Read the hash-pinned detail for every role, effect, receipt and sweep row.",
    )
    if len(json.dumps(reply, indent=2).encode()) > _MAX_REPLY:
        reply["report"] = dict(kind=report["kind"], projection="read full detail; summary exceeded reply bound")
    return reply


def broker_response(
    request,
    *,
    workspace,
    facts,
    compiler_sha256,
    rid,
    started_ns,
    bundle_manifest=None,
    repo=None,
    contract=None,
    target=None,
    descriptor=None,
    capsules_root=None,
):
    """The broker-decorated verify reply; failures are explicit gaps, never a pass."""
    import time

    from merlin.development.backend.brokers.selfcheck import _decorate_response
    from merlin.development.backend.feedback import reference_bank

    base = dict(
        schema="merlin.reference_analysis.v1",
        all_pass=None,
        numeric_assessment="not_run",
        official=False,
        qualification_issued=False,
        semantic_equivalence="unknown",
        performance="unknown",
    )
    try:
        pin = reference_bank.selected_pin()
        if pin is None:
            raise ValueError("verify_requires_cca_bank_arm")
        validate(request.get("reference_analysis"))
        body = handle(
            request.get("reference_analysis"),
            base=base,
            workspace=workspace,
            facts=facts,
            contract=contract,
            target=target,
            descriptor=descriptor,
            compiler_sha256=compiler_sha256,
            capsules_root=capsules_root,
            bank=bank_decoding(workspace, bundle_manifest, repo, pin),
        )
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        code = (
            str(error)
            if str(error) in GAPS
            else (
                "analysis_dependency_unavailable"
                if isinstance(error, ImportError)
                else "analysis_input_unavailable_invalid_or_unsupported"
            )
        )
        body = dict(
            base,
            status="unknown",
            reason=code,
            next_action=(GAPS[code] if code in GAPS else "Check owned paths, the selected decoder inputs and bounds."),
            gap=dict(kind="evidence_or_adapter_gap", code=code, retryable=False, detail=str(error)[:300]),
        )
    document = _decorate_response(
        json.dumps(body), request, rid, compiler_sha256, started_ns=started_ns, completed_ns=time.time_ns()
    )
    document["official"] = False
    return document


def _compiled_stream(lowered, decoder, symbol, build):
    """Retain the original diagnostic policy and live decoder callbacks."""
    return _candidate_artifacts._compiled_stream(
        lowered, decoder, symbol, build, _elf_stream=lambda *args, **kwargs: _elf_stream(*args, **kwargs)
    )


def sweep_table(sweep, before_cases, build, *, mechanism, verdicts, spaces, capsules_root, baseline_cases=None):
    """Retain the original diagnostic policy and live decoder callbacks."""
    return _candidate_comparison.sweep_table(
        sweep,
        before_cases,
        build,
        mechanism=mechanism,
        verdicts=verdicts,
        spaces=spaces,
        capsules_root=capsules_root,
        baseline_cases=baseline_cases,
        verify=lambda *args, **kwargs: verify(*args, **kwargs),
        compare=lambda *args, **kwargs: compare(*args, **kwargs),
        _against=lambda *args, **kwargs: _against(*args, **kwargs),
    )
