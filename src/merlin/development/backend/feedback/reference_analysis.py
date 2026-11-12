"""Selected, read-only analysis of author-visible references and candidate bytes.

This is an existing selfcheck diagnostic action, not an evaluator or compiler
runner. It imports no candidate code and reads no oracle, golden or score cache."""

from __future__ import annotations

import io
import json
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from merlin.development.backend.feedback import candidate_verification, reference_bank, reference_experiments
from merlin.development.backend.feedback import reference_formats as _reference_formats
from merlin.development.backend.feedback.reference_decoder import (  # noqa: F401 -- preserve established imports at this boundary
    _LISTING_DECODER_ISA,
    _instructions,
    _listing_symbol_span,
    _operand_decoder,
    _protocol_decoder_structure,
    _selected_roles_of,
    _table,
    _trip_counts,
    _without_pinned_locations,
)
from merlin.development.backend.feedback.reference_inputs import (  # noqa: F401 -- preserve established imports at this boundary
    _FORMATS,
    _MAX_BYTES,
    _MAX_ROWS,
    _MAX_TEXT,
    _AnalysisGap,
    _entry,
    _member,
    _path,
    _read,
    _reference_root,
    _relative,
)
from merlin.development.backend.feedback.reference_lifting import (  # noqa: F401 -- preserve established imports at this boundary
    _lift,
    _source_summary,
)
from merlin.development.backend.feedback.reference_results import (  # noqa: F401 -- preserve established imports at this boundary
    _GAPS,
    ACTION,
    SCHEMA,
    _analysis_summary,
    _bounded_response,
    _brief_analysis,
    _response,
)


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
):
    """Decorate a diagnostic through the existing broker without grading it."""
    from merlin.development.backend.brokers.selfcheck import _decorate_response

    body = handle(
        request,
        workspace=workspace,
        facts=facts,
        compiler_sha256=compiler_sha256,
        bundle_manifest=bundle_manifest,
        repo=repo,
        contract=contract,
        target=target,
        descriptor=descriptor,
    )
    document = _decorate_response(
        json.dumps(body), request, rid, compiler_sha256, started_ns=started_ns, completed_ns=time.time_ns()
    )
    document["official"] = False
    return _bounded_response(document, workspace=workspace, pretty=True)


def _compare(left, right, expert, ours, *, observation_fields=()):
    """Freshly analyzed operands, never author-supplied CCA summaries or requested settings."""
    from merlin.analysis.kernels.cca_compare import compare_evidence
    from merlin.analysis.kernels.kernel_evidence import compare_regions

    if observation_fields and left["kind"] not in (
        "representation_bundle",
        "static_instruction_analysis",
        "emitted_ir_analysis",
        "artifact_cross_view_analysis",
    ):
        raise ValueError("analysis_observation_fields_unsupported")

    if left["kind"] == right["kind"] == "representation_bundle":
        from merlin.development.backend.feedback.reference_representations import compare_bundles

        return compare_bundles(
            left,
            right,
            expert,
            ours,
            compare_machine=lambda *args: _compare(*args, observation_fields=observation_fields),
        )

    if left["kind"] == "captured_reference_analysis" and right["kind"] == "artifact_cross_view_analysis":
        from merlin.development.backend.feedback.reference_capture_views import compare_candidate

        return compare_candidate(left, right, expert, ours)
    if left["kind"] != right["kind"]:
        raise ValueError("analysis_compare_domains_differ")
    if left["kind"] == "artifact_cross_view_analysis":
        from merlin.development.backend.feedback.reference_views import compare_views

        return compare_views(left, right, observation_fields=observation_fields)
    if left["kind"] == "declared_semantic_region":
        return dict(
            kind="declared_semantic_comparison",
            reference=left["input"],
            candidate=right["input"],
            declared_contracts=compare_regions(expert, ours).to_dict(),
            artifact_origin="unverified",
            semantic_equivalence="unknown",
            next_action="Bind both declared contracts to the actual frontend/lowering products before ranking cycles.",
        )
    if left["kind"] == "emitted_ir_analysis":
        from merlin.analysis.kernels.instruction_comparison import compare_operand_summaries
        from merlin.analysis.kernels.instruction_observations import compare_observations, from_projection

        return dict(
            kind="emitted_instruction_comparison",
            reference=left,
            candidate=right,
            operand_comparison=compare_operand_summaries(
                left.get("operand_summary", {}), right.get("operand_summary", {})
            ),
            observation_comparison=compare_observations(
                *[
                    from_projection(
                        row.get("operand_summary", {}),
                        evidence_id=row["input"]["sha256"],
                        unknowns=("source_tensor_binding_unknown", "dynamic_multiplicity_unknown"),
                    )
                    for row in (left, right)
                ],
                requested_fields=observation_fields,
            ),
            semantic_compatibility="unknown",
            schedule_consumption="unknown",
            performance="unknown",
        )
    if left["kind"] != "static_instruction_analysis":
        raise ValueError("analysis_compare_domains_differ")
    ref, cand = left["regions"][0], right["regions"][0]
    from merlin.analysis.kernels.instruction_comparison import compare_instruction_summaries

    comparison = (
        compare_evidence(expert, ours).to_dict()
        if expert is not None and ours is not None
        else dict(
            status="unknown",
            denominator=0,
            compared_axes=[],
            divergences=[],
            unknowns=[dict(reason="selected_instruction_decoder_unavailable")],
        )
    )
    # Both complete instruction censuses were admitted above; absent instruction kinds have zero sites.
    ref_counts = Counter(ref["selected_decode"]["counts"])
    cand_counts = Counter(cand["selected_decode"]["counts"])
    command_mix = {
        name: dict(reference=ref_counts[name], candidate=cand_counts[name])
        for name in sorted(ref_counts.keys() | cand_counts.keys())
        if ref_counts.get(name) or cand_counts.get(name)
    }
    from merlin.analysis.kernels.instruction_observations import compare_observations
    from merlin.development.backend.feedback.reference_schedule_receipts import compare_schedule_receipts

    return dict(
        kind="reference_candidate_comparison",
        reference=left,
        candidate=right,
        cca=comparison,
        observation_comparison=compare_observations(
            getattr(expert, "instructions", None),
            getattr(ours, "instructions", None),
            requested_fields=observation_fields,
        ),
        instruction_comparison=compare_instruction_summaries(ref, cand),
        command_site_comparison=command_mix,
        construction_comparison=compare_schedule_receipts(
            left.get("construction_receipts", {}), right.get("construction_receipts", {})
        ),
        instruction_site_delta=cand["listed_instructions"] - ref["listed_instructions"],
        count_basis="listed_instruction_sites; no runtime multiplicity or latency implication",
        mechanism_status="unknown",
        schedule_consumption="unknown",
        semantic_compatibility="unknown",
        required_evidence=[
            "same source graph, fusion/live-outs, shapes, layouts and complete numerical contract",
            "actual schedule.plan consumer and emitted mechanism witness",
            "source-bound numerical selfcheck and matched complete-region cycle measurements",
        ],
        next_action="Compile alternatives with selected library APIs; reanalyze emission and record missing adapters.",
    )


def handle(
    request,
    *,
    workspace,
    facts=None,
    compiler_sha256=None,
    bundle_manifest=None,
    repo=None,
    contract=None,
    target=None,
    descriptor=None,
):
    """Whitelisted diagnostic entrypoint called by the existing host broker."""
    base = dict(
        schema=SCHEMA,
        all_pass=None,
        numeric_assessment="not_run",
        official=False,
        qualification_issued=False,
        semantic_equivalence="unknown",
        performance="unknown",
    )
    try:
        # A coverage bank arm's pinned precomputed bank is its whole service; every other
        # selection keeps the operator-pinned reference experiment requirement unchanged.
        bank_pin = reference_bank.selected_pin()
        selected = None if bank_pin is not None else reference_experiments._current()
        if bank_pin is None and selected is None:
            return dict(base, status="denied", reason="reference_guidance_disabled")
        packet = None if bank_pin is not None else reference_experiments._load(selected)[0]
        value = request.get("reference_analysis")
        operation = value.get("operation") if isinstance(value, dict) else None
        if operation in reference_bank.BANK_OPERATIONS and bank_pin is None:
            raise ValueError("bank_operation_requires_cca_bank_arm")
        if operation == "verify":
            # Owned before/after builds only; no bank unit or reference bytes are read.
            candidate_verification.validate(value)
            return candidate_verification.handle(
                value,
                base=base,
                workspace=Path(workspace).resolve(),
                facts=facts,
                contract=contract,
                target=target,
                descriptor=descriptor,
                compiler_sha256=compiler_sha256,
                capsules_root=candidate_verification.public_capsules_root(descriptor),
                bank=candidate_verification.bank_decoding(workspace, bundle_manifest, repo, bank_pin),
            )
        if not isinstance(value, dict) or set(value) - {
            "operation",
            "path",
            "format",
            "member",
            "offset",
            "limit",
            "symbol",
            "entry",
            "region",
            "instruction_offset",
            "candidate",
            "observation_fields",
            "shape",
            "candidate_schedule",
        }:
            raise ValueError("invalid_analysis_request")
        schedule = value.get("candidate_schedule")
        if schedule is not None:
            from merlin.development.backend.feedback.schedule_attribution import validate_path

            if value.get("operation") not in ("compare", "applicability") or value.get("observation_fields"):
                raise ValueError("invalid_analysis_request")
            validate_path(schedule)
            if bank_pin is None:
                raise ValueError("bank_operation_requires_cca_bank_arm")
        # This is an authored analysis selector: omitted format/page/location options
        # select automatic detection and the first bounded page, not observed facts.
        value = {
            "format": "auto",
            "offset": 0,
            "limit": 8,
            "instruction_offset": 0,
            "symbol": "",
            "entry": "",
            "region": "",
            **value,
        }
        operation, format = value.get("operation"), value["format"]
        observation_fields = value.get("observation_fields", [])
        if (
            not isinstance(observation_fields, list)
            or len(observation_fields) > 8
            or any(type(name) is not str or not name or len(name) > 512 for name in observation_fields)
            or observation_fields
            and operation != "compare"
        ):
            raise ValueError("invalid_analysis_request")
        offset, limit = value["offset"], value["limit"]
        if (
            operation not in ("inventory", "lift", "candidate", "compare", "applicability")
            or format not in _FORMATS
            or type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 16
        ):
            raise ValueError("invalid_analysis_operation_or_page")
        for key in ("symbol", "entry", "region"):
            if not isinstance(value[key], str) or len(value[key]) > 512:
                raise ValueError("invalid_analysis_request")
        if type(value["instruction_offset"]) is not int or value["instruction_offset"] < 0:
            raise ValueError("invalid_analysis_request")
        other = value.get("candidate")
        if operation in ("compare", "applicability") and other is None and schedule is not None:
            pass  # the top-down schedule alone is the candidate side
        elif operation in ("compare", "applicability"):
            # The candidate has the same optional selector defaults, but its path is mandatory.
            if isinstance(other, dict):
                other = {"format": "auto", "symbol": "", "region": "", "instruction_offset": 0, "offset": 0, **other}
            if (
                not isinstance(other, dict)
                or set(other) - {"path", "format", "symbol", "offset", "region", "instruction_offset"}
                or not isinstance(other.get("path"), str)
                or other["format"] not in _FORMATS
                or not isinstance(other["symbol"], str)
                or len(other["symbol"]) > 512
                or not isinstance(other["region"], str)
                or len(other["region"]) > 512
                or type(other["instruction_offset"]) is not int
                or other["instruction_offset"] < 0
                or type(other["offset"]) is not int
                or other["offset"] < 0
            ):
                raise ValueError("invalid_analysis_request")
        elif other or operation == "candidate" and value.get("entry"):
            raise ValueError("invalid_analysis_request")
        workspace = Path(workspace).resolve()
        if bank_pin is not None:
            return reference_bank.handle(
                value,
                base=base,
                workspace=workspace,
                bundle_manifest=bundle_manifest,
                repo=repo,
                pin=bank_pin,
                facts=facts,
                contract=contract,
                target=target,
                descriptor=descriptor,
                compiler_sha256=compiler_sha256,
                observation_fields=observation_fields,
            )
        # A before/after baseline is candidate evidence too. It cannot acquire
        # a map entry's authority merely by occupying the left comparison slot.
        own_left = operation == "candidate" or (
            operation == "compare"
            and not value.get("entry")
            and isinstance(value.get("path"), str)
            and value["path"].split("/", 1)[0] != "reference-kernels"
        )
        reference_root = _reference_root(workspace, bundle_manifest, repo) if not own_left else None
        if operation != "inventory":
            left, expert = _analyze(
                value,
                workspace=workspace,
                reference_root=reference_root,
                facts=facts,
                candidate=own_left,
                compare=operation == "compare",
                contract=contract,
                target=target,
                descriptor=descriptor,
                candidate_build_manifest=bundle_manifest,
            )
            left["input_role"] = "candidate_artifact" if own_left else "selected_reference"
            if operation == "compare":
                right, ours = _analyze(
                    other,
                    workspace=workspace,
                    reference_root=None,
                    facts=facts,
                    candidate=True,
                    compare=True,
                    contract=contract,
                    target=target,
                    descriptor=descriptor,
                    candidate_build_manifest=bundle_manifest,
                )
                right["input_role"] = "candidate_artifact"
                left = _compare(left, right, expert, ours, observation_fields=observation_fields)
            return _response(base, packet, left, compiler_sha256, workspace=workspace)
        path, label = _path(
            workspace,
            (value["path"] if "path" in value else "reference-kernels"),
            candidate=operation == "candidate",
            reference_root=reference_root,
        )
        if operation == "inventory" and path.is_dir():
            entries = sorted(p for p in path.rglob("*") if p.is_file() or p.is_symlink())
            if len(entries) > 4096:
                raise ValueError("visible_reference_inventory_exceeds_bound")
            page = []
            for entry in entries[offset : offset + limit]:
                relative = (PurePosixPath(label) / entry.relative_to(path)).as_posix()
                _path(workspace, relative, reference_root=reference_root)
                payload, pin = _read(entry, relative, allow_elf=True)
                from merlin.analysis.artifact_reader import PinnedELF

                if isinstance(payload, PinnedELF):
                    payload.close()
                page.append(pin)
            report = dict(
                kind="visible_reference_files",
                total=len(entries),
                files=page,
                next_offset=offset + len(page) if offset + len(page) < len(entries) else None,
            )
        elif path.suffix.lower() == ".zip" and not value.get("member"):
            raw, pin = _read(path, label)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                report = dict(
                    kind="visible_archive_members",
                    input=pin,
                    total=len(entries),
                    members=[
                        dict(member=r.filename[:1024], size_bytes=r.file_size, directory=r.is_dir())
                        for r in entries[offset : offset + limit]
                    ],
                    next_offset=offset + limit if offset + limit < len(entries) else None,
                )
        else:
            report, _ = _analyze(
                value,
                workspace=workspace,
                reference_root=reference_root,
                facts=facts,
                contract=contract,
                target=target,
                descriptor=descriptor,
            )
        return _response(base, packet, report, compiler_sha256, workspace=workspace)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
        RecursionError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as error:
        code = (
            str(error)
            if isinstance(error, ValueError) and str(error) in _GAPS
            else (
                "analysis_dependency_unavailable"
                if isinstance(error, ImportError)
                else "analysis_input_unavailable_invalid_or_unsupported"
            )
        )
        return dict(
            base,
            status="unknown",
            reason=code,
            next_action=_GAPS[code],
            gap=dict(
                kind="evidence_or_adapter_gap",
                code=code,
                retryable=code == "analysis_input_changed",
                **(dict(detail=error.detail) if isinstance(error, _AnalysisGap) else {}),
            ),
        )


def _analyze(
    value,
    *,
    workspace,
    reference_root,
    facts,
    candidate=False,
    compare=False,
    contract=None,
    target=None,
    descriptor=None,
    candidate_build_manifest=None,
):
    """Keep the original live operand-decoder selection at the diagnostic boundary."""
    return _reference_formats._analyze(
        value,
        workspace=workspace,
        reference_root=reference_root,
        facts=facts,
        candidate=candidate,
        compare=compare,
        contract=contract,
        target=target,
        descriptor=descriptor,
        candidate_build_manifest=candidate_build_manifest,
        _operand_decoder=lambda *args, **kwargs: _operand_decoder(*args, **kwargs),
    )


def _analyze_open(
    value,
    *,
    raw,
    pin,
    label,
    entry,
    workspace,
    reference_root,
    facts,
    candidate=False,
    compare=False,
    contract=None,
    target=None,
    descriptor=None,
    candidate_build_manifest=None,
):
    """Keep the original live operand-decoder selection at the diagnostic boundary."""
    return _reference_formats._analyze_open(
        value,
        raw=raw,
        pin=pin,
        label=label,
        entry=entry,
        workspace=workspace,
        reference_root=reference_root,
        facts=facts,
        candidate=candidate,
        compare=compare,
        contract=contract,
        target=target,
        descriptor=descriptor,
        candidate_build_manifest=candidate_build_manifest,
        _operand_decoder=lambda *args, **kwargs: _operand_decoder(*args, **kwargs),
    )


def _representation_bundle(document, pin, *, workspace, reference_root, facts, candidate, contract, target, descriptor):
    """Keep the original live operand-decoder selection at the diagnostic boundary."""
    return _reference_formats._representation_bundle(
        document,
        pin,
        workspace=workspace,
        reference_root=reference_root,
        facts=facts,
        candidate=candidate,
        contract=contract,
        target=target,
        descriptor=descriptor,
        _operand_decoder=lambda *args, **kwargs: _operand_decoder(*args, **kwargs),
    )
