"""Serve one pinned, precomputed CCA bank to a coverage bank arm; never lift raw references.

The host author profile (inherited by the broker) selects this mode and pins the bank: the
bundle's single read-only ``cca-evidence`` grant must hold one condition payload written by
``merlin.mining.reference_subsets`` whose ``bank.json`` has exactly that SHA256 and verifies
completely. Units, occurrences and locators are derived at run time from that verified bank
(``bank.json``, unit cards and occurrence records); no separate map or curated list exists.

Responses carry those ahead-of-time records verbatim with their unknowns. Nothing is ranked,
recommended or summarized into a setting. Any reference-side request naming a raw
``reference-kernels/`` file, region or symbol, or a unit outside the bank, is refused as
``outside_selected_bank``; on-demand CCA of raw references is never computed in this mode.
Candidate-side analysis of owned files is unchanged and uses the ordinary owner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from merlin.verification.lean_contract import document_digest

GRANT = "cca-evidence"
RAW = "reference-kernels"
OUTSIDE = "outside_selected_bank"
#: Operations served by a CCA bank arm only; every other arm refuses them before reading anything.
BANK_OPERATIONS = ("applicability", "verify")
_REPORTS = ("analysis.json", "cca.json", "cca.mlir", "instructions.dis")
GAPS = {
    OUTSIDE: "This arm serves precomputed CCA only for the units of its selected bank. List them with "
    "--reference-analysis inventory and select one with --reference-entry UNIT_ID (and "
    "--reference-region REGION_ID for one occurrence) or a cca-evidence/ path. Raw reference-kernels/ "
    "files stay readable as ordinary files; no on-demand CCA is computed for them.",
    "bank_operation_requires_cca_bank_arm": "applicability and verify are served by a CCA bank arm's "
    "analysis service only.",
    "unknown_or_ambiguous_unit_id": "No bank unit has this ID, or the prefix matches several units; the "
    "reply lists the matching candidates. Use a full UNIT_ID or a unique prefix from inventory.",
    "cca_bank_requires_original_bundle": "Reference analysis needs the run's original host bundle manifest.",
    "cca_bank_requires_one_readonly_grant": "The bundle must carry exactly one read-only cca-evidence bank grant.",
    "cca_bank_alias_does_not_match_original_grant": "cca-evidence/ must be the original granted bank alias.",
    "cca_bank_pin_mismatch": "The granted bank differs from the profile's pinned bank.",
    "cca_bank_unverified": "A bank file, the unit roster or a file pin failed verification; the bank is not served.",
    "cca_bank_record_inconsistent": "A unit's precomputed CCA JSON, MLIR and analysis records disagree.",
    "cca_bank_dependency_unavailable": "The broker environment lacks Merlin's mining workflow "
    "(install 'merlin[mining]') needed to verify the bank.",
}


class _Refused(ValueError):
    pass


def selected_pin():
    """The pinned bank of the host's selected author profile; None outside a bank arm."""
    from ..author_profiles import current

    profile = current()
    return None if profile is None else profile.bank_sha256


def verify(root, pin):
    """Verify one whole bank payload against its pin; return the bank record."""
    root = Path(root)
    if any(part.is_symlink() for part in (root, *root.parents)) or not root.is_dir():
        raise _Refused("cca_bank_unverified")
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in root.rglob("*")):
        raise _Refused("cca_bank_unverified")
    try:
        from merlin.mining.reference_subsets import verify_bank
    except ImportError as error:
        raise _Refused("cca_bank_dependency_unavailable") from error
    if not (root / "bank.json").is_file():
        raise _Refused("cca_bank_unverified")
    if hashlib.sha256((root / "bank.json").read_bytes()).hexdigest() != pin:
        raise _Refused("cca_bank_pin_mismatch")
    try:
        bank = verify_bank(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise _Refused("cca_bank_unverified") from error
    if not bank.get("cca_enabled") or not bank.get("units"):
        raise _Refused("cca_bank_unverified")
    return bank


def open_bank(workspace, bundle_manifest, repo, pin):
    """Resolve the original read-only bank grant to frozen bytes and verify it."""
    if bundle_manifest is None or repo is None:
        raise _Refused("cca_bank_requires_original_bundle")
    import yaml

    from merlin.targetgen.sandbox import bwrap as BW

    manifest = Path(bundle_manifest)
    if (
        manifest.is_relative_to(workspace)
        or manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > 4 << 20
    ):
        raise _Refused("cca_bank_requires_original_bundle")
    bundle = yaml.safe_load(manifest.read_text())
    grants = [row for row in bundle.get("allowed", []) if row.get("as") == GRANT]
    if len(grants) != 1 or grants[0].get("mode") != "ro":
        raise _Refused("cca_bank_requires_one_readonly_grant")
    source = BW.resolve_grant(grants[0]["path"], Path(repo)).absolute()
    try:
        [frozen] = BW.snapshot_input_paths(workspace, bundle, [source], repo=Path(repo))
    except RuntimeError as error:
        raise _Refused("cca_bank_unverified") from error
    alias = workspace / GRANT
    if not alias.is_symlink() or alias.readlink() != source:
        raise _Refused("cca_bank_alias_does_not_match_original_grant")
    return frozen, verify(frozen, pin)


def _occurrences(unit):
    """Occurrence locators derived from the unit's own pinned report paths."""
    groups = {}
    for relative in unit["reports"]:
        path = PurePosixPath(relative)
        if path.name not in _REPORTS:
            raise _Refused("cca_bank_unverified")
        groups.setdefault(path.parent, {})[path.name] = relative
    rows = []
    for parent, files in groups.items():
        if set(files) != set(_REPORTS):
            raise _Refused("cca_bank_unverified")
        rows.append(dict(region_id=parent.name, original_id=parent.parent.name, files=files))
    return sorted(rows, key=lambda row: (row["original_id"], row["region_id"]))


def _pinned(root, bank, relative):
    from . import reference_analysis as A

    raw, pin = A._read(root / relative, GRANT + "/" + relative)
    if pin["sha256"] != bank["files"].get(relative):
        raise _Refused("cca_bank_unverified")
    return raw, pin


def _reference(value):
    """Classify the reference side syntactically, before any bank or raw byte is read."""
    path = value.get("path") or ""
    first = path.split("/", 1)[0]
    if value.get("operation") == "inventory" and not path:
        return True
    if first == RAW and (path != RAW or value.get("operation") != "inventory"):
        raise _Refused(OUTSIDE)
    selected = first == GRANT or bool(value.get("entry")) or path == RAW
    if selected and value.get("symbol"):
        raise _Refused(OUTSIDE)  # Precomputed units carry no symbol selector of a raw executable.
    return selected


def _select(value, bank):
    """One admitted unit (and optional occurrence) named by unit ID or a cca-evidence/ path."""
    units = {unit["kernel_id"]: unit for unit in bank["units"]}
    entry, path, region = value.get("entry") or "", value.get("path") or "", value.get("region") or ""
    if value.get("member") or value.get("format", "auto") != "auto" or value.get("instruction_offset", 0):
        raise ValueError("invalid_analysis_request")
    unit = units.get(entry) if entry else None
    if entry and unit is None:
        # A unique prefix names its unit; anything else is an unknown or ambiguous ID, not an outside unit.
        matches = sorted(name for name in units if name.startswith(entry))
        if len(matches) != 1:
            from .reference_analysis import _AnalysisGap

            raise _AnalysisGap("unknown_or_ambiguous_unit_id", candidates=matches[:8], matching=len(matches))
        unit = units[matches[0]]
    if path.startswith(GRANT + "/"):
        relative = path[len(GRANT) + 1 :]
        owners = []
        for candidate in bank["units"]:
            if relative == candidate["card"]:
                owners.append((candidate, None))
                continue
            for row in _occurrences(candidate):
                folder = PurePosixPath(row["files"]["analysis.json"]).parent.as_posix()
                if relative == folder or relative in row["files"].values():
                    owners.append((candidate, row))
        if len(owners) != 1 or unit is not None and owners[0][0] is not unit:
            raise _Refused(OUTSIDE)
        unit, occurrence = owners[0]
        if occurrence is not None:
            if region and region != occurrence["region_id"]:
                raise _Refused(OUTSIDE)
            region = occurrence["region_id"]
    elif path and path != GRANT and not entry:
        raise _Refused(OUTSIDE)
    if unit is None:
        raise _Refused(OUTSIDE)
    occurrences = _occurrences(unit)
    if region:
        occurrences = [row for row in occurrences if row["region_id"] == region]
        if len(occurrences) != 1:
            raise _Refused(OUTSIDE)
    return unit, occurrences


def _bank_record(bank, pin):
    return dict(
        sha256=pin,
        label=bank.get("label"),
        schema=bank.get("schema"),
        semantics=bank.get("semantics"),
        units=len(bank["units"]),
        served="precomputed_selected_units_only",
        on_demand_analysis="not_performed",
    )


def _extents(analysis):
    """Shape-side facts of one occurrence: proved loop trip counts, transfer shapes per role, entry symbol."""
    trips = ((analysis.get("cfg") or {}).get("trip_counts") or {}).get("proved") or []
    width = ((analysis.get("derived_effects") or {}).get("facts") or {}).get("role_transfer_width") or {}
    shapes = {}
    for role, entry in ((width.get("value") or {}).get("data_movement") or {}).items():
        rows = [
            dict(rows=row.get("rows"), cols=row.get("cols"), count=row.get("count"))
            for row in (entry or {}).get("shapes") or ()
            if isinstance(row, dict)
        ]
        if rows:
            shapes[role] = rows[:6]
    extent = analysis.get("static_function_extent") or {}
    return dict(
        loop_trip_counts=[row.get("value") for row in trips if isinstance(row, dict)][:16],
        loop_trip_counts_status="proved" if trips else "unknown",
        transfer_shapes=shapes or None,
        transfer_shapes_status=width.get("status", "unknown"),
        entry_symbol=analysis.get("symbol") or extent.get("symbol") or None,
        address_range=analysis.get("address_range"),
        instructions=analysis.get("listed_instructions"),
    )


def _inventory(value, root, bank, pin):
    offset, limit = value.get("offset", 0), value.get("limit", 8)
    units = sorted(bank["units"], key=lambda unit: unit["kernel_id"])
    path = value.get("path") or GRANT
    if path not in (GRANT, RAW):
        unit, _ = _select(dict(path=path), bank)
        units = [unit]
    rows = []
    for unit in units[offset : offset + limit]:
        raw, pin_card = _pinned(root, bank, unit["card"])
        card = json.loads(raw)
        if card.get("kernel_id") != unit["kernel_id"]:
            raise _Refused("cca_bank_record_inconsistent")
        occurrences = card.get("occurrences", [])
        fields, mechanisms, extents = set(), None, None
        for row in _occurrences(unit)[:1]:
            analysis = json.loads(_pinned(root, bank, row["files"]["analysis.json"])[0])
            from .cca_families import mechanism_summary

            # Summary first: roles A/W/output, width, buffering, overlap, prefetch, each observed or unknown.
            mechanisms = dict(region_id=row["region_id"], **mechanism_summary(analysis))
            extents = _extents(analysis)
            if extents.get("entry_symbol") is None:
                extents["entry_symbol"] = next((o["symbol"] for o in occurrences if o.get("symbol")), None)
            for item in (analysis.get("mechanism_projection") or {}).get("field_fingerprints", []):
                fields.add(item["instruction"] + "." + item["field"])
        # A compact, unranked summary of the card so a unit can be chosen without opening every file.
        summary = dict(
            originals=sorted({o.get("artifact_id") for o in occurrences if o.get("artifact_id")}),
            symbols=sorted({o["symbol"] for o in occurrences if o.get("symbol")}),
            accelerator_sites=[o.get("accelerator_sites") for o in occurrences],
            instruction_classes=sorted(
                {k for o in occurrences for k in (o.get("instruction_class_static_sites") or {})}
            ),
            observation_fields=sorted(fields),
        )
        rows.append(
            dict(
                unit_id=unit["kernel_id"],
                mechanisms=mechanisms,
                extents=extents,
                summary=summary,
                card=pin_card,
                occurrences=[
                    dict(
                        region_id=row["region_id"],
                        original_id=row["original_id"],
                        files={name: GRANT + "/" + relative for name, relative in row["files"].items()},
                    )
                    for row in _occurrences(unit)
                ],
                card_occurrences=occurrences,
            )
        )
    return dict(
        kind="cca_bank_inventory",
        bank=_bank_record(bank, pin),
        total=len(units),
        offset=offset,
        units=rows,
        next_offset=offset + len(rows) if offset + len(rows) < len(units) else None,
        ordering="unit_id",
        interface=dict(
            lift="--reference-analysis lift --reference-entry UNIT_ID [--reference-region REGION_ID]",
            compare="--reference-analysis compare --reference-entry UNIT_ID --reference-region REGION_ID "
            "--candidate-path OWNED_FILE [--candidate-symbol SYMBOL]",
            applicability="--reference-analysis applicability --reference-entry UNIT_ID --reference-region "
            "REGION_ID --candidate-path OWNED_FILE [--candidate-symbol SYMBOL] "
            "[--candidate-shape DIM=N,...]",
            files="every listed cca-evidence/ file is also readable directly",
        ),
        semantic_equivalence="unknown",
        performance="unknown",
    )


def _cca(root, bank, occurrence, analysis):
    from merlin.kernels.cca_mlir import from_mlir

    files = occurrence["files"]
    record = json.loads(_pinned(root, bank, files["cca.json"])[0])
    mlir = _pinned(root, bank, files["cca.mlir"])[0].decode("utf-8")
    try:
        cca = from_mlir(mlir)
    except Exception as error:  # A malformed retained record is a bank defect, never a lift trigger.
        raise _Refused("cca_bank_record_inconsistent") from error
    if json.loads(json.dumps(cca.to_dict())) != record or analysis.get("cca") != record:
        raise _Refused("cca_bank_record_inconsistent")
    return cca


def _lift(value, root, bank, pin, *, compare=False):
    unit, occurrences = _select(value, bank)
    offset, limit = (0, 1) if compare else (value.get("offset", 0), value.get("limit", 8))
    if compare and len(occurrences) != 1:
        raise ValueError("analysis_compare_requires_one_region")
    regions, ccas = [], []
    for occurrence in occurrences[offset : offset + limit]:
        files = occurrence["files"]
        raw, analysis_pin = _pinned(root, bank, files["analysis.json"])
        analysis = json.loads(raw)
        if analysis.get("stable_id", occurrence["region_id"]) != occurrence["region_id"]:
            raise _Refused("cca_bank_record_inconsistent")
        ccas.append(_cca(root, bank, occurrence, analysis))
        _, listing = _pinned(root, bank, files["instructions.dis"])
        regions.append(
            dict(
                analysis,
                bank_occurrence=dict(
                    unit_id=unit["kernel_id"],
                    region_id=occurrence["region_id"],
                    original_id=occurrence["original_id"],
                    analysis=analysis_pin,
                    instructions=listing,
                    files={name: GRANT + "/" + relative for name, relative in files.items()},
                ),
            )
        )
    _, card = _pinned(root, bank, unit["card"])
    report = dict(
        kind="static_instruction_analysis",
        source="precomputed_cca_bank",
        count_basis="listed_instruction_sites",
        total=len(occurrences),
        offset=offset,
        regions=regions,
        next_offset=offset + len(regions) if offset + len(regions) < len(occurrences) else None,
        bank=_bank_record(bank, pin),
        input=card,
        reference_entry=dict(id=unit["kernel_id"], kind="cca_bank_unit", semantic_compatibility="unknown"),
        selected_target_correspondence="unknown",
        next_action=(
            "Complete region records are in the detail file named in this reply; pass "
            "--reference-region REGION_ID (from inventory) to select one occurrence. "
            "Observation field names are listed per unit in inventory summaries."
        ),
    )
    return report, ccas[0] if len(ccas) == 1 else None


DECODER = ("decoder/target_contract.yaml", "decoder/facts.json", "decoder/operand-protocol.json")


def _bank_decoder(root, bank):
    """The bank's own pinned operand decoder, relocated to this bank copy; None for older banks."""
    import tempfile

    import yaml

    if not all(name in bank["files"] for name in DECODER):
        return None
    contract_raw, _ = _pinned(root, bank, DECODER[0])
    for name in DECODER[1:]:
        _pinned(root, bank, name)
    document = yaml.safe_load(contract_raw)
    selection = document.get("rocc_operand_selection") or {}
    pins = {"original_facts": DECODER[1], "sidecar": DECODER[2]}
    for key, relative in pins.items():
        entry = selection.get(key)
        if not isinstance(entry, dict) or entry.get("sha256") != bank["files"][relative]:
            raise _Refused("cca_bank_record_inconsistent")
        entry["path"] = str(root / relative)
    text = yaml.safe_dump(document, sort_keys=False)
    # One relocated contract per distinct content, reused across requests; never one file per request.
    digest = hashlib.sha256(text.encode()).hexdigest()[:24]
    contract = Path(tempfile.gettempdir()) / ("cca-bank-decoder-" + digest + ".yaml")
    if not contract.is_file() or contract.read_text() != text:
        partial = contract.with_name(f"{contract.name}.{os.getpid()}.partial")
        partial.write_text(text)
        os.replace(partial, contract)
    return dict(contract=contract, facts=root / DECODER[1])


def handle(
    value,
    *,
    base,
    workspace,
    bundle_manifest,
    repo,
    pin,
    facts=None,
    contract=None,
    target=None,
    descriptor=None,
    compiler_sha256=None,
    observation_fields=(),
):
    """Serve one validated reference-analysis request in bank mode."""
    from . import reference_analysis as A

    operation = value.get("operation")
    packet = dict(identity=document_digest(dict(schema="merlin.reference_bank_selection.v1", bank_sha256=pin)))
    analyze = dict(
        workspace=workspace,
        facts=facts,
        contract=contract,
        target=target,
        descriptor=descriptor,
        candidate_build_manifest=bundle_manifest,
    )
    if operation == "candidate":
        report, _ = A._analyze(value, reference_root=None, candidate=True, **analyze)
        report["input_role"] = "candidate_artifact"
        return A._response(base, packet, report, compiler_sha256, workspace=workspace)
    # Applicability takes the compare selection and refusals, then reports routes instead of a comparison.
    applicability = operation == "applicability"
    if applicability:
        from .reference_applicability import validate_shape

        validate_shape(value.get("shape"))
        operation = "compare"
    elif "shape" in value:
        raise ValueError("invalid_analysis_request")
    if operation == "compare" and not _reference(value):
        if applicability:
            raise _Refused(OUTSIDE)  # Applicability reads a bank unit's precomputed effects.
        # Owned before/after artifacts on both sides: the ordinary candidate owner, unchanged.
        left, ours_left = A._analyze(value, reference_root=None, candidate=True, compare=True, **analyze)
        right, ours = A._analyze(value["candidate"], reference_root=None, candidate=True, compare=True, **analyze)
        left["input_role"] = right["input_role"] = "candidate_artifact"
        report = A._compare(left, right, ours_left, ours, observation_fields=observation_fields)
        return A._response(base, packet, report, compiler_sha256, workspace=workspace)
    if not _reference(value):
        raise _Refused(OUTSIDE)
    root, bank = open_bank(workspace, bundle_manifest, repo, pin)
    decoder = _bank_decoder(root, bank)
    if decoder is not None:
        # The bank's records were decoded with the bank's pinned decoder; the candidate side of a
        # reference comparison uses the same one, so observations share one decoder identity.
        analyze.update(facts=decoder["facts"], contract=decoder["contract"], descriptor=None)
    if operation == "inventory":
        return A._response(base, packet, _inventory(value, root, bank, pin), compiler_sha256, workspace=workspace)
    left, expert = _lift(value, root, bank, pin, compare=operation == "compare")
    left["input_role"] = "selected_reference"
    context = {key: analyze[key] for key in ("facts", "contract", "target", "descriptor")}
    if operation == "compare" and "candidate" not in value:  # a top-down schedule alone: comparison (a)
        from . import reference_applicability as R

        return R.respond_schedule(
            left,
            value,
            base=base,
            packet=packet,
            workspace=workspace,
            decoding_context=context,
            compiler_sha256=compiler_sha256,
        )
    if operation == "compare":
        right, ours = A._analyze(value["candidate"], reference_root=None, candidate=True, compare=True, **analyze)
        right["input_role"] = "candidate_artifact"
        if applicability:
            from . import reference_applicability as R

            return R.respond(
                left,
                right,
                value,
                base=base,
                packet=packet,
                workspace=workspace,
                facts=facts,
                contract=contract,
                target=target,
                descriptor=descriptor,
                decoding_context={key: analyze[key] for key in ("facts", "contract", "target", "descriptor")},
                compiler_sha256=compiler_sha256,
            )
        region = left["regions"][0]
        left = A._compare(left, right, expert, ours, observation_fields=observation_fields)
        if value.get("candidate_schedule"):
            from .reference_applicability import _candidate_effects
            from .schedule_attribution import attribute_schedule

            emitted, _, _ = _candidate_effects(value["candidate"], right["regions"][0], workspace=workspace, **context)
            brief, full = attribute_schedule(
                workspace, value["candidate_schedule"], reference=region, emitted=emitted, decoding_context=context
            )
            left = dict(schedule_attribution=brief, **left, schedule_attribution_detail=full)
    return A._response(base, packet, left, compiler_sha256, workspace=workspace)
