"""Build one bounded selfcheck family record from explicitly selected bank and lifting callbacks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from merlin.development.backend.feedback.cca_bank import (  # noqa: F401 -- preserve established imports at this boundary
    _KERNEL,
    Skip,
    _read_json,
)
from merlin.development.backend.feedback.cca_comparison import (  # noqa: F401 -- preserve established imports at this boundary
    FACT_KINDS,
    SCHEMA,
    build,
    mechanism_summary,
)
from merlin.development.backend.feedback.cca_report import (  # noqa: F401 -- preserve established imports at this boundary
    SELFCHECK_SCHEMA,
    _jsonable,
)

_MAX_ELF = 64 << 20


def selfcheck_kernels(selfcheck_dir, cases=None):
    """``({case: linked kernel ELF}, {case: reason})`` of one self-check's raw runs.

    The self-check grades each case in ``<runs root>/runs/<suite>/<case>/`` (``capsule_result.json``
    beside ``generated/package_kernel.elf``, as :mod:`.selfcheck` reads them); the runs root, its
    ``runs/`` directory or the suite directory is accepted. ``cases`` limits the answer to those cases."""
    root, results = Path(selfcheck_dir), []
    for directory, pattern in (
        (root, "*/capsule_result.json"),
        (root, "*/*/capsule_result.json"),
        (root / "runs", "*/*/capsule_result.json"),
    ):
        results = sorted(directory.glob(pattern))
        if results:
            break
    found = {}
    for result in results:
        found.setdefault(result.parent.name, result.parent / "generated" / _KERNEL)
    kernels, missing = {}, {}
    for case in sorted(found if cases is None else set(cases)):
        path = found.get(case)
        if path is None:
            missing[case] = "no result for this case in the self-check output"
        elif not path.is_file():
            missing[case] = "the self-check retained no linked kernel ELF for this case"
        else:
            kernels[case] = path
    return kernels, missing


def _digest(path):
    if Path(path).stat().st_size > _MAX_ELF:
        raise Skip(f"{Path(path).name} exceeds {_MAX_ELF} bytes")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(*parts):
    return hashlib.sha256(json.dumps(_jsonable(parts), sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str))
    os.replace(temporary, path)


def _facts_document(facts):
    if facts is None or isinstance(facts, dict):
        return facts
    return json.loads(Path(facts).read_text())


def _family_view(family, units):
    """One built family plus the nearest unit's own mechanisms and the representative compared with it."""
    match = family.get("nearest_unit")
    unit = next(
        (u for u in units if match and (u["unit_id"], u["region_id"]) == (match["unit_id"], match["region_id"])), None
    )
    reps = [rep for rep in family.get("representatives") or () if rep.get("status") == "compared"]
    return dict(
        family,
        nearest_mechanisms=mechanism_summary(unit["analysis"]) if unit else None,
        compared=reps[0] if reps else None,
    )


def _selfcheck_record(
    workspace,
    selfcheck_dir,
    *,
    bank_root,
    facts,
    target,
    pin,
    cases,
    budget_s,
    cache_dir,
    started,
    deadline,
    clock,
    open_selected_bank,
    _decoder_for,
    _units_for,
    case_semantics,
    _lifted,
):
    if target is None:
        try:
            from merlin.development.backend.feedback import case_declarations

            target = case_declarations.target()
        except Exception:
            target = None
    if not target:
        raise Skip("no selected target registered in this process")
    kernels, missing = selfcheck_kernels(selfcheck_dir, cases)
    if not kernels:
        raise Skip(
            "no linked kernel ELF in this self-check's output"
            + "".join((f"; `{case}`: {why}" for case, why in sorted(missing.items())[:3]))
        )
    digests = {case: _digest(path) for case, path in kernels.items()}
    from merlin.development.backend.feedback.reference_bank import GRANT

    root = Path(bank_root) if bank_root is not None else Path(workspace) / GRANT
    pin_basis = "supplied by the arm"
    if pin is None:
        try:
            pin = hashlib.sha256((Path(os.path.realpath(root)) / "bank.json").read_bytes()).hexdigest()
        except OSError as error:
            raise Skip(f"no bank.json under {root}") from error
        pin_basis = "digest of bank.json (the arm supplied no pin)"
    facts_doc = _facts_document(facts)
    operations, shapes = case_semantics(workspace, sorted(kernels))
    key = _key(SELFCHECK_SCHEMA, SCHEMA, pin, target, digests, operations, shapes, FACT_KINDS, facts_doc)
    cached = _read_json(Path(cache_dir) / f"record-{key[:32]}.json")
    if isinstance(cached, dict) and cached.get("key") == key and (cached.get("status") == "compared"):
        return dict(cached, cache="hit", not_lifted=dict(cached.get("not_lifted") or {}, **missing))
    if clock() > deadline:
        raise Skip(f"budget of {budget_s}s exhausted before the bank was opened")
    root, bank, decoder_files = open_selected_bank(workspace, pin, root=bank_root)
    decoder, structure = _decoder_for(decoder_files, target)
    units = _units_for(root, bank, pin)
    members, failed, lifts = ([], {}, {})
    for case, path in sorted(kernels.items()):
        if clock() > deadline:
            failed[case] = "budget exhausted before lifting"
            continue
        try:
            regions, hit = _lifted(
                path,
                digests[case],
                pin=pin,
                target=target,
                decoder=decoder,
                decoder_files=decoder_files,
                structure=structure,
                cache_dir=cache_dir,
            )
        except Exception as error:
            failed[case] = f"{type(error).__name__}: {str(error)[:120]}"
            continue
        lifts[case] = "cached" if hit else "lifted"
        if not regions:
            failed[case] = "no accelerator region in the linked kernel"
        members.extend((dict(region, case=case) for region in regions))
    from merlin.kernels.mechanism_applicability import selected_geometry

    geometry, reason = selected_geometry(
        target, facts_doc if facts_doc is not None else json.loads(Path(decoder_files["facts"]).read_text())
    )
    full = build(
        members,
        units,
        operations=operations,
        shapes=shapes,
        geometry=geometry,
        geometry_reason=reason,
        structure=structure,
        deadline=deadline,
        clock=clock,
    )
    labels = {family["key"]: f"F{index}" for index, family in enumerate(full["families"], 1)}
    per_case = {
        case: dict(
            elf=str(path),
            elf_sha256=digests[case],
            lift=lifts.get(case),
            kernels=[
                dict(region_id=m["region_id"], identity=m["identity"], family=labels.get(m["family"]))
                for m in sorted(members, key=lambda m: m["region_id"])
                if m["case"] == case
            ],
        )
        for case, path in sorted(kernels.items())
    }
    budget_cut = [case for case, why in failed.items() if why.startswith("budget")]
    complete = (
        not budget_cut
        and (not full["skipped_families"])
        and (not any((family.get("skipped_members") for family in full["families"])))
    )
    record = dict(
        schema=SELFCHECK_SCHEMA,
        key=key,
        status="compared" if complete else "partial",
        target=target,
        budget_s=budget_s,
        cases=per_case,
        not_lifted=dict(failed, **missing),
        bank=dict(sha256=pin, pin_basis=pin_basis, root=str(root), units=len({u["unit_id"] for u in units})),
        families=[dict(_family_view(family, units), label=labels[family["key"]]) for family in full["families"]],
        fact_kinds=full["fact_kinds"],
        ordering=full["ordering"],
        semantic_equivalence="unknown",
        performance="unknown",
        cache="miss",
    )
    record = _jsonable(record)
    if complete:
        try:
            _write_json(Path(cache_dir) / f"record-{key[:32]}.json", record)
        except OSError as error:
            logging.getLogger(__name__).warning("CCA selfcheck cache not written: %s", type(error).__name__)
    return record


# Preserve the established callable pickle identities through explicit facade exports.
for _export in (
    selfcheck_kernels,
    _digest,
    _key,
    _write_json,
    _facts_document,
    _family_view,
):
    _export.__module__ = "merlin.development.backend.feedback.cca_families"
