"""CCA families in the grade feedback of a CCA bank arm: your kernels beside the nearest expert unit.

Measured on the CCA coverage arms: authors made about 10,000 tool calls and asked the bank service
twice, both ``inventory``. The service was one paragraph near the end of TASK.md and absent from the
feedback the authors read. This module brings its comparison into ``qa/round_brief.md``.

After a published grade, the emitted kernels of the grade (each case's linked ``package_kernel.elf``)
are lifted with the bank's own pinned decoder, through the same lifter and the same effect derivation
that built the bank, and grouped three ways, all derived by tooling:

* **identical**: the SHA256 of the linked accelerator-region bytes (the bank's own unit identity);
* **family**: :func:`merlin.mining.reference_subset_cards.family_signature` (decoded roles,
  instruction classes, configured-field set, loop-nest shape, control status, non-numeric dataflow
  facts), the same function over bank units and over authors' kernels; counts and shapes excluded;
* **operation class**: the case's declared source semantics (:func:`.family_rollup.describe`), used
  only to shortlist bank units when units carry one; bank units carry none today, which is stated.

Per family: the nearest bank unit by signature agreement, with the fields that agree and differ;
representatives (the most frequent body, and the slowest member by the latest modeled cycles when any
member has them); the applicability comparison of each representative with that unit
(:func:`merlin.kernels.mechanism_applicability.applicability`); a member-by-member comparison with the
representative (:func:`~merlin.kernels.mechanism_applicability.mechanism_differences`) whose deviating
members are listed as exceptions; and, per member, the shape-dependent route admissions.

Facts only: nothing is ranked by assumed speed, structural agreement is never semantic equivalence, and
performance stays unknown. An arm without a round brief gets the same comparison for the cases of one
self-check from :func:`selfcheck_section` (bounded text, full record in a named detail file, cached by
ELF digest). Enabled only for a CCA bank profile. It runs off the grading path in one
background thread with a wall-clock budget; work left at the budget is skipped with that reason. It
never raises into a grade. No target name, regex or opcode literal."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from . import cca_selfcheck as _cca_selfcheck
from .cca_bank import (  # noqa: F401 -- preserve established imports at this boundary
    _KERNEL,
    _MAX_JSON,
    Skip,
    _card_symbols,
    _read_json,
    bank_units,
    case_semantics,
    grade_kernels,
    latest_cycles,
    lift_kernel,
    open_selected_bank,
    selected_decoder,
)
from .cca_comparison import (  # noqa: F401 -- preserve established imports at this boundary
    _SUMMARY_KEYS,
    EXPERT_UNITS,
    FACT_KINDS,
    SCHEMA,
    SUMMARY_ROLES,
    _admissions,
    _directions,
    _kind,
    _per_role,
    _rows,
    _summary_text,
    _unknown_groups,
    _value,
    build,
    group,
    mechanism_summary,
    nearest,
    nearest_units,
)
from .cca_report import (  # noqa: F401 -- preserve established imports at this boundary
    _VALUE_CHARS,
    RECORD,
    SELFCHECK_DIFFERENCES,
    SELFCHECK_MECHANISMS,
    SELFCHECK_MEMBERS,
    SELFCHECK_SCHEMA,
    SELFCHECK_TEXT_BYTES,
    SELFCHECK_UNKNOWN_GROUPS,
    SHOWN_CASES,
    SHOWN_FAMILIES,
    SHOWN_ROWS,
    _compact,
    _diff_line,
    _family_lines,
    _flat,
    _jsonable,
    _mechanism,
    _short,
    _summary_record,
    _unknown_groups_of,
    render,
    selfcheck_text,
)
from .cca_selfcheck import (  # noqa: F401 -- preserve established imports at this boundary
    _MAX_ELF,
    _digest,
    _facts_document,
    _family_view,
    _key,
    _write_json,
    selfcheck_kernels,
)

BUDGET_S = 300.0


_LOCK = threading.Lock()


def brief_lines(workspace):
    """The cached section; ``[]`` outside a bank arm or before the first record."""
    if _bank_pin() is None:
        return []
    return render(_read_json(Path(workspace) / "qa" / RECORD))


def _bank_pin():
    try:
        from .reference_bank import selected_pin

        return selected_pin()
    except Exception:  # noqa: BLE001 -- no profile, no section
        return None


def _write(workspace, record):
    path = Path(workspace) / "qa" / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{RECORD}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str))
    os.replace(temporary, path)


def compute(run_dir, workspace, *, pin, target, budget_s=BUDGET_S, clock=time.monotonic):
    """One bounded record for the newest completed grade; a stated skip instead of an exception."""
    started = clock()
    deadline = started + budget_s
    grade = _read_json(Path(workspace) / "qa" / "latest_completed_grade.json") or {}
    token = (grade.get("performance_headline") or {}).get("grade", {}).get("token")
    base = dict(schema=SCHEMA, grade_token=token, budget_s=budget_s)
    try:
        if not target:
            raise Skip("no selected target registered in this process")
        root, bank, decoder_files = open_selected_bank(workspace, pin)
        decoder, structure = selected_decoder(decoder_files, target)
        units = bank_units(root, bank)
        kernels = grade_kernels(run_dir, workspace)
        if not kernels:
            raise Skip("the newest completed grade retained no emitted kernel ELFs")
        members, failed = [], {}
        for case, path in sorted(kernels.items()):
            if clock() > deadline:
                failed[case] = "budget exhausted before lifting"
                continue
            try:
                for region in lift_kernel(path, decoder=decoder, facts=decoder_files["facts"], structure=structure):
                    members.append(dict(region, case=case))
            except Exception as error:  # noqa: BLE001 -- one unliftable kernel is reported, not fatal
                failed[case] = f"{type(error).__name__}: {str(error)[:120]}"
        operations, shapes = case_semantics(workspace, sorted({m["case"] for m in members}))
        from merlin.kernels.mechanism_applicability import selected_geometry

        geometry, reason = selected_geometry(target, json.loads(Path(decoder_files["facts"]).read_text()))
        record = build(
            members,
            units,
            cycles=latest_cycles(workspace),
            operations=operations,
            shapes=shapes,
            geometry=geometry,
            geometry_reason=reason,
            structure=structure,
            deadline=deadline,
            clock=clock,
        )
        record.update(
            base,
            bank_sha256=pin,
            bank_units=len({u["unit_id"] for u in units}),
            not_lifted=failed,
            seconds=round(clock() - started, 1),
        )
        return _jsonable(record)
    except Skip as reason:
        return dict(base, status="skipped", reason=str(reason))
    except Exception as error:  # noqa: BLE001 -- feedback never fails a grade
        return dict(base, status="skipped", reason=f"{type(error).__name__}: {str(error)[:200]}")


def refresh_async(run_dir, workspace, *, target=None, on_done=None):
    """Start one background computation after a grade; returns the thread, or None when not applicable.

    At most one runs at a time; a grade published meanwhile is picked up by the next refresh."""
    pin = _bank_pin()
    if pin is None or not _LOCK.acquire(blocking=False):
        return None
    if target is None:
        try:
            from . import case_declarations

            target = case_declarations.target()
        except Exception:  # noqa: BLE001 -- compute() states the missing target
            target = None

    def run():
        try:
            _write(workspace, compute(run_dir, workspace, pin=pin, target=target))
            if on_done is not None:
                on_done()
        except Exception as error:  # noqa: BLE001
            print(f"[grade] cca family feedback skipped: {type(error).__name__}: {error}", flush=True)
        finally:
            _LOCK.release()

    thread = threading.Thread(target=run, name="cca-families", daemon=True)
    thread.start()
    return thread


SELFCHECK_BUDGET_S = 60.0


_DECODERS: dict = {}


_UNITS: dict = {}


def arm_enabled(arm):
    """``(enabled, bank pin or None)`` of an explicit arm flag (bool), arm mapping or author profile."""
    if isinstance(arm, bool) or arm is None:
        return bool(arm), None
    if isinstance(arm, dict):
        pin = arm.get("bank_sha256") or arm.get("cca_bank_sha256")
        return bool(arm.get("cca", pin is not None)), pin
    pin = getattr(arm, "bank_sha256", None)
    return pin is not None, pin


def _decoder_for(decoder_files, target):
    key = (str(decoder_files["contract"]), str(decoder_files["facts"]), target)
    if key not in _DECODERS:
        _DECODERS.clear()
        _DECODERS[key] = selected_decoder(decoder_files, target)
    return _DECODERS[key]


def _units_for(root, bank, pin):
    key = (str(root), pin)
    if key not in _UNITS:
        _UNITS.clear()
        _UNITS[key] = bank_units(root, bank)
    return _UNITS[key]


def _lifted(path, digest, *, pin, target, decoder, decoder_files, structure, cache_dir):
    """Lifted regions of one ELF, cached by its digest; the JSON form both ways, so cold equals cached."""
    cache = Path(cache_dir) / f"lift-{_key(SCHEMA, pin, target, digest)[:32]}.json"
    cached = _read_json(cache)
    if isinstance(cached, dict) and cached.get("elf_sha256") == digest and isinstance(cached.get("regions"), list):
        return cached["regions"], True
    regions = json.loads(
        json.dumps(
            _jsonable(lift_kernel(path, decoder=decoder, facts=decoder_files["facts"], structure=structure)),
            default=str,
        )
    )
    try:
        _write_json(cache, dict(elf_sha256=digest, regions=regions))
    except OSError as error:
        logging.getLogger(__name__).warning("CCA lift cache not written: %s", type(error).__name__)
    return regions, False


def selfcheck_section(
    workspace,
    selfcheck_dir,
    *,
    arm,
    bank_root=None,
    facts=None,
    target=None,
    pin=None,
    cases=None,
    budget_s=SELFCHECK_BUDGET_S,
    detail_dir=None,
    cache_dir=None,
    limit=SELFCHECK_TEXT_BYTES,
    clock=time.monotonic,
):
    """CCA family feedback for the cases one self-check built; ``None`` when the arm has no CCA bank.

    ``arm`` is the explicit enablement: ``True``/``False``, a mapping ``{"cca": bool, "bank_sha256": pin}``
    or an author profile (enabled when it pins a bank). ``selfcheck_dir`` is the self-check's raw runs root
    (see :func:`selfcheck_kernels`), read before the check releases its scratch; ``cases`` the requested
    dev cases. ``bank_root`` defaults to the workspace's ``cca-evidence/``; ``pin`` to the arm's pin, else
    the digest of its ``bank.json`` (stated in the record). ``facts`` (path or document) selects the RTL
    geometry for route admission, default the bank's pinned decoder facts; ``target`` defaults to the
    registered selected target.

    Returns ``{"text", "record", "detail_path"}``: ``text`` at most ``limit`` bytes, ``record`` its compact
    JSON, ``detail_path`` the full record (``selfcheck_out/details/`` by default; ``None`` when skipped).
    Bounded by ``budget_s``: work past it is skipped with that reason. Records and lifts are cached by ELF
    digest in ``cache_dir`` (default ``qa/.cca_selfcheck_cache/``). Never raises."""
    enabled, arm_pin = arm_enabled(arm)
    if not enabled:
        return None
    started = clock()
    workspace = Path(workspace)
    detail_dir = Path(detail_dir) if detail_dir is not None else workspace / "selfcheck_out" / "details"
    cache_dir = Path(cache_dir) if cache_dir is not None else workspace / "qa" / ".cca_selfcheck_cache"
    try:
        record = _selfcheck_record(
            workspace,
            selfcheck_dir,
            bank_root=bank_root,
            facts=facts,
            target=target,
            pin=pin or arm_pin,
            cases=cases,
            budget_s=budget_s,
            cache_dir=cache_dir,
            started=started,
            deadline=started + budget_s,
            clock=clock,
        )
    except Skip as reason:
        record = dict(schema=SELFCHECK_SCHEMA, status="skipped", reason=str(reason), budget_s=budget_s)
    except Exception as error:  # noqa: BLE001 -- feedback never fails a self-check
        record = dict(
            schema=SELFCHECK_SCHEMA,
            status="skipped",
            budget_s=budget_s,
            reason=f"{type(error).__name__}: {str(error)[:200]}",
        )
    record["seconds"] = round(clock() - started, 3)
    detail_path, detail = None, None
    if record.get("status") != "skipped":
        detail_path = detail_dir / f"cca_families_{record['key'][:16]}.json"
        try:
            _write_json(detail_path, record)
            detail = (
                str(detail_path.relative_to(workspace)) if detail_path.is_relative_to(workspace) else str(detail_path)
            )
        except OSError as error:
            detail_path, detail = None, f"not written ({type(error).__name__})"
    return dict(
        text=selfcheck_text(record, detail, limit=limit),
        record=_summary_record(record, detail),
        detail_path=None if detail_path is None else str(detail_path),
    )


def _selfcheck_record(
    workspace, selfcheck_dir, *, bank_root, facts, target, pin, cases, budget_s, cache_dir, started, deadline, clock
):
    """Retain the existing live selected-source callback boundary."""
    return _cca_selfcheck._selfcheck_record(
        workspace,
        selfcheck_dir,
        bank_root=bank_root,
        facts=facts,
        target=target,
        pin=pin,
        cases=cases,
        budget_s=budget_s,
        cache_dir=cache_dir,
        started=started,
        deadline=deadline,
        clock=clock,
        open_selected_bank=lambda *args, **kwargs: open_selected_bank(*args, **kwargs),
        _decoder_for=lambda *args, **kwargs: _decoder_for(*args, **kwargs),
        _units_for=lambda *args, **kwargs: _units_for(*args, **kwargs),
        case_semantics=lambda *args, **kwargs: case_semantics(*args, **kwargs),
        _lifted=lambda *args, **kwargs: _lifted(*args, **kwargs),
    )
