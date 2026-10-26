"""The completion-order check as the capsule ladder runs it: L2 structural evidence.

A numerical pass on the functional model (the L2 tier) says the program computes the declared
operation when every accelerator command completes in program order. :mod:`merlin.targetgen.
completion_order` asks the question that pass cannot: does the program depend on that order where
the hardware does not provide it? It reads the command stream the functional run actually executed
and the target's RTL-derived ordering facts, so it decides at L2, before any RTL engine runs, and a
flagged pair fails the capsule there with the commands and rows that race.

* ``hazard``: the functional tier is failed with :func:`merlin.targetgen.completion_order.reason`.
* ``clear``: recorded; also tells the opt-in memory-perturbation sweep it has nothing to look for.
* ``unknown``: recorded with every reason (facts not derived, an opaque command, an unmodeled
  configuration); never a pass by omission, never a failure of a program the facts cannot judge.

The verdict is a property of the executable's bytes and the facts, so it is cached under both
digests and a re-grade of unchanged bytes does not re-run the commit-log trace.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Set to ``off`` to skip the check; the result then records that it was skipped and why.
SWITCH_ENV = "MERLIN_COMPLETION_ORDER_CHECK"
TIMEOUT_ENV = "MERLIN_COMPLETION_ORDER_TIMEOUT_S"
_CACHE_NAMESPACE = "completion_order"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checker_digest() -> str:
    from merlin.perf import executed_trace, streamed_executed_trace
    from merlin.targetgen import completion_order, rtl_checks_generic, semantic_families
    from merlin.targetgen.rocc import decode, semantics

    readers = (
        completion_order,
        executed_trace,
        streamed_executed_trace,
        rtl_checks_generic,
        semantic_families,
        semantics,
        decode,
    )
    pins = {m.__name__: _sha(Path(m.__file__)) for m in readers}
    pins[__name__] = _sha(Path(__file__))
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def _observation_identity(target: str, elf: Path) -> dict[str, Any] | None:
    """Reopen the actual decoder and selected functional engine before reusing a trace verdict.

    No resolvable engine identity means no persistent cache, rather than an identity by target name.
    The normal invocation explicitly names its accelerator receiver via --extlib.
    """
    import shutil

    from merlin.perf.executed_trace import trace_invocation
    from merlin.targetgen.completion_order import trace_isa

    invocation = trace_invocation(target, elf)
    if invocation is None:
        return None
    argv, env = invocation
    if not argv:
        return None
    engine = shutil.which(str(argv[0]), path=env.get("PATH"))
    if engine is None:
        return None
    files = {str(Path(engine).resolve())}
    for i, arg in enumerate(argv[1:], 1):
        if arg.startswith("--extlib="):
            path = Path(arg.partition("=")[2])
            if not path.is_file():
                return None
            files.add(str(path.resolve()))
        elif arg == "--extlib":
            if i + 1 >= len(argv) or not Path(argv[i + 1]).is_file():
                return None
            files.add(str(Path(argv[i + 1]).resolve()))
        elif Path(arg).is_file() and Path(arg).resolve() != elf.resolve():
            files.add(str(Path(arg).resolve()))
    return {
        "argv": [str(a) for a in argv],
        "environment": dict(env),
        "files": {p: _sha(Path(p)) for p in sorted(files)},
        "isa": trace_isa(target),
    }


def check(target: str, generated: Path | str) -> dict[str, Any]:
    """The completion-order record for the executable built in ``generated`` (never raises)."""
    from merlin.targetgen import completion_order as CO

    from .elf_lanes import PACKAGE_ELF_NAME

    if os.environ.get(SWITCH_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return {"schema": CO.SCHEMA, "verdict": "skipped", "detail": f"{SWITCH_ENV} switched the check off"}
    elf = Path(generated) / PACKAGE_ELF_NAME
    if not elf.is_file():
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"no executable at {elf}"}
    try:
        from merlin.targetgen.rtl.facts import load_facts

        document = load_facts(target, regenerate=False)
        body = document.get("facts") or {}
    except Exception as exc:  # noqa: BLE001 - no facts: recorded as the reason, never a pass
        return {
            "schema": CO.SCHEMA,
            "verdict": "unknown",
            "detail": f"no RTL facts for {target!r}: {type(exc).__name__}: {exc}",
        }
    from merlin.targetgen.rtl import command_ordering_selection as selection_owner

    selection = None
    if selected := os.environ.get(selection_owner.ENV):
        try:
            ordering, selection = selection_owner.read(json.loads(selected), target=target, document=document)
            body = {**body, "command_ordering": ordering}
        except Exception as exc:  # noqa: BLE001 - an invalid explicit selection never falls back
            return {
                "schema": CO.SCHEMA,
                "verdict": "unknown",
                "detail": f"ordering selection unavailable: {type(exc).__name__}: {exc}",
            }
    ordering = body.get("command_ordering")
    if not ordering:
        return {
            "schema": CO.SCHEMA,
            "verdict": "unknown",
            "detail": (
                "the RTL facts carry no command_ordering section (merlin.targetgen.rtl.command_ordering derives it)"
            ),
        }
    try:
        model = CO.OrderingModel.for_target(target, body=body)
    except CO.Unavailable as exc:
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": str(exc)}
    try:
        observed = _observation_identity(target, elf)
        identity = {
            "target": target,
            "elf": _sha(elf),
            "facts": document,
            "ordering_selection": selection,
            "row_masks": model.row_masks,
            "select_flag": model.select_flag,
            "sentinel": model.sentinel,
            "readers": _checker_digest(),
            "observation": observed,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        from merlin.common.artifacts import cache_dir

        cached = cache_dir(_CACHE_NAMESPACE) / f"{key}.json"
        if observed is not None and cached.is_file():
            record = json.loads(cached.read_text(encoding="utf-8"))
            record["cache"] = "carried"
            return record
        timeout = float(os.environ.get(TIMEOUT_ENV) or 1200)
        record = CO.check_elf(target, elf, model=model, timeout_s=timeout)
        if selection is not None:
            record["ordering_selection"] = selection
        if "detail" not in record and observed is not None and observed == _observation_identity(target, elf):
            # Only a complete observation with unchanged selected engine/decoder identity is reusable.
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            tmp.replace(cached)
        record["cache"] = "measured"
        return record
    except Exception as exc:  # noqa: BLE001 - a crashed check is recorded, never read as a pass
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"check crashed: {type(exc).__name__}: {exc}"}


def failure_reason(record: dict[str, Any]) -> str | None:
    """The author-facing reason when ``record`` fails the capsule, else None."""
    if record.get("verdict") != "hazard":
        return None
    from merlin.targetgen import completion_order as CO

    return CO.reason(record)


def functional_tier(tiers: dict, cfg) -> str | None:
    """The mandatory functional (non-RTL) tier that passed: the L2 evidence this check belongs to."""
    rtl = set(getattr(cfg, "rtl_tiers", ()) or ())
    passed = [
        t
        for t in sorted(tiers)
        if t not in rtl and getattr(tiers[t], "status", None) == "pass" and getattr(tiers[t], "mandatory", False)
    ]
    return passed[0] if passed else None


def after_functional_pass(target: str, tier: str, tiers: dict, generated) -> tuple[dict[str, Any], str | None]:
    """Run the check for ``tier`` (a passed functional tier); returns ``(record, failure reason)``.

    On a hazard the tier's record is rewritten to ``fail`` with the reason, in place."""
    import dataclasses

    record = check(target, generated)
    record["tier"] = tier
    reason = failure_reason(record)
    if reason is not None and tier in tiers:
        tiers[tier] = dataclasses.replace(tiers[tier], status="fail", reason=reason)
    return record, reason


def ladder_step(target: str, tier: str | None, tiers: dict, generated, cfg=None):
    """``(record, CertFailure | None)`` for the ladder. ``tier`` None: find the passed functional tier
    (one carried from an earlier grade); ``(None, None)`` when there is none to qualify."""
    from merlin.targetgen.capsule_common import _cat
    from merlin.targetgen.package_runtime import CertFailure

    if tier is None:
        tier = functional_tier(tiers, cfg)
        if tier is None:
            return None, None
    record, reason = after_functional_pass(target, tier, tiers, generated)
    failure = None if reason is None else CertFailure("completion_order", _cat("PROTOCOL_VIOLATION"), reason)
    return record, failure


def suppress_later(tiers: dict, sequence, failed: str, required, rtl_tiers, factory) -> None:
    """Record every tier after ``failed`` the ladder will not run as skipped (never absent)."""
    for later in list(sequence)[list(sequence).index(failed) + 1 :]:
        if later not in tiers:
            tiers[later] = factory(later, later in required, failed, from_rtl=later in set(rtl_tiers or ()))


def sweep_trigger(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The completion-order verdict as the memory-perturbation sweep's static trigger, or None."""
    if not record or record.get("verdict") not in ("clear", "hazard", "unknown"):
        return None
    verdict = {"clear": "clear", "hazard": "hazard", "unknown": "unresolved"}[record["verdict"]]
    return {
        "verdict": verdict,
        "source": "completion_order",
        "hazard_pairs": record.get("hazard_pairs"),
        "uncertain": record.get("uncertain") or ({"detail": record.get("detail")} if record.get("detail") else {}),
    }
