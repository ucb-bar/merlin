"""Carry a seed run's tier verdicts into a seeded run that grades with another framework export.

:mod:`merlin.targetgen.tier_cache` keys a verdict on the program (execution identity) and on the
judge (instrument digest). A run started without ``MERLIN_TIER_CERT_PORTABLE=1`` names its judge's
files by absolute path inside its own frozen export, so its verdicts can never match a run that
executes from another export, even when every grading byte is the same.

This re-keys them, and only when that is provably sound:

1. The seeded run's judge files are listed in THIS runtime (the new export), exactly as the grader
   lists them (``tier_cache.grading_path``), and each is located at the same relative name in the seed
   run's export (a file outside the checkout keeps its name).
2. The seed run's stored instrument must be REPRODUCED from those old bytes (absolute or portable
   naming, the stored record's own engine). A record whose instrument cannot be reproduced was earned
   under a judge nobody can name now and is left to re-run.
3. Every judge file must decide verdicts the same way in both exports. A byte-identical file does; a
   Python file whose code is identical apart from comments and docstrings does (compared as syntax
   trees); a file the operator reviewed as equivalent does, but only for the exact old and new bytes
   the review names (``reviewed``). Anything else is a changed judge: nothing is re-keyed, those tiers
   re-run, and the report lists every changed file with both digests so it can be reviewed.

A re-keyed record keeps its execution identity, verdict and cycles, gets the new export's portable
instrument, and says where it came from (``source: carried from <run>``; ``carried_from``). Only
``pass`` records are carried, as in the cache itself. The seeded run must set
``MERLIN_TIER_CERT_PORTABLE=1`` (the reviewed joint seed requires it) for a lookup to find them,
and ``MERLIN_SEEDED_GRADE=<seed run>`` for its grades to carry the screen tier too and to order and
label their work by what the seed proved (:mod:`merlin.targetgen.seeded_grade`, which reads the
index of carried test cases written here).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "merlin.seed_certificate_carry.v1"
BASIS_ABSOLUTE = "absolute"
BASIS_PORTABLE = "checkout_relative"


def _relative(path: Path, root: Path) -> Path | None:
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base)
        except ValueError:
            continue
    return None


def relocate(files, new_root: Path, old_root: Path) -> list[tuple[Path, Path]]:
    """``(new file, the same name in the old export)`` for every judge file."""
    pairs = []
    for item in files:
        path = Path(item)
        relative = _relative(path, Path(new_root))
        pairs.append((path, Path(old_root) / relative if relative is not None else path))
    return pairs


def _code(data: bytes) -> str | None:
    """A Python file's syntax tree without docstrings, or ``None`` when it is not Python."""
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return ast.dump(tree, include_attributes=False)


def compare_judges(pairs, new_root: Path, reviewed=None) -> dict:
    """Whether the old export's judge decides verdicts as the new one does, naming each difference."""
    reviewed = dict(reviewed or {})
    changed, missing, equivalent, approved = [], [], [], []
    for new, old in pairs:
        relative = _relative(new, Path(new_root))
        label = relative.as_posix() if relative is not None else str(new)
        try:
            after, before = new.read_bytes(), old.read_bytes()
        except OSError:
            missing.append(label)
            continue
        if after == before:
            continue
        digests = {"old_sha256": hashlib.sha256(before).hexdigest(), "new_sha256": hashlib.sha256(after).hexdigest()}
        review = reviewed.get(label)
        if new.suffix == ".py" and _code(after) is not None and _code(after) == _code(before):
            equivalent.append(label)
        elif isinstance(review, dict) and all(review.get(k) == v for k, v in digests.items()):
            approved.append({"file": label, "reason": review.get("reason")})
        else:
            changed.append({"file": label, **digests})
    return {
        "identical": not changed and not missing,
        "changed": sorted(changed, key=lambda row: row["file"]),
        "missing": sorted(missing),
        "equivalent_code": sorted(equivalent),
        "reviewed_equivalent": sorted(approved, key=lambda row: row["file"]),
    }


def portable_digest(files, root: Path) -> str:
    """``tier_cache._portable_source_digest`` for a checkout other than this implementation's own.

    Kept here rather than parameterized there: ``tier_cache`` is itself a judge file, and editing it
    would make every export after the edit a different judge from every run before it.
    """
    import hashlib

    root = Path(root).resolve()
    labelled = []
    for path in files:
        resolved = Path(path).resolve()
        label = str(resolved)
        if resolved.is_relative_to(root):
            label = "checkout:" + resolved.relative_to(root).as_posix()
        labelled.append((label, resolved))
    digest = hashlib.sha256()
    for label, resolved in sorted(labelled):
        digest.update(label.encode("utf-8"))
        try:
            digest.update(resolved.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _instrument(engine: str, tier: str, grading_path: str, basis: str) -> str:
    from merlin.targetgen import tier_cache

    payload = {"version": tier_cache.RECORD_VERSION, "engine": engine, "tier": str(tier), "grading_path": grading_path}
    if basis == BASIS_PORTABLE:
        payload["path_basis"] = BASIS_PORTABLE
    return tier_cache._digest_of(payload)


def old_instruments(pairs, old_root: Path, keys) -> dict:
    """``{stored instrument: (engine, tier, basis)}`` reproducible from the old export's bytes."""
    from merlin.common.provenance import source_digest
    from merlin.targetgen import tier_cache

    olds = [old for _, old in pairs]
    digests = {
        BASIS_ABSOLUTE: source_digest([str(path) for path in olds]),
        BASIS_PORTABLE: portable_digest(olds, old_root),
    }
    return {
        _instrument(engine, tier, digest, basis): (engine, tier, basis)
        for engine, tier in keys
        for basis, digest in digests.items()
    }


def is_rtl(record: dict) -> bool:
    result = record.get("tier_result") or {}
    return result.get("derived_from_rtl") is True or result.get("fidelity") == "elaborated_rtl"


def earned_by(record: dict, run_id: str) -> bool:
    """Whether the stored record names a package of ``run_id`` (the store is shared across runs)."""
    result = record.get("tier_result") or {}
    submission = result.get("submission") if isinstance(result.get("submission"), dict) else {}
    for value in (submission.get("package"), result.get("toolchain")):
        if isinstance(value, str) and run_id in Path(value).parts:
            return True
    return False


def _records(store: Path):
    for path in sorted(Path(store).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("status") == "pass" and isinstance(record.get("tier_result"), dict):
            yield record


def rekey(*, store, seed_framework, seed_run_id, target, certificates=True, results=True, dest=None,
          new_root=None, files=None, reviewed=None) -> dict:
    """Re-key the seed run's carried verdicts into ``dest``; returns what was carried and why not."""
    from merlin.common.paths import checkout_root
    from merlin.targetgen import tier_cache

    store, old_root = Path(store), Path(seed_framework)
    new_root = Path(new_root) if new_root is not None else checkout_root()
    dest = Path(dest) if dest is not None else tier_cache.cache_root()
    report = {"schema": SCHEMA, "seed_run": seed_run_id, "seed_framework": str(old_root), "store": str(store),
              "destination": str(dest) if dest else None, "selected": {"certificates": certificates,
                                                                      "results": results},
              "carried": 0, "already_portable": 0, "judge_unreproducible": 0, "not_selected": 0,
              "other_runs": 0, "by_tier": {}}
    files = list(files) if files is not None else list(tier_cache.grading_path(target) or ())
    if not files or new_root is None or dest is None:
        report["decision"] = "re-run: the grading path, checkout or certificate store of this export is unavailable"
        return report
    pairs = relocate(files, new_root, old_root)
    judge = compare_judges(pairs, new_root, reviewed)
    report["judge"] = judge
    seed = [r for r in _records(store) if earned_by(r, seed_run_id)]
    report["other_runs"] = sum(1 for _ in _records(store)) - len(seed)
    wanted = [r for r in seed if (certificates if is_rtl(r) else results)]
    report["not_selected"] = len(seed) - len(wanted)
    if not judge["identical"]:
        report["decision"] = ("re-run: grading code changed between the seed's export and this one "
                              "(review the listed files to carry)")
        report["rerun"] = len(wanted)
        return report
    keys = {((r["tier_result"].get("engine") or "") if is_rtl(r) else "", str(r.get("tier"))) for r in wanted}
    known = old_instruments(pairs, old_root, keys)
    new_grading = portable_digest([new for new, _ in pairs], new_root)
    index: dict = {}
    for record in wanted:
        found = known.get(record.get("instrument"))
        if found is None:
            report["judge_unreproducible"] += 1
            continue
        engine, tier, basis = found
        instrument = _instrument(engine, tier, new_grading, BASIS_PORTABLE)
        payload = {k: record[k] for k in ("version", "capsule", "tier", "execution_identity", "status")}
        payload.update(
            instrument=instrument,
            recorded_at=record.get("recorded_at"),
            run_id=seed_run_id,
            source=f"carried from {seed_run_id}",
            carried_from={"run_id": seed_run_id, "instrument": record.get("instrument"), "path_basis": basis,
                          "rekeyed": instrument != record.get("instrument")},
            tier_result=record["tier_result"],
        )
        target_path = tier_cache._record_path(dest, payload["capsule"], tier, payload["execution_identity"],
                                              instrument)
        dest.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target_path)
        report["carried"] += 1
        report["already_portable"] += instrument == record.get("instrument")
        report["by_tier"][tier] = report["by_tier"].get(tier, 0) + 1
        index.setdefault(payload["capsule"], []).append(str(tier))
    if index:
        # Which test cases a seeded grade may expect to carry: orders its work, never a pass.
        from merlin.targetgen.seeded_grade import write_index

        report["seeded_index"] = str(write_index(dest, seed_run_id, index))
    same = "byte-identical" if not judge["equivalent_code"] and not judge["reviewed_equivalent"] else (
        "equivalent (comment/docstring-only or reviewed differences)")
    report["decision"] = (
        f"re-keyed: grading code is {same} across exports; carried verdicts keep their execution "
        "identity and take this export's portable instrument" if report["carried"] else
        "nothing carried: no seed-run verdict had a reproducible judge")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--store", help="certificate store holding the seed run's verdicts (default: this run's)")
    parser.add_argument("--seed-framework", required=True, help="the seed run's frozen framework export")
    parser.add_argument("--seed-run-id", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--no-certificates", action="store_true", help="do not carry RTL-tier certificates")
    parser.add_argument("--no-results", action="store_true", help="do not carry functional-tier results")
    parser.add_argument("--dest", help="store to write (default: this run's tier certificate store)")
    parser.add_argument("--report", help="write the report here as well")
    parser.add_argument("--reviewed", help="JSON {file: {old_sha256, new_sha256, reason}} of reviewed judge changes")
    args = parser.parse_args(argv)
    from merlin.targetgen import tier_cache

    store = Path(args.store) if args.store else tier_cache.cache_root()
    report = rekey(store=store, seed_framework=args.seed_framework, seed_run_id=args.seed_run_id,
                   target=args.target, certificates=not args.no_certificates, results=not args.no_results,
                   dest=args.dest, reviewed=json.loads(Path(args.reviewed).read_text()) if args.reviewed else None)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
