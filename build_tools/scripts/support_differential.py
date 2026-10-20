#!/usr/bin/env python3
"""Differential qualification of a target SUPPORT provider: grade fixed candidates, then compare.

A support provider (the package selected by ``MERLIN_TARGET_PATH``: backend, harness renderer, RoCC
semantics, RTL checks, simulator adapters) sits on the host side of every capsule grade. Replacing one
provider with another must not change what a candidate compiler earns. This tool makes that a
measurement instead of a claim:

``grade``
    Grades ONE candidate compiler package against an explicitly selected support provider through
    Merlin's ordinary grading path -- :func:`merlin.targetgen.capsule_grade.grade` (the function the
    Phase 1 EL4 feedback ``qa.run`` and ``fast_grade`` call), with the target's own oracle ladder
    (:func:`merlin.targetgen.capsule_runner.oracle_adapters`) restricted to the requested tiers, then
    :func:`merlin.targetgen.rtl_check_runner.screen_run` over every run dir (the EL4 ``rtlchecks``
    treatment). Per capsule it writes::

        <out>/capsules/<capsule>/capsule_result.json   normalized runner record
        <out>/capsules/<capsule>/verdict.json          the EL4 per-capsule feedback row (qa projection)
        <out>/capsules/<capsule>/rtl_checks.json       normalized RTL-check report (null when none ran)
        <out>/capsules/<capsule>/harness/...           rendered harness sources (*.c, *.h, *.S, *.ld)
        <out>/index.json                               summary rows + run metadata (wall time etc.)
        <out>/perf.json                                per-capsule tier cycles, engines, METRIC lines,
                                                       static instruction histogram (``perf``)
        <out>/_work/                                   raw grade tree (not compared)

    Normalization replaces absolute work/candidate/support/temporary paths with placeholders and drops
    durations, load samples and timestamps, so two grades of the same bytes compare equal.

``diff``
    Compares two ``grade`` output directories capsule by capsule: verdict (status, tiers earned,
    failure plane/category/tier), numeric results (numeric status, mismatch count, per-tier cycles),
    RTL-check outcomes and rendered harness sources. Prints a summary table and exits 1 when any
    VERDICT-class field differs (status, tiers, failure plane/category/tier, numeric status, mismatch
    count, RTL-check verdict or per-check outcome, or a capsule present on only one side). Cycle and
    harness differences are reported; ``--strict`` makes them fail too (exit 2).

Fresh by construction: the tier-certificate and ELF-build caches are disabled for ``grade`` (pass
``--allow-cache`` to keep them), so no verdict is carried from a grade made under another provider.

Example::

    python build_tools/scripts/support_differential.py grade \\
        --support /path/to/support-A --candidate /path/to/candidate \\
        --capsules all --tiers L2,L3 --out out/support-differential/A/cand --workers 4
    python build_tools/scripts/support_differential.py grade --support /path/to/support-B ... --out .../B/cand
    python build_tools/scripts/support_differential.py diff out/support-differential/A/cand .../B/cand
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

_TIER_ORDER = ("L0", "L1", "L2", "L3", "L4", "L5")
_ALWAYS_TIERS = ("L0", "L1")  # structural/reference floor: always graded, never an oracle adapter
_DEFAULT_CATEGORIES = ("isa", "layers", "model_slices", "_model_layers")
_HARNESS_SUFFIXES = (".c", ".h", ".S", ".s", ".ld")
_INDEX_SCHEMA = "merlin.support_differential.index.v1"

# Keys whose values are wall-clock, host-load or scheduling observations: they describe the run, not
# the verdict, and two grades of identical bytes legitimately differ in them.
_VOLATILE_KEYS = frozenset(
    {
        "timing",
        "timing_diagnostic",
        "timing_rollup",
        "concurrency",
        "load_avg",
        "nproc",
        "workers",
        "max_workers",
        "parallel_speedup",
        "pid",
        "hostname",
    }
)
_VOLATILE_SUFFIXES = ("_s", "_ms", "_sec", "_seconds", "_at", "_time", "_timestamp", "_wall")
_VOLATILE_SUBSTRINGS = ("timestamp", "elapsed", "duration")
# Run-identity keys whose value is a checkout or invocation identity rather than a verdict. Kept in
# index metadata; dropped from the compared records so a provider swap at a new commit still compares.
_IDENTITY_KEYS = frozenset({"merlin"})

# Verdict-class fields (exit 1 on difference) and informational ones (reported; exit 2 with --strict).
_VERDICT_FIELDS = (
    "status",
    "tiers",
    "failure_plane",
    "failure_category",
    "failure_tier",
    "numeric_status",
    "mismatch_count",
    "trace_status",
    "rtl_verdict",
    "rtl_outcomes",
)
_INFO_FIELDS = ("tier_cycles", "harness")


# ---------------------------------------------------------------------------------------- helpers
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tier_key(tier: str) -> int:
    return _TIER_ORDER.index(tier) if tier in _TIER_ORDER else len(_TIER_ORDER)


def _parse_tiers(text: str) -> list[str]:
    tiers = [t.strip() for t in text.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in _TIER_ORDER]
    if unknown or not tiers:
        raise SystemExit(f"--tiers: unknown or empty tier list {text!r} (expected a subset of {_TIER_ORDER})")
    return sorted(set(tiers), key=_tier_key)


class _Normalizer:
    """Replace run-specific absolute paths with stable placeholders; drop volatile keys.

    Structural (no regex): placeholders are substituted longest-prefix-first, and any path segment that
    follows a temporary-directory prefix is collapsed, because its name is random by construction.
    """

    def __init__(self, replacements: dict[str, str], temp_roots: list[str]):
        pairs = []
        for raw, label in replacements.items():
            if not raw:
                continue
            for form in {raw, str(Path(raw).resolve())}:
                pairs.append((form.rstrip("/"), label))
        self._pairs = sorted(set(pairs), key=lambda p: -len(p[0]))
        self._temp = sorted({t.rstrip("/") for t in temp_roots if t}, key=len, reverse=True)

    def text(self, s: str) -> str:
        for raw, label in self._pairs:
            if raw in s:
                s = s.replace(raw, label)
        for root in self._temp:
            s = self._collapse_after(s, root + "/")
        return s

    @staticmethod
    def _collapse_after(s: str, prefix: str) -> str:
        out, i = [], 0
        while True:
            j = s.find(prefix, i)
            if j < 0:
                out.append(s[i:])
                return "".join(out)
            out.append(s[i:j] + "<TMP>/")
            k = j + len(prefix)
            while k < len(s) and s[k] not in "/ \t\n\"',:;)]}":
                k += 1
            i = k + 1 if k < len(s) and s[k] == "/" else k

    @staticmethod
    def volatile(key: str) -> bool:
        k = key.lower()
        return (
            k in _VOLATILE_KEYS
            or k.endswith(_VOLATILE_SUFFIXES)
            or any(sub in k for sub in _VOLATILE_SUBSTRINGS)
            or k in _IDENTITY_KEYS
        )

    def doc(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                (self.text(k) if isinstance(k, str) else k): self.doc(v)
                for k, v in value.items()
                if not (isinstance(k, str) and self.volatile(k))
            }
        if isinstance(value, list):
            return [self.doc(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


# ------------------------------------------------------------------------------------ grade: setup
def _select_environment(args: argparse.Namespace) -> dict[str, str | None]:
    """Fix the process environment BEFORE any Merlin module loads a provider (plugins are process-local)."""
    support = str(Path(args.support).resolve())
    os.environ["MERLIN_TARGET_PATH"] = support
    if not args.allow_cache:
        os.environ["MERLIN_TIER_CERT_CACHE"] = "0"
        os.environ["MERLIN_ELF_BUILD_CACHE"] = "0"
    if args.rtl_facts:
        os.environ["MERLIN_RTL_FACTS"] = str(Path(args.rtl_facts).resolve())
    # A candidate entrypoint may be a `#!/usr/bin/env python3` script that imports the grader's own
    # Python dependencies (the Phase 1 harness runs it from the grader's activated environment). Put this
    # interpreter's bin directory first so that resolution does not depend on the caller's shell.
    bindir = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    if path.split(os.pathsep)[:1] != [bindir]:
        os.environ["PATH"] = bindir + (os.pathsep + path if path else "")
    keys = (
        "MERLIN_TARGET_PATH",
        "MERLIN_RTL_FACTS",
        "MERLIN_REQUIRED_RTL_ENGINE",
        "MERLIN_TIER_CERT_CACHE",
        "MERLIN_ELF_BUILD_CACHE",
        "MERLIN_FULL_LADDER",
        "MERLIN_GSIM_REQUIRE_RECEIPT",
    )
    return {k: os.environ.get(k) for k in keys}


def _candidate_target(candidate: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest = candidate / "manifest.yaml"
    if not manifest.is_file():
        manifest = candidate / "submission" / "manifest.yaml"
    try:
        doc = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read candidate manifest {manifest}: {exc}; pass --target") from exc
    target = doc.get("target") if isinstance(doc, dict) else None
    if not isinstance(target, str) or not target:
        raise SystemExit(f"candidate manifest {manifest} declares no target; pass --target")
    return target


def _copy_candidate(candidate: Path, dest: Path) -> Path:
    """Copy exactly as the Phase 1 grader does: no VCS, caches or build state; then strip stale cmake."""
    src = candidate / "submission" if (candidate / "submission" / "manifest.yaml").is_file() else candidate
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"), symlinks=True)
    for path in (dest, *dest.rglob("*")):
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | (0o700 if path.is_dir() else 0o600))
    from merlin_experiments.phase1 import run_inputs

    run_inputs.strip_build_state(dest)
    return dest


def _select_capsules(corpus: Path, categories: list[str], selection: str) -> list[Path]:
    found: dict[str, Path] = {}
    for category in categories:
        root = corpus / category
        if not root.is_dir():
            raise SystemExit(f"corpus category {root} does not exist")
        for cy in sorted(root.glob("*/capsule.yaml")):
            name = cy.parent.name
            if name in found:
                raise SystemExit(f"duplicate capsule name {name!r}: {found[name]} and {cy.parent}")
            found[name] = cy.parent
    if selection.strip() == "all":
        return [found[n] for n in sorted(found)]
    if selection.startswith("@"):
        names = [ln.strip() for ln in Path(selection[1:]).read_text(encoding="utf-8").splitlines()]
    else:
        names = [n.strip() for n in selection.split(",")]
    names = [n for n in names if n and not n.startswith("#")]
    missing = sorted(set(names) - set(found))
    if missing:
        raise SystemExit(f"--capsules names capsules absent from {categories}: {missing}")
    return [found[n] for n in sorted(set(names))]


def _stage_cohort(
    capsules: list[Path], dest: Path, tiers: list[str], *, drop_tier_cap: bool = False
) -> dict[str, dict]:
    """Copy capsules verbatim, capping ``required_oracle_tiers`` to the floor + the requested tiers.

    Mirrors :func:`merlin.targetgen.contract.materialize.materialize_public_capsules`: a pure
    intersection (never a substitution), the dropped declared tiers recorded as
    ``unreachable_required_oracle_tiers`` when no RTL tier survives, and the ceiling recorded as
    ``oracle_tier_ceiling`` so no optional adapter above it runs.
    """
    keep = set(_ALWAYS_TIERS) | set(tiers)
    ceiling = max(tiers, key=_tier_key)
    staged: dict[str, dict] = {}
    for src in capsules:
        category = src.parent.name
        d = dest / category / src.name
        shutil.copytree(src, d, symlinks=False)
        cy = d / "capsule.yaml"
        cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        declared = list(cap.get("required_oracle_tiers") or [])
        kept = [t for t in declared if t in keep]
        dropped = [t for t in declared if t not in keep]
        cap["required_oracle_tiers"] = kept
        cap["oracle_tier_ceiling"] = ceiling
        uncapped = None
        if drop_tier_cap and "max_oracle_tier" in cap:
            # PERF MEASUREMENT ONLY: the capsule caps its own correctness ceiling because a cycle-accurate
            # run of its full shape is expensive. Removing the cap buys a cycle count on that shape; it
            # is not the capsule's declared grading contract and is recorded as such.
            uncapped = cap.pop("max_oracle_tier")
        if dropped and not any(t in kept for t in _TIER_ORDER[2:]):
            cap["unreachable_required_oracle_tiers"] = dropped
        cy.write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
        staged[src.name] = {
            "category": category,
            "label": cap.get("label"),
            "kind": cap.get("kind"),
            "declared_tiers": declared,
            "graded_tiers": kept,
            "dropped_tiers": dropped,
            "source_capsule_yaml_sha256": _sha256_file(src / "capsule.yaml"),
            "dropped_max_oracle_tier": uncapped,
        }
    return staged


# ------------------------------------------------------------------------------ grade: collection
def _rtl_outcomes(report: dict | None) -> dict | None:
    if not isinstance(report, dict):
        return None
    screen = report.get("screen") or {}
    checks = {str(c.get("id")): c.get("status") for c in screen.get("checks") or [] if isinstance(c, dict)}
    for skipped in screen.get("skipped") or []:
        if isinstance(skipped, dict):
            checks.setdefault(str(skipped.get("id")), "skipped")
    return {
        "verdict": report.get("verdict"),
        "screen_verdict": screen.get("verdict"),
        "filecheck": {k: (v or {}).get("ok") for k, v in sorted((report.get("filecheck") or {}).items())},
        "checks": dict(sorted(checks.items())),
    }


def _harness_files(run_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.suffix in _HARNESS_SUFFIXES and "invocations" not in p.relative_to(run_dir).parts:
            out[str(p.relative_to(run_dir))] = p
    return out


def _summary_row(name: str, verdict: dict, rtl: dict | None, harness: dict[str, str], staged: dict) -> dict:
    tiers = verdict.get("tiers") or {}
    return {
        "capsule": name,
        "category": staged.get("category"),
        "label": staged.get("label"),
        "status": verdict.get("status"),
        "tiers": {t: tiers[t] for t in sorted(tiers, key=_tier_key)},
        "tiers_passed": sorted((t for t, s in tiers.items() if s == "pass"), key=_tier_key),
        "failure_plane": verdict.get("failure_plane"),
        "failure_category": verdict.get("failure_category"),
        "failure_tier": verdict.get("failure_tier"),
        "numeric_status": verdict.get("numeric_status"),
        "mismatch_count": verdict.get("mismatch_count"),
        "trace_status": verdict.get("trace_status"),
        "tier_cycles": {
            t: c for t, c in sorted((verdict.get("tier_cycles") or {}).items(), key=lambda i: _tier_key(i[0]))
        },
        "tier_engines": verdict.get("tier_engines") or {},
        "tiers_abandoned": verdict.get("tiers_abandoned") or [],
        "rtl_verdict": (rtl or {}).get("verdict"),
        "rtl_outcomes": rtl,
        "harness": harness,
    }


def _no_run_verdict(name: str, score: dict, grade_error: str | None) -> dict:
    """Verdict for a capsule that left no run dir, taken from the grade's own score record.

    Either the package was refused before any capsule ran (load/integrity/build: the same failure for
    every capsule), or the grade withheld the capsule as not measured (e.g. outside the target's
    declared capability), or nothing was recorded at all.
    """
    pkg_fail = score.get("failure") if isinstance(score.get("failure"), dict) else None
    if pkg_fail:
        return {
            "status": "package_failure",
            "failure_plane": pkg_fail.get("plane"),
            "failure_category": str(pkg_fail.get("category")),
        }
    not_measured = (score.get("not_measured_status") or {}).get(name)
    if not_measured:
        return {"status": str(not_measured), "failure_plane": "not_measured", "failure_category": str(not_measured)}
    return {"status": "no_result", "failure_plane": "grade_error" if grade_error else None}


def _totals(rows: list[dict]) -> dict:
    totals: dict[str, Any] = {"capsules": len(rows), "status": {}, "tier_pass": {}, "failure_plane": {}}
    for row in rows:
        totals["status"][str(row["status"])] = totals["status"].get(str(row["status"]), 0) + 1
        for t in row["tiers_passed"]:
            totals["tier_pass"][t] = totals["tier_pass"].get(t, 0) + 1
        if row["status"] != "pass":
            key = f"{row['failure_plane']}/{row['failure_category']}"
            totals["failure_plane"][key] = totals["failure_plane"].get(key, 0) + 1
    totals["rtl_verdict"] = {}
    for row in rows:
        totals["rtl_verdict"][str(row["rtl_verdict"])] = totals["rtl_verdict"].get(str(row["rtl_verdict"]), 0) + 1
    return totals


def cmd_grade(args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    started = _utc_now()
    env = _select_environment(args)
    tiers = _parse_tiers(args.tiers)
    candidate_src = Path(args.candidate).resolve()
    support = Path(args.support).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"--out {out} is not empty; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir()

    from merlin.common.paths import repo_root
    from merlin.common.tree_hash import hash_tree

    repo = repo_root()
    corpus = Path(args.corpus_root).resolve() if args.corpus_root else repo / "merlin" / "contract" / "capsules"
    contract = repo / "merlin" / "contract"
    target = _candidate_target(candidate_src, args.target)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    selected = _select_capsules(corpus, categories, args.capsules)
    if not selected:
        raise SystemExit("no capsules selected")

    cand = _copy_candidate(candidate_src, work / "candidate")
    candidate_sha = hash_tree(cand)["sha256"]
    cohort = work / "capsules"
    staged = _stage_cohort(selected, cohort, tiers, drop_tier_cap=args.drop_tier_cap)
    runs_root = work / "runs"

    from merlin_experiments.phase1.feedback import qa as QA

    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import rtl_check_runner as RUN

    full = CR.oracle_adapters(target)
    adapters = {t: a for t, a in full.items() if t in tiers}
    unreachable = [t for t in tiers if t not in full]
    print(f"[support-differential] target={target} support={support}", flush=True)
    print(
        f"[support-differential] {len(selected)} capsule(s); tiers requested={tiers} "
        f"adapters={sorted(adapters)} unreachable={unreachable}",
        flush=True,
    )
    t_grade = time.perf_counter()
    score: dict[str, Any] = {}
    grade_error = None
    try:
        score = CG.grade(
            str(cand),
            capsules_root=str(cohort),
            runs_root=str(runs_root),
            model_snapshot_root=runs_root / ".private_model_sources",
            labels={"public", "dev", "hidden"},
            contract=str(contract),
            oracle_adapters=adapters,
            timeout=args.timeout,
            max_workers=args.workers,
            target=target,
        )
    except Exception as exc:  # noqa: BLE001 -- recorded; whatever capsules finished are still collected
        grade_error = f"{type(exc).__name__}: {exc}"
        print(f"[support-differential] grade raised {grade_error}", flush=True)
    grade_wall = time.perf_counter() - t_grade

    from merlin.targetgen.rtl.facts import load_facts, rtl_facts_path

    facts_rec = None
    facts_error = None
    try:
        facts_rec = load_facts(target)
    except Exception as exc:  # noqa: BLE001 -- RTL checks then report unavailable for every capsule
        facts_error = f"{type(exc).__name__}: {exc}"
    fc_candidates = [Path(args.filecheck)] if args.filecheck else []
    mlir_install = os.environ.get("MERLIN_MLIR_INSTALL")
    if mlir_install:
        fc_candidates.append(Path(mlir_install) / "bin" / "FileCheck")
    filecheck = RUN.find_filecheck(fc_candidates)
    index = RUN.capsule_index([cohort])

    norm = _Normalizer(
        {
            str(cand): "<CANDIDATE>",
            str(cohort): "<COHORT>",
            str(runs_root): "<RUNS>",
            str(work): "<WORK>",
            str(out): "<OUT>",
            str(support): "<SUPPORT>",
            str(candidate_src): "<CANDIDATE_SRC>",
            str(repo): "<REPO>",
        },
        [tempfile.gettempdir(), "/tmp", os.environ.get("TMPDIR", "")],
    )
    verdicts = QA._per_capsule_from_results(runs_root)
    rows: list[dict] = []
    suite_dirs = (
        sorted(p for p in (runs_root / "runs").glob("*") if p.is_dir()) if (runs_root / "runs").is_dir() else []
    )
    run_dirs = {d.name: d for s in suite_dirs for d in s.iterdir() if (d / "capsule_result.json").is_file()}
    for name in sorted(staged):
        cdir = out / "capsules" / name
        cdir.mkdir(parents=True, exist_ok=True)
        run_dir = run_dirs.get(name)
        if run_dir is None:
            stub = _no_run_verdict(name, score, grade_error)
            _write_json(cdir / "capsule_result.json", None)
            _write_json(cdir / "verdict.json", norm.doc(stub))
            _write_json(cdir / "rtl_checks.json", None)
            row = _summary_row(name, norm.doc(stub), None, {}, staged[name])
            rows.append(row)
            continue
        result = _read_json(run_dir / "capsule_result.json")
        _write_json(cdir / "capsule_result.json", norm.doc(result))
        verdict = verdicts.get(name) or {}
        _write_json(cdir / "verdict.json", norm.doc(verdict))
        rtl = None
        if facts_rec is not None:
            try:
                rtl = RUN.screen_run(run_dir, facts_rec, index, filecheck, write=False, target=target)
            except Exception as exc:  # noqa: BLE001 -- advisory: a raising check is an outcome, not a crash
                rtl = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}"}
        else:
            rtl = {"verdict": "unavailable", "error": facts_error}
        _write_json(cdir / "rtl_checks.json", norm.doc(rtl))
        harness: dict[str, str] = {}
        for rel, src in _harness_files(run_dir).items():
            text = norm.text(src.read_text(encoding="utf-8", errors="replace"))
            dst = cdir / "harness" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            harness[rel] = _sha256_bytes(text.encode("utf-8"))
        rows.append(_summary_row(name, norm.doc(verdict), norm.doc(_rtl_outcomes(rtl)), harness, staged[name]))

    totals = _totals(rows)

    facts_path = rtl_facts_path(target)
    support_sha = None
    try:
        support_sha = hash_tree(support)["sha256"]
    except Exception:  # noqa: BLE001 -- identity is metadata; absence is recorded, not fatal
        pass
    index_doc = {
        "schema": _INDEX_SCHEMA,
        "target": target,
        "tiers_requested": tiers,
        "adapters": sorted(adapters),
        "tiers_unreachable": unreachable,
        "rows": rows,
        "totals": totals,
        "meta": {
            "note": "meta is run identity and cost; `diff` never compares it",
            "started_utc": started,
            "finished_utc": _utc_now(),
            "wall_s": round(time.perf_counter() - t_start, 1),
            "grade_wall_s": round(grade_wall, 1),
            "workers": args.workers,
            "timeout_s": args.timeout,
            "support": str(support),
            "support_tree_sha256": support_sha,
            "candidate": str(candidate_src),
            "candidate_tree_sha256": candidate_sha,
            "corpus_root": str(corpus),
            "categories": categories,
            "capsule_selection": args.capsules,
            "staged": staged,
            "merlin_head": _git_head(repo),
            "python": sys.executable,
            "environment": env,
            "rtl_facts": str(facts_path),
            "rtl_facts_sha256": _sha256_file(facts_path) if facts_path.is_file() else None,
            "filecheck": filecheck,
            "grade_error": grade_error,
            "score": norm.doc({k: v for k, v in score.items() if k != "per_capsule"}),
        },
    }
    _write_json(out / "index.json", index_doc)
    _write_json(out / "perf.json", perf_rows(out))
    if not args.keep_work:
        shutil.rmtree(work / "candidate", ignore_errors=True)
    _print_grade_summary(index_doc)
    return 0 if grade_error is None else 3


def _print_grade_summary(doc: dict) -> None:
    t = doc["totals"]
    m = doc["meta"]
    print(f"\n[support-differential] {doc['target']} tiers={doc['tiers_requested']} capsules={t['capsules']}")
    print(f"  status:      {t['status']}")
    print(f"  tier passes: {t['tier_pass']}")
    print(f"  failures:    {t['failure_plane']}")
    print(f"  rtl checks:  {t['rtl_verdict']}")
    print(f"  wall: {m['wall_s']} s (grade {m['grade_wall_s']} s)")


# -------------------------------------------------------------------------------------------- diff
def _load_side(root: Path) -> tuple[dict, dict[str, dict]]:
    doc = _read_json(root / "index.json")
    if doc.get("schema") != _INDEX_SCHEMA:
        raise SystemExit(f"{root}/index.json is not a {_INDEX_SCHEMA} document")
    return doc, {row["capsule"]: row for row in doc.get("rows") or []}


def _differing_paths(a: Any, b: Any, prefix: str = "", limit: int = 12) -> list[str]:
    out: list[str] = []

    def walk(x: Any, y: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                walk(x.get(k, "<absent>"), y.get(k, "<absent>"), f"{p}.{k}" if p else str(k))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{p}[{i}]")
        elif x != y:
            out.append(p or "<root>")

    walk(a, b, prefix)
    return out


def _fmt_tiers(row: dict | None) -> str:
    if row is None:
        return "-"
    passed = row.get("tiers_passed") or []
    return f"{row.get('status')}[{','.join(passed) or '-'}]"


def cmd_diff(args: argparse.Namespace) -> int:
    a_root, b_root = Path(args.a).resolve(), Path(args.b).resolve()
    a_doc, a_rows = _load_side(a_root)
    b_doc, b_rows = _load_side(b_root)
    names = sorted(set(a_rows) | set(b_rows))
    verdict_diffs: dict[str, list[str]] = {}
    info_diffs: dict[str, list[str]] = {}
    record_diffs: dict[str, list[str]] = {}
    for name in names:
        ra, rb = a_rows.get(name), b_rows.get(name)
        if ra is None or rb is None:
            verdict_diffs[name] = ["present on one side only"]
            continue
        v = [f for f in _VERDICT_FIELDS if ra.get(f) != rb.get(f)]
        i = [f for f in _INFO_FIELDS if ra.get(f) != rb.get(f)]
        if v:
            verdict_diffs[name] = v
        if i:
            info_diffs[name] = i
        if args.records:
            ca = a_root / "capsules" / name / "capsule_result.json"
            cb = b_root / "capsules" / name / "capsule_result.json"
            if ca.is_file() and cb.is_file():
                paths = _differing_paths(_read_json(ca), _read_json(cb))
                if paths:
                    record_diffs[name] = paths

    width = max([len(n) for n in names] + [8])
    print(f"A: {a_root}\n   support={a_doc['meta'].get('support')}")
    print(f"B: {b_root}\n   support={b_doc['meta'].get('support')}")
    if a_doc.get("tiers_requested") != b_doc.get("tiers_requested"):
        print(f"WARNING: tiers differ: A={a_doc.get('tiers_requested')} B={b_doc.get('tiers_requested')}")
    if a_doc["meta"].get("candidate_tree_sha256") != b_doc["meta"].get("candidate_tree_sha256"):
        print("WARNING: the two sides graded different candidate bytes")
    if a_doc["meta"].get("rtl_facts_sha256") != b_doc["meta"].get("rtl_facts_sha256"):
        print("WARNING: the two sides read different RTL facts")
    print()
    header = f"{'capsule':{width}}  {'A':22}  {'B':22}  differences"
    print(header)
    print("-" * len(header))
    for name in names:
        diffs = verdict_diffs.get(name, []) + [f"({f})" for f in info_diffs.get(name, [])]
        if not diffs and not args.all:
            continue
        left, right = _fmt_tiers(a_rows.get(name)), _fmt_tiers(b_rows.get(name))
        print(f"{name:{width}}  {left:22}  {right:22}  {', '.join(diffs) or 'same'}")
        if args.verbose:
            for f in verdict_diffs.get(name, []) + info_diffs.get(name, []):
                if f == "present on one side only":
                    continue
                va = json.dumps((a_rows.get(name) or {}).get(f), sort_keys=True)[:300]
                vb = json.dumps((b_rows.get(name) or {}).get(f), sort_keys=True)[:300]
                print(f"{'':{width}}    {f}: A={va}")
                print(f"{'':{width}}    {'':{len(f)}}  B={vb}")
            for p in record_diffs.get(name, []):
                print(f"{'':{width}}    record: {p}")
    print()
    print(
        f"{len(names)} capsule(s): {len(verdict_diffs)} with VERDICT differences, "
        f"{len(info_diffs)} with cycle/harness differences"
        + (f", {len(record_diffs)} with other normalized-record differences" if args.records else "")
    )
    print(f"A totals: {a_doc['totals']['status']} tier passes {a_doc['totals']['tier_pass']}")
    print(f"B totals: {b_doc['totals']['status']} tier passes {b_doc['totals']['tier_pass']}")
    if args.json:
        _write_json(
            Path(args.json),
            {"a": str(a_root), "b": str(b_root), "verdict": verdict_diffs, "info": info_diffs, "records": record_diffs},
        )
    if verdict_diffs:
        return 1
    if args.strict and info_diffs:
        return 2
    return 0


# -------------------------------------------------------------------------------------------- perf
_PERF_SCHEMA = "merlin.support_differential.perf.v1"


def _console_metrics(path: Path) -> dict[str, str]:
    """``METRIC <name> <value>`` lines a harness prints (structural split, no regex)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "METRIC":
            out[parts[1]] = " ".join(parts[2:])
    return out


def perf_rows(grade_dir: Path) -> dict:
    """Per-capsule cost record of one ``grade`` output: tier cycles, engines, METRIC lines and the
    STATIC decoded instruction histogram of the emitted program (not a dynamic count: a loop counts once)."""
    run_dirs = {d.name: d for d in (grade_dir / "_work" / "runs" / "runs").glob("*/*") if d.is_dir()}
    if (grade_dir / "index.json").is_file():
        doc = _read_json(grade_dir / "index.json")
    else:
        # An interrupted grade wrote no index: report every capsule that finished, marked partial.
        doc = {"target": None, "meta": {"partial": "grade interrupted before index.json was written"}, "rows": []}
        for name, rd in sorted(run_dirs.items()):
            if (rd / "capsule_result.json").is_file():
                rec = _read_json(rd / "capsule_result.json")
                tiers = rec.get("tiers") or {}
                doc["rows"].append(
                    {
                        "capsule": name,
                        "category": None,
                        "status": rec.get("status"),
                        "tiers_passed": sorted(
                            (t for t, v in tiers.items() if isinstance(v, dict) and v.get("status") == "pass"),
                            key=_tier_key,
                        ),
                    }
                )
    rows = []
    for row in doc.get("rows") or []:
        name = row["capsule"]
        rd = run_dirs.get(name)
        result = _read_json(rd / "capsule_result.json") if rd and (rd / "capsule_result.json").is_file() else {}
        tiers = {}
        for tier, rec in sorted((result.get("tiers") or {}).items(), key=lambda i: _tier_key(i[0])):
            if not isinstance(rec, dict) or tier in _ALWAYS_TIERS:
                continue
            log = rec.get("console_log")
            tiers[tier] = {
                "status": rec.get("status"),
                "cycles": rec.get("cycles"),
                "engine": rec.get("engine"),
                "cycle_accurate": rec.get("cycle_accurate"),
                "derived_from_rtl": rec.get("derived_from_rtl"),
                "metrics": _console_metrics(rd / "artifacts" / log) if rd and isinstance(log, str) else {},
                "reason": (str(rec.get("reason"))[:400] if rec.get("status") != "pass" and rec.get("reason") else None),
            }
        hist = None
        trace = rd / "generated" / "instruction_trace.json" if rd else None
        if trace is not None and trace.is_file():
            try:
                hist = (_read_json(trace).get("summary") or {}).get("class_histogram")
            except ValueError:
                hist = None
        work = result.get("work_volume") or {}
        move = result.get("movement_volume") or {}
        rows.append(
            {
                "capsule": name,
                "category": row.get("category"),
                "status": row.get("status"),
                "tiers_passed": row.get("tiers_passed"),
                "tiers": tiers,
                "rtl_cycles": next(
                    (t["cycles"] for t in tiers.values() if t.get("derived_from_rtl") and t.get("cycles") is not None),
                    None,
                ),
                "rtl_status": next((t["status"] for t in tiers.values() if t.get("derived_from_rtl")), None),
                "static_instruction_histogram": hist,
                "static_instruction_count": sum(hist.values()) if isinstance(hist, dict) else None,
                "exact_macs": work.get("exact_macs"),
                "movement_bytes": move.get("exact_bytes"),
            }
        )
    return {
        "schema": _PERF_SCHEMA,
        "target": doc.get("target"),
        "partial": doc["meta"].get("partial"),
        "candidate": doc["meta"].get("candidate"),
        "candidate_tree_sha256": doc["meta"].get("candidate_tree_sha256"),
        "support": doc["meta"].get("support"),
        "rtl_facts_sha256": doc["meta"].get("rtl_facts_sha256"),
        "note": (
            "cycles are the harness-reported cycle window (METRIC cycles) of each tier's simulator; "
            "rtl_cycles is the first RTL-derived tier (elaborated RTL engine). Values on a failed tier are "
            "the cost of a wrong program and are not a performance reference. "
            "static_instruction_histogram counts decoded instructions in the emitted program text."
        ),
        "rows": rows,
    }


def cmd_perf(args: argparse.Namespace) -> int:
    combined = {"schema": _PERF_SCHEMA + ".set", "candidates": {}}
    for d in args.dirs:
        grade_dir = Path(d).resolve()
        doc = perf_rows(grade_dir)
        _write_json(grade_dir / "perf.json", doc)
        combined["candidates"][grade_dir.name] = doc
        n = sum(1 for r in doc["rows"] if r["rtl_status"] == "pass")
        print(f"{grade_dir.name}: {len(doc['rows'])} capsule(s), {n} with a passing RTL tier -> {grade_dir}/perf.json")
    if args.out:
        _write_json(Path(args.out), combined)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Re-derive no-run rows and totals of existing grade outputs from their recorded score."""
    for d in args.dirs:
        root = Path(d).resolve()
        doc = _read_json(root / "index.json")
        score = doc["meta"].get("score") or {}
        for i, row in enumerate(doc["rows"]):
            if _read_json(root / "capsules" / row["capsule"] / "capsule_result.json") is not None:
                continue
            stub = _no_run_verdict(row["capsule"], score, doc["meta"].get("grade_error"))
            staged = (doc["meta"].get("staged") or {}).get(row["capsule"]) or {}
            doc["rows"][i] = _summary_row(row["capsule"], stub, None, {}, staged)
            _write_json(root / "capsules" / row["capsule"] / "verdict.json", stub)
        doc["totals"] = _totals(doc["rows"])
        _write_json(root / "index.json", doc)
        _write_json(root / "perf.json", perf_rows(root))
        print(f"{root.name}: {doc['totals']['status']}")
    return 0


# -------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="grade one candidate against one support provider")
    g.add_argument("--support", required=True, help="support provider root (becomes MERLIN_TARGET_PATH)")
    g.add_argument("--candidate", required=True, help="candidate compiler package (manifest.yaml at its root)")
    g.add_argument(
        "--capsules",
        default="all",
        help="'all', a comma-separated capsule-name list, or @file with one name per line (default all)",
    )
    g.add_argument("--tiers", default="L2,L3", help="oracle tiers to grade, e.g. L2,L3 (L0/L1 always run)")
    g.add_argument("--out", required=True, help="output directory (must be empty unless --force)")
    g.add_argument("--workers", type=int, default=4, help="concurrent capsules (= concurrent simulators)")
    g.add_argument("--timeout", type=int, default=1800, help="per-capsule oracle timeout in seconds")
    g.add_argument(
        "--categories",
        default=",".join(_DEFAULT_CATEGORIES),
        help=f"corpus sub-directories to draw capsules from (default {','.join(_DEFAULT_CATEGORIES)})",
    )
    g.add_argument("--corpus-root", default=None, help="capsule corpus root (default <repo>/merlin/contract/capsules)")
    g.add_argument("--target", default=None, help="target name (default: the candidate manifest's target)")
    g.add_argument("--rtl-facts", default=None, help="RTL facts artifact (sets MERLIN_RTL_FACTS)")
    g.add_argument("--filecheck", default=None, help="explicit FileCheck binary")
    g.add_argument("--allow-cache", action="store_true", help="keep tier-certificate and ELF-build caches on")
    g.add_argument("--keep-work", action="store_true", help="keep the candidate copy under _work/")
    g.add_argument("--force", action="store_true", help="replace a non-empty --out")
    g.add_argument(
        "--drop-tier-cap",
        action="store_true",
        help="PERF ONLY: remove capsules' own max_oracle_tier so capped capsules also run the cert tier "
        "(recorded per capsule as dropped_max_oracle_tier; not the declared grading contract)",
    )
    g.set_defaults(func=cmd_grade)

    d = sub.add_parser("diff", help="compare two grade output directories")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--all", action="store_true", help="list identical capsules too")
    d.add_argument("--verbose", "-v", action="store_true", help="print both values of each differing field")
    d.add_argument("--records", action="store_true", help="also diff the full normalized capsule_result.json")
    d.add_argument("--strict", action="store_true", help="exit 2 on cycle/harness differences")
    d.add_argument("--json", default=None, help="write the difference report to this JSON file")
    d.set_defaults(func=cmd_diff)

    pf = sub.add_parser("perf", help="(re)write perf.json for grade output directories")
    pf.add_argument("dirs", nargs="+")
    pf.add_argument("--out", default=None, help="also write all candidates into one combined JSON file")
    pf.set_defaults(func=cmd_perf)

    ri = sub.add_parser("reindex", help="re-derive no-run rows, totals and perf.json of grade outputs")
    ri.add_argument("dirs", nargs="+")
    ri.set_defaults(func=cmd_reindex)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
