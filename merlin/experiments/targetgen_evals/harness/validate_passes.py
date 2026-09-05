"""Execute the MLIR pass tests, and classify the ones that cannot execute — with reasons.

History, because it explains the shape: this validator used to COUNT `.mlir` files and report
`pass_tests_total: 8` while executing none of them. The count read like coverage. That is the
"a check that could not run reported success" failure this repo keeps re-encountering, and it is why
nothing here reports a bare number any more.

Two populations, kept separate on purpose:

* **The executable suite** (``merlin/tests/data/lit``) — real ``// RUN:`` lines against ``merlin-opt``
  and upstream ``mlir-opt``, run through ``llvm-lit``. Real pass/fail numbers.
* **Dataset specimens** (``datasets/<target>/tests``) — written before ``merlin-opt`` existed, naming
  passes that were never implemented and carrying placeholder bodies. Each is classified by whether
  the pass its RUN line names is actually registered. They are reported as ``unexecutable`` with the
  reason, and are NEVER added to a pass/total count.

RUN lines are parsed structurally (``split``/``partition``), never by pattern matching — the repo's
no-regex rule exists because a too-narrow pattern silently drops valid input.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _registered_passes() -> set[str]:
    """Pass names ``merlin-opt`` can actually run. Empty set if the driver is unavailable."""
    try:
        from merlin.xdsl_dialects.opt import merlin_passes
        ok, _ = merlin_passes()
        return set(ok)
    except Exception:
        return set()


def _passes_named_by(path: Path) -> list[str]:
    """Pass names a file's RUN lines request via ``-p``. Structural parse, no regex."""
    named: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, sep, rest = line.partition("// RUN:")
        if not sep:
            continue
        tokens = rest.split()
        for i, tok in enumerate(tokens):
            if tok == "-p" and i + 1 < len(tokens):
                named.append(tokens[i + 1].strip("\"'"))
    return named


def _classify_specimens(tests_dir: Path, registered: set[str]) -> list[dict]:
    out: list[dict] = []
    for path in sorted(tests_dir.rglob("*.mlir")):
        wanted = _passes_named_by(path)
        missing = [p for p in wanted if p not in registered]
        out.append({
            "file": path.name,
            "passes_named": wanted,
            "executable": bool(wanted) and not missing,
            "reason": ("" if not missing else
                       "names pass(es) that are not registered: " + ", ".join(sorted(missing)))
            if wanted else "no -p pass named in any RUN line",
        })
    return out


def _run_lit_suite() -> dict:
    from merlin.common.paths import merlin_dir
    from merlin.verify import tools

    lit, fc = tools.find_lit(), tools.find_filecheck()
    suite = merlin_dir() / "tests" / "data" / "lit"
    if not (lit and fc and suite.is_dir()):
        return {"status": "unavailable",
                "reason": f"llvm-lit/FileCheck not found ({tools.availability()})"}
    r = subprocess.run([lit, "-s", str(suite)], capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    discovered = 0
    if "Total Discovered Tests:" in out:
        discovered = int(out.split("Total Discovered Tests:")[1].split()[0])
    return {"status": "ok" if r.returncode == 0 else "failed",
            "discovered": discovered, "returncode": r.returncode,
            "output_tail": out[-2000:]}


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    tests_dir = Path(__file__).parent.parent / "datasets" / target / "tests"
    registered = _registered_passes()

    metrics: dict = {
        "validator": "passes",
        "executable_suite": _run_lit_suite(),
        "specimens": [],
        "specimens_executable": 0,
        "specimens_unexecutable": 0,
        "errors": [],
    }
    if not registered:
        metrics["errors"].append("merlin-opt unavailable; specimen classification is UNMEASURED")

    if not tests_dir.exists():
        metrics["errors"].append(f"no dataset specimens at {tests_dir}")
        return metrics

    specimens = _classify_specimens(tests_dir, registered)
    metrics["specimens"] = specimens
    metrics["specimens_executable"] = sum(1 for s in specimens if s["executable"])
    metrics["specimens_unexecutable"] = sum(1 for s in specimens if not s["executable"])
    if metrics["specimens_unexecutable"]:
        metrics["errors"].append(
            f"{metrics['specimens_unexecutable']} specimen(s) cannot execute; see 'specimens' for "
            "the per-file reason. These are NOT counted as tests.")
    return metrics
