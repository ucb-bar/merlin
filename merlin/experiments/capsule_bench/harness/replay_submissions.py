#!/usr/bin/env python3
"""Replay every frozen agent submission through the CURRENT runner. Offline, free, deterministic.

Why this exists
---------------
The offline gates we run before a campaign (`readiness_check`, `preflight`, `test_sandbox`) all
exercise the harness against a package that is either CORRECT (the reference backend), ABSENT
(missing manifest), EMPTY, or DELIBERATELY CHEATING (a forbidden import). None of them exercise a
package that is present, schema-valid, and semantically WRONG -- which is what every real agent
produces, and therefore the only input the grading path ever sees in production.

Every harness defect measured in the 2026-08 model campaign lived in that untested middle ground: the
runner was correct on a correct package and misreported a wrong one as a different kind of wrong. The
cost was twelve agent rounds and a mismeasured model.

This tool closes the gap using evidence we already have: the frozen `submission/` trees left behind by
past runs are a free corpus of real, adversarial, contract-violating packages. Replaying them costs
nothing and needs no agent, no oracle and no money.

What it checks
--------------
For every declared command of every discovered submission, resolve the argv the runner WOULD execute
and classify the outcome:

``ok``           every placeholder substituted and the referenced script exists under the package root
``actionable``   the runner refuses with a ``CertFailure`` that NAMES the problem -- a good failure
``unactionable`` the runner would hand the package an argv that cannot work, and the agent would see a
                 bare ``FileNotFoundError`` from inside its own traceback -- indistinguishable from its
                 compiler being broken. EVERY entry here is a harness defect waiting to be paid for.

A non-empty ``unactionable`` bucket is the finding; the exit status reflects it so this can gate a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common.paths import runs_dir
from merlin.targetgen.oot_runner import CertFailure, _resolve_argv, load_package

# Tokens the runner substitutes. A surviving "{...}" in a resolved argv means the manifest named
# something the runner does not provide -- the package receives it verbatim and dies opaquely.
_OPEN, _CLOSE = "{", "}"


def _unsubstituted(tok: str) -> str | None:
    """Return the first surviving placeholder in `tok`, or None. Structural scan, no regex."""
    lo = tok.find(_OPEN)
    if lo == -1:
        return None
    hi = tok.find(_CLOSE, lo)
    return tok[lo:hi + 1] if hi != -1 else None


def _looks_like_a_path(tok: str) -> bool:
    """A token worth existence-checking: it names a file rather than a flag or a bare word."""
    return ("/" in tok or tok.endswith(".py")) and not tok.startswith("-")


def discover(root: Path) -> list[Path]:
    """Every frozen submission under `root` (skips per-round scratch copies under _qa_work/)."""
    out = []
    for m in root.rglob("submission/manifest.yaml"):
        if "_qa_work" in m.parts or "__pycache__" in m.parts:
            continue
        out.append(m.parent)
    return sorted(out)


def replay(pkg_dir: Path, *, input_mlir: Path) -> list[dict]:
    """Classify every command of one package. Never executes the package -- resolution only."""
    rows: list[dict] = []
    try:
        pkg = load_package(pkg_dir)
    except CertFailure as e:
        return [{"command": "<load>", "verdict": "actionable", "note": str(e)[:200]}]
    except Exception as e:  # noqa: BLE001 -- an unreadable package is a finding, not a crash
        return [{"command": "<load>", "verdict": "unactionable",
                 "note": f"{type(e).__name__}: {str(e)[:180]}"}]

    for name in sorted((pkg.manifest.get("commands") or {})):
        row: dict = {"command": name}
        try:
            argv = _resolve_argv(pkg, name, input_mlir, pkg_dir / "_replay_out.json")
        except CertFailure as e:
            row.update(verdict="actionable", note=str(e)[:200])
            rows.append(row)
            continue
        except Exception as e:  # noqa: BLE001
            row.update(verdict="unactionable", note=f"{type(e).__name__}: {str(e)[:180]}")
            rows.append(row)
            continue

        bad = next((p for p in (_unsubstituted(t) for t in argv) if p), None)
        if bad:
            row.update(verdict="unactionable", note=f"unsubstituted placeholder {bad!r} reaches the package")
            rows.append(row)
            continue

        missing = [t for t in argv[1:]
                   if _looks_like_a_path(t) and not Path(t).is_absolute()
                   and not (pkg.directory / t).exists()]
        if missing:
            row.update(verdict="unactionable",
                       note=f"argv names {missing[0]!r}, absent from the package root")
        else:
            row.update(verdict="ok", note="")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=None, help="where to look for frozen submissions")
    ap.add_argument("--input", default=None, help="interface MLIR to resolve against")
    ap.add_argument("--json", action="store_true", help="emit the full table as JSON")
    a = ap.parse_args(argv)

    root = Path(a.runs_root).resolve() if a.runs_root else runs_dir()
    src = Path(a.input).resolve() if a.input else None
    if src is None:
        from merlin.common.paths import repo_root
        src = repo_root() / "merlin/contract/examples/g0_matmul.interface.mlir"

    pkgs = discover(root)
    if not pkgs:
        print(f"no frozen submissions under {root} — nothing to replay")
        return 0

    table, tally = [], {"ok": 0, "actionable": 0, "unactionable": 0}
    for p in pkgs:
        for row in replay(p, input_mlir=src):
            row["package"] = str(p)
            tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1
            table.append(row)

    if a.json:
        print(json.dumps({"tally": tally, "rows": table}, indent=2))
    else:
        print(f"=== replayed {len(pkgs)} frozen submission(s) from {root} ===")
        print(f"  ok           {tally['ok']:4d}   argv resolves and every referenced file exists")
        print(f"  actionable   {tally['actionable']:4d}   runner refuses and NAMES the problem")
        print(f"  unactionable {tally['unactionable']:4d}   agent would see an opaque error  <-- harness defects")
        shown = 0
        for r in table:
            if r["verdict"] != "unactionable" or shown >= 15:
                continue
            shown += 1
            print(f"\n  [{r['command']}] {Path(r['package']).parent.name}")
            print(f"      {r['note']}")
    return 1 if tally["unactionable"] else 0


if __name__ == "__main__":
    sys.exit(main())
