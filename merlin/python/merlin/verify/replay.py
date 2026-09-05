"""Replay historical fixes past the verification layers, on a BLIND sample.

**The question this answers, and the weaker answer it replaces.** "Would this layer have caught real
defects?" was first answered by walking a hand-picked list of seven fixes and reporting six caught.
That number is unusable: the seven were chosen after the layers existed, by someone who knew what the
layers check. It is a demonstration, not a measurement, and it must not be cited as a rate.

This module measures instead. The population is every `fix(` commit that touched a file the layers can
see; the sample is drawn by a SEEDED shuffle recorded in the artifact, before any outcome is known; and
every commit drawn is reported, including the ones that could not be replayed at all.

**How a "would it have caught this" verdict is produced.** The layers did not exist when these commits
landed, so they cannot be run at the historical tree. Instead the DEFECT is brought forward: for one
sampled commit, the parent's version of each library file it touched is written into a shadow copy of
the package, and the layers run against that shadow. A layer that is green on the real tree and red on
the shadow would have caught that defect.

The shadow is a whole-package copy rather than a `sys.path` prepend, because `merlin` is a regular
package (it has `__init__.py`): the first `merlin/` on the path wins ENTIRELY, so a partial shadow
would hide every module it does not contain and every layer would fail for the wrong reason.

**What is deliberately not counted.**

* A layer already red on the real tree is disqualified for that run -- it cannot be credited with a
  detection it would have produced anyway. Measured per run, not assumed.
* A commit whose parent files no longer apply (the file was deleted, renamed, or restructured past
  recognition) is `unreplayable`. It is REPORTED, never dropped and never folded into "missed": the
  denominator is what makes the number honest, and silently shrinking it is how a detection rate gets
  flattered.
* A commit that touches no library file under `merlin/python/merlin` is not in the population at all,
  since there is nothing to shadow. That restriction is part of the population definition and is
  recorded with the result rather than left implicit.

**What a low number would mean.** Most `fix(` commits in this repo are harness, packaging, plotting and
experiment-driver work, which no compiler-verification layer can see and none claims to. A low rate is
therefore the expected result and the honest one; the number worth reporting is the rate WITHIN the
population, alongside the population's size and how it was defined.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Paths whose content a verification layer can actually observe. A commit touching only files outside
#: these is not in the population -- not "missed", not in the denominator.
OBSERVED_ROOTS = (
    "merlin/python/merlin/runtime/",
    "merlin/python/merlin/xdsl_dialects/",
    "merlin/python/merlin/targetgen/capsule_golden.py",
    "merlin/python/merlin/targetgen/corpus_spec.py",
    "merlin/python/merlin/verify/",
)

#: The package root a shadow copy replaces.
PACKAGE = "merlin/python/merlin"

#: The commit that first added anything under `merlin/python/merlin/verify/`. A sampled fix that
#: POSTDATES it may have shipped with a regression test that is now one of the layers, so counting it
#: measures the test that came with the fix rather than the layer's ability to catch an unseen defect.
#: Such commits stay in the sample -- removing them after seeing the outcome is exactly the move this
#: module exists to avoid -- but they are flagged, and the record reports the historical-only rate
#: separately so a reader can use whichever denominator the claim needs.
LAYERS_LANDED = "836cb6f354052320e3558b84e6e01a90f51dd649"


@dataclass
class Replayed:
    sha: str
    subject: str
    files: list[str]
    outcome: str                                  # detected | missed | unreplayable | disqualified
    layers_red: list[str] = field(default_factory=list)
    note: str = ""
    #: False when the fix landed after the layers did -- see LAYERS_LANDED.
    predates_layers: bool = True


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True,
                          check=False).stdout


def _ancestors_of_layers(repo: Path) -> set[str]:
    """Every commit reachable from the commit that introduced the layers.

    Ancestry, not dates: a rebased or cherry-picked commit can carry an author date older than work it
    actually followed, and this question is about what the layers could have been written to catch.
    """
    out = _git("rev-list", LAYERS_LANDED, cwd=repo)
    return {line.strip() for line in out.splitlines() if line.strip()}


def population(repo: Path, ref: str = "HEAD") -> list[tuple[str, str, list[str]]]:
    """Every `fix(` commit touching an observed path, newest first, with its observed files.

    Deterministic given ``ref``: git log order is a total order and the filter is a prefix test on the
    commit's own file list. ``ref`` is not cosmetic -- the population GROWS as work lands, and since
    the sample is drawn by shuffling the population, a run a few commits later draws a different 25.
    The record stores the resolved sha so a rerun reproduces exactly the same sample rather than
    approximately the same one.
    """
    out = _git("log", ref, "--format=%H%x00%s", "--name-only", "--grep=^fix(", "--", *OBSERVED_ROOTS,
               cwd=repo)
    entries: list[tuple[str, str, list[str]]] = []
    sha = subject = ""
    files: list[str] = []
    for line in out.splitlines():
        if "\x00" in line:
            if sha:
                entries.append((sha, subject, files))
            sha, subject = line.split("\x00", 1)
            files = []
        elif line.strip():
            if any(line.startswith(r) for r in OBSERVED_ROOTS):
                files.append(line.strip())
    if sha:
        entries.append((sha, subject, files))
    return [e for e in entries if e[2]]


def draw(pool: list, n: int, seed: int) -> list:
    """A seeded shuffle, taken from the front. The seed goes in the artifact.

    Shuffle-then-take rather than `random.sample` so that raising `n` EXTENDS the previous sample
    instead of replacing it -- a later, larger run is then a superset, and cannot be a quiet reroll
    after seeing an unwelcome result.
    """
    ordered = list(pool)
    random.Random(seed).shuffle(ordered)
    return ordered[:n]


def _shadow(repo: Path, sha: str, files: list[str], dest: Path) -> list[str]:
    """Copy the package to ``dest`` and write the PARENT's version of each file over it.

    Returns the files that could not be restored (deleted, renamed, or absent at the parent). A
    non-empty return means the commit is `unreplayable`; the caller must not treat it as a miss.
    """
    pkg = dest / "merlin"
    if pkg.exists():
        shutil.rmtree(pkg)
    # `_data` is EXCLUDED, and that is deliberate rather than an optimisation. It is the packaging
    # mirror of `merlin/contract`, read only when merlin is installed as a wheel; the shadow runs from
    # source with MERLIN_REPO_ROOT pinned at the real checkout, so nothing in it is reachable. It is
    # also the one volatile part of the tree -- a mirror of symlinks that another session can be
    # rebuilding while this copy walks it, which took two whole replay runs down with a bare
    # "No such file or directory" on a path that exists. symlinks=True copies links AS links for the
    # rest, so the shadow stays a faithful copy rather than a materialized one, and stays cheap: this
    # runs once per sampled commit.
    shutil.copytree(repo / PACKAGE, pkg, symlinks=True, ignore_dangling_symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_data"))
    failed: list[str] = []
    for rel in files:
        proc = subprocess.run(("git", "show", f"{sha}^:{rel}"), cwd=repo,
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            failed.append(rel)
            continue
        target = dest / "merlin" / rel[len(PACKAGE) + 1:]
        if not target.parent.is_dir():
            failed.append(rel)
            continue
        target.write_text(proc.stdout, encoding="utf-8")
    return failed


#: The layers, each a module invocation returning non-zero when it REJECTS. Kept to checks that run in
#: seconds: a replay over a sample has to be affordable or it will be run once and never repeated.
#:
#: THE INSTRUMENT MUST BE THE LAYERS THAT EXIST, not a convenient subset. A first run wired only the
#: three pytest files and reported 0 detections over 20 historical fixes -- an honest number for that
#: instrument and a misleading one for the work, because it left out the static layer (lit/FileCheck
#: over the passes) and the numeric oracle, which are the two checks most likely to see a lowering
#: defect. Both are here now. Adding a layer can only find MORE, so a rate measured with a smaller
#: instrument is a lower bound on this one; both runs are kept rather than the first being replaced.
_PYTEST = ("-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider")

LAYERS: dict[str, tuple[str, ...]] = {
    "engines-agree": _PYTEST + ("merlin/tests/ir/test_readout_dtype_divergence.py",),
    "cb-semantics": _PYTEST + ("merlin/tests/ir/test_cb_semantics.py",),
    "compilation-validation": _PYTEST + ("merlin/tests/ir/test_compilation_validation.py",),
    # The static layer: one pass, one module, assert what it did. This is where a lowering defect
    # shows, and it was missing from the first run.
    "lit-pass-tests": ("-m", "merlin.verify.replay_layers", "lit"),
    # The numeric oracle over the real corpus -- the pre-existing dynamic check the formal layers sit
    # beside. Included so a detection can be attributed: a defect BOTH catch is not evidence for the
    # new layer, and only this comparison can tell the two apart.
    "numeric-golden": ("-m", "merlin.verify.replay_layers", "oracle"),
}


def _run_layers(repo: Path, pythonpath: str, timeout: int) -> dict[str, str]:
    import os

    # The shadow replaces the CODE, never the data. `repo_root()` resolves from the package's own
    # location, so inside a shadow it points at the temp directory -- where there is no capsule corpus,
    # no lit suite and no llvm-build. Every layer would then find nothing to check and report a clean
    # pass, which is the "check that could not run reporting success" shape this repo has been bitten
    # by repeatedly. Pinning MERLIN_REPO_ROOT to the real checkout is what keeps the layers pointed at
    # their inputs while the code under them is the historical one.
    env = dict(os.environ, PYTHONPATH=pythonpath, MERLIN_REPO_ROOT=str(repo))
    verdicts: dict[str, str] = {}
    for name, argv in LAYERS.items():
        try:
            proc = subprocess.run((str(repo / ".venv" / "bin" / "python"),) + argv, cwd=repo,
                                  capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            verdicts[name] = "timeout"
            continue
        # pytest's exit codes carry the distinction this measurement depends on. 0 is a pass and 1 is a
        # genuine test FAILURE -- the layer looked at the defect and rejected it. Anything higher is the
        # run not happening: 2 is an interrupt, which is what a collection error reports, 3 an internal
        # error, 4 a usage error, 5 nothing collected. A shadowed package pinned at an old parent can
        # easily fail to IMPORT (a sibling module has moved since), and scoring that as a rejection
        # would credit the layer with catching a defect it never saw. It is `error`, and `error` is not
        # a detection.
        verdicts[name] = {0: "green", 1: "red"}.get(proc.returncode, "error")
    return verdicts


def replay(repo: Path, n: int = 20, seed: int = 20260905, timeout: int = 300,
           ref: str = "HEAD") -> dict:
    """Draw a sample, replay each defect, and return the record. Never raises on one bad commit."""
    resolved = _git("rev-parse", ref, cwd=repo).strip()
    pool = population(repo, resolved or ref)
    sample = draw(pool, n, seed)

    baseline = _run_layers(repo, str(repo / "merlin" / "python"), timeout)
    usable = [k for k, v in baseline.items() if v == "green"]
    historical = _ancestors_of_layers(repo)

    results: list[Replayed] = []
    with tempfile.TemporaryDirectory(prefix="merlin-replay-") as tmp:
        dest = Path(tmp)
        for sha, subject, files in sample:
            failed = _shadow(repo, sha, files, dest)
            old = sha in historical
            if failed:
                results.append(Replayed(sha[:8], subject, files, "unreplayable",
                                        note=f"parent version unavailable for {failed}",
                                        predates_layers=old))
                continue
            verdicts = _run_layers(repo, str(dest), timeout)
            red = [k for k in usable if verdicts.get(k) == "red"]
            broken = [k for k in usable if verdicts.get(k) in ("error", "timeout")]
            if broken and not red:
                # Every usable layer failed to RUN against this shadow, so nothing was measured. Calling
                # it a miss would be as wrong as calling it a detection.
                results.append(Replayed(sha[:8], subject, files, "unreplayable",
                                        note=f"the shadowed package did not run: {broken}",
                                        predates_layers=old))
                continue
            if not usable:
                results.append(Replayed(sha[:8], subject, files, "disqualified",
                                        note="no layer was green on the real tree",
                                        predates_layers=old))
            else:
                results.append(Replayed(sha[:8], subject, files,
                                        "detected" if red else "missed", red,
                                        predates_layers=old))

    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    replayable = counts.get("detected", 0) + counts.get("missed", 0)
    hist = [r for r in results if r.predates_layers and r.outcome in ("detected", "missed")]
    hist_detected = sum(1 for r in hist if r.outcome == "detected")
    return {
        "schema": "verify_historical_replay/v1",
        "population_size": len(pool),
        "population_definition": {"grep": "^fix(", "observed_roots": list(OBSERVED_ROOTS),
                                  "ref": resolved or ref},
        "sample_size": len(sample),
        "seed": seed,
        "baseline": baseline,
        "layers_usable": usable,
        "counts": counts,
        "detected_of_replayable": f"{counts.get('detected', 0)}/{replayable}" if replayable else "0/0",
        # The number to cite. A fix that postdates the layers may have shipped with the very test that
        # now does the catching, so the all-commits rate can only overstate.
        "detected_of_replayable_historical": f"{hist_detected}/{len(hist)}" if hist else "0/0",
        "layers_landed": LAYERS_LANDED[:8],
        "results": [asdict(r) for r in results],
    }


def render(rec: dict) -> str:
    unusable = {k: v for k, v in rec["baseline"].items() if v != "green"}
    lines = [
        f"population {rec['population_size']} fix( commits touching an observed path, "
        f"at {rec['population_definition']['ref'][:8]}",
        f"sample     {rec['sample_size']} drawn with seed {rec['seed']} (shuffle-then-take)",
        f"baseline   {rec['baseline']}",
        f"detected   {rec['detected_of_replayable']} of the REPLAYABLE commits",
        f"           {rec['detected_of_replayable_historical']} counting only fixes that PREDATE the "
        f"layers ({rec['layers_landed']}) -- the citable rate",
        "",
    ]
    if unusable:
        # Loud, because a layer that did not run silently NARROWS the instrument, and the rate then
        # describes a smaller thing than the sentence around it claims. Twice now a layer was wired to
        # a module name that did not exist and the run reported a number for the remaining three.
        lines.insert(3, f"WARNING   {len(unusable)} of {len(rec['baseline'])} layers were not usable "
                        f"and could detect nothing: {unusable}")
    for r in rec["results"]:
        mark = {"detected": "CAUGHT", "missed": "missed", "unreplayable": "n/a  ",
                "disqualified": "dq   "}[r["outcome"]]
        layers = (" <- " + ", ".join(r["layers_red"])) if r["layers_red"] else ""
        age = "" if r["predates_layers"] else "  [postdates the layers]"
        lines.append(f"  {mark} {r['sha']} {r['subject'][:64]}{layers}{age}")
    for outcome, label in (("unreplayable", "unreplayable"), ("disqualified", "disqualified")):
        n = rec["counts"].get(outcome, 0)
        if n:
            lines.append(f"\n{n} {label}; reported, never folded into 'missed'")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    from merlin.common.paths import repo_root

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20, help="sample size")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--timeout", type=int, default=300, help="per-layer seconds")
    ap.add_argument("--ref", default="HEAD",
                    help="commit the population is taken from; pin it to reproduce a sample exactly")
    ap.add_argument("--write", action="store_true", help="write the record as a versioned product")
    a = ap.parse_args(argv)

    rec = replay(repo_root(), n=a.n, seed=a.seed, timeout=a.timeout, ref=a.ref)
    print(render(rec))
    if a.write:
        from merlin.common.artifacts import new_product

        prod = new_product("verification", version=1, sources=[
            f"{rec['population_size']} fix( commits touching {len(OBSERVED_ROOTS)} observed paths",
            f"sample of {rec['sample_size']}, seed {rec['seed']}",
        ], notes=("Historical replay: each sampled fix's PARENT files are shadowed over the package and "
                  "the layers re-run. Unreplayable commits are reported, never dropped."))
        out = prod.add_artifact("historical_replay.json")
        out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        prod.write_manifest()
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
