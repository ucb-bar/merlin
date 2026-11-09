"""Replay historical fixes past the verification layers, on a BLIND sample.

This optional analysis workflow requires ``merlin-analysis[replay]`` and a source checkout containing
the pinned history, tests and corpus. Importing it or showing ``--help`` does not run a measurement.

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

The shadow is a composite copy of core and every optional source owner. Namespace extensions must
come from that same snapshot, never an installed sibling checkout. A trusted live bootstrap qualifies
executed checks separately from missing dependencies, skipped checks and startup/runtime failures.
Records name this qualification policy: older exit-code-only measurements are not equivalent.
Lit remains in the declared instrument but is currently unavailable for qualified replay: its child
commands do not yet provide exception/import receipts. Standalone lit remains accessible. New replay
records and their rendered summaries state this limitation rather than claiming five usable layers.

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

# Population and historical records remain unchanged; this identifies the repaired instrument.
QUALIFICATION_POLICY = "executed_checks_v2"
QUALIFICATION_LIMITS = {
    "lit-pass-tests": "unavailable: lit child commands lack qualified exception/import receipts",
}

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
    outcome: str  # detected | missed | unreplayable | disqualified
    layers_red: list[str] = field(default_factory=list)
    note: str = ""
    #: False when the fix landed after the layers did -- see LAYERS_LANDED.
    predates_layers: bool = True


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True, check=False).stdout


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
    out = _git("log", ref, "--format=%H%x00%s", "--name-only", "--grep=^fix(", "--", *OBSERVED_ROOTS, cwd=repo)
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


def _shadow(repo: Path, sha: str, files: list[str], dest: Path, *, baseline: Path | None = None) -> list[str]:
    """Copy the package to ``dest`` and write the PARENT's version of each file over it.

    Returns the files that could not be restored (deleted, renamed, or absent at the parent). A
    non-empty return means the commit is `unreplayable`; the caller must not treat it as a miss.
    """
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("replay shadow destination must be empty")
    dest.mkdir(parents=True, exist_ok=True)
    # `_data` is EXCLUDED, and that is deliberate rather than an optimisation. It is the packaging
    # mirror of `merlin/contract`, read only when merlin is installed as a wheel; the shadow runs from
    # source with MERLIN_REPO_ROOT pinned at the real checkout, so nothing in it is reachable. It is
    # also the one volatile part of the tree -- a mirror of symlinks that another session can be
    # rebuilding while this copy walks it, which took two whole replay runs down with a bare
    # "No such file or directory" on a path that exists. symlinks=True copies links AS links for the
    # rest, so the shadow stays a faithful copy rather than a materialized one, and stays cheap: this
    # runs once per sampled commit.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "_data", "*.egg-info", "AGENT.md", "CLAUDE.md")
    if baseline is not None:
        shutil.copytree(baseline, dest, dirs_exist_ok=True, symlinks=True, ignore=ignore)
    else:
        core = repo / "src"
        if not (core / "merlin").is_dir():
            core = repo / "merlin/python"
        owners = [core, *sorted((repo / "packages").glob("*/src"))]
        copied: set[Path] = set()
        for owner in owners:
            # Directories may be shared namespaces, but files must have exactly one owner.
            for source in owner.rglob("*"):
                rel = source.relative_to(owner)
                if any(
                    part in {"_data", "__pycache__", "AGENT.md", "CLAUDE.md"} or part.endswith((".pyc", ".egg-info"))
                    for part in rel.parts
                ):
                    continue
                if source.is_file() or source.is_symlink():
                    if rel in copied:
                        raise ValueError(f"duplicate replay source ownership: {rel}")
                    copied.add(rel)
            shutil.copytree(owner, dest, dirs_exist_ok=True, symlinks=True, ignore=ignore)
    failed: list[str] = []
    for rel in files:
        if not rel.startswith(PACKAGE + "/") or ".." in Path(rel).parts:
            failed.append(rel)
            continue
        proc = subprocess.run(("git", "show", f"{sha}^:{rel}"), cwd=repo, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            failed.append(rel)
            continue
        target = dest / "merlin" / rel[len(PACKAGE) + 1 :]
        if not target.parent.is_dir() or not target.resolve().is_relative_to(dest.resolve()):
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
    import hashlib
    import os
    import sys

    from merlin.common.paths import module_source_path

    # The shadow replaces the CODE, never the data. `repo_root()` resolves from the package's own
    # location, so inside a shadow it points at the temp directory -- where there is no capsule corpus,
    # no lit suite and no llvm-build. Every layer would then find nothing to check and report a clean
    # pass, which is the "check that could not run reporting success" shape this repo has been bitten
    # by repeatedly. Pinning MERLIN_REPO_ROOT to the real checkout is what keeps the layers pointed at
    # their inputs while the code under them is the historical one.
    env = dict(
        os.environ,
        PYTHONPATH=pythonpath,
        MERLIN_REPO_ROOT=str(repo),
        MERLIN_REPLAY_PYTHONPATH=pythonpath,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    verdicts: dict[str, str] = {}
    root = Path(pythonpath).absolute()
    # The live instrument captures the final composite, including each historical overlay.
    # The child checks these pins; it does not bless whatever bytes happen to be present later.
    try:
        helper = module_source_path("merlin.common.frozen_imports")
        helper_digest = hashlib.sha256(helper.read_bytes()).hexdigest()
        sources = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }
    except OSError as exc:
        print(f"replay source inventory unavailable: {exc}", file=sys.stderr)
        return dict.fromkeys(LAYERS, "error")
    context = {
        "snapshot_root": str(root),
        "import_roots": [str(root)],
        "sources": sources,
        "helper": str(helper),
        "helper_sha256": helper_digest,
        "python_version": list(sys.version_info[:2]),
        # Ordinary dependency directories only: never execute sitecustomize or editable .pth.
        "dependencies": [entry for entry in sys.path if entry and Path(entry).is_dir()]
        + [str(repo / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")],
    }
    for name, argv in LAYERS.items():
        try:
            with tempfile.TemporaryDirectory(prefix="merlin-replay-layer-") as work:
                receipt = Path(work) / "qualification.json"
                context_path = Path(work) / "imports.json"
                context_bytes = json.dumps(context).encode()
                context_path.write_bytes(context_bytes)
                env["MERLIN_REPLAY_IMPORT_CONTEXT"] = str(context_path)
                env["MERLIN_REPLAY_IMPORT_CONTEXT_SHA256"] = hashlib.sha256(context_bytes).hexdigest()
                layers = Path(__file__).with_name("replay_layers.py")
                env["MERLIN_REPLAY_BOOTSTRAP_SHA256"] = hashlib.sha256(layers.read_bytes()).hexdigest()
                bootstrap = """
import hashlib,os,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise RuntimeError('replay qualification bootstrap is not a regular file')
source=path.read_bytes()
if hashlib.sha256(source).hexdigest()!=os.environ['MERLIN_REPLAY_BOOTSTRAP_SHA256']:
    raise RuntimeError('replay qualification bootstrap hash mismatch')
namespace={'__file__':str(path),'__name__':'_replay_qualification'}
exec(compile(source,str(path),'exec'),namespace)
raise SystemExit(namespace['qualify_layer'](sys.argv[2:]))
"""
                proc = subprocess.run(
                    (
                        str(repo / ".venv/bin/python"),
                        "-I",
                        "-S",
                        "-c",
                        bootstrap,
                        str(layers),
                        str(receipt),
                        QUALIFICATION_POLICY,
                        *argv,
                    ),
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout,
                )
                record = json.loads(receipt.read_text()) if receipt.is_file() else {}
        except subprocess.TimeoutExpired:
            verdicts[name] = "timeout"
            continue
        except (OSError, ValueError):
            verdicts[name] = "error"
            continue
        # An exit 1 alone can be a missing module or interpreter crash. Only the live bootstrap's
        # completed receipt can qualify it as a genuine rejection by an executed check.
        status = record.get("status") if isinstance(record, dict) else None
        verdicts[name] = (
            status
            if (
                isinstance(record, dict)
                and record.get("qualification_policy") == QUALIFICATION_POLICY
                and status in ("green", "red")
                and proc.returncode == {"green": 0, "red": 1}[status]
            )
            else "error"
        )
        if verdicts[name] == "error" and isinstance(record, dict) and isinstance(record.get("detail"), str):
            print(f"{name}: {record['detail']}", file=sys.stderr)
    return verdicts


def replay(repo: Path, n: int = 20, seed: int = 20260905, timeout: int = 300, ref: str = "HEAD") -> dict:
    """Draw a sample, replay each defect, and return the record. Never raises on one bad commit."""
    resolved = _git("rev-parse", ref, cwd=repo).strip()
    pool = population(repo, resolved or ref)
    sample = draw(pool, n, seed)

    historical = _ancestors_of_layers(repo)

    results: list[Replayed] = []
    with tempfile.TemporaryDirectory(prefix="merlin-replay-") as tmp:
        frozen = Path(tmp) / "baseline"
        _shadow(repo, "", [], frozen)
        baseline = _run_layers(repo, str(frozen), timeout)
        usable = [k for k, v in baseline.items() if v == "green"]
        for index, (sha, subject, files) in enumerate(sample):
            dest = Path(tmp) / f"shadow-{index}"
            failed = _shadow(repo, sha, files, dest, baseline=frozen)
            old = sha in historical
            if failed:
                results.append(
                    Replayed(
                        sha[:8],
                        subject,
                        files,
                        "unreplayable",
                        note=f"parent version unavailable for {failed}",
                        predates_layers=old,
                    )
                )
                continue
            verdicts = _run_layers(repo, str(dest), timeout)
            red = [k for k in usable if verdicts.get(k) == "red"]
            broken = [k for k in usable if verdicts.get(k) in ("error", "timeout")]
            if broken and not red:
                # Every usable layer failed to RUN against this shadow, so nothing was measured. Calling
                # it a miss would be as wrong as calling it a detection.
                results.append(
                    Replayed(
                        sha[:8],
                        subject,
                        files,
                        "unreplayable",
                        note=f"the shadowed package did not run: {broken}",
                        predates_layers=old,
                    )
                )
                continue
            if not usable:
                results.append(
                    Replayed(
                        sha[:8],
                        subject,
                        files,
                        "disqualified",
                        note="no layer was green on the real tree",
                        predates_layers=old,
                    )
                )
            else:
                results.append(
                    Replayed(sha[:8], subject, files, "detected" if red else "missed", red, predates_layers=old)
                )

    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    replayable = counts.get("detected", 0) + counts.get("missed", 0)
    hist = [r for r in results if r.predates_layers and r.outcome in ("detected", "missed")]
    hist_detected = sum(1 for r in hist if r.outcome == "detected")
    return {
        "schema": "verify_historical_replay/v1",
        "qualification_policy": QUALIFICATION_POLICY,
        "qualification_limits": QUALIFICATION_LIMITS,
        "population_size": len(pool),
        "population_definition": {"grep": "^fix(", "observed_roots": list(OBSERVED_ROOTS), "ref": resolved or ref},
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
        f"instrument {rec.get('qualification_policy', 'legacy_exit_codes_v1')}",
        f"detected   {rec['detected_of_replayable']} of the REPLAYABLE commits",
        f"           {rec['detected_of_replayable_historical']} counting only fixes that PREDATE the "
        f"layers ({rec['layers_landed']}) -- the citable rate",
        "",
    ]
    if unusable:
        # Loud, because a layer that did not run silently NARROWS the instrument, and the rate then
        # describes a smaller thing than the sentence around it claims. Twice now a layer was wired to
        # a module name that did not exist and the run reported a number for the remaining three.
        lines.insert(
            3,
            f"WARNING   {len(unusable)} of {len(rec['baseline'])} layers were not usable "
            f"and could detect nothing: {unusable}",
        )
    for layer, reason in rec.get("qualification_limits", {}).items():
        lines.insert(5, f"limit      {layer}: {reason}")
    for r in rec["results"]:
        mark = {"detected": "CAUGHT", "missed": "missed", "unreplayable": "n/a  ", "disqualified": "dq   "}[
            r["outcome"]
        ]
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

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=(
            f"Qualification policy: {QUALIFICATION_POLICY}. lit-pass-tests: {QUALIFICATION_LIMITS['lit-pass-tests']}."
        ),
    )
    ap.add_argument("--n", type=int, default=20, help="sample size")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--timeout", type=int, default=300, help="per-layer seconds")
    ap.add_argument(
        "--ref", default="HEAD", help="commit the population is taken from; pin it to reproduce a sample exactly"
    )
    ap.add_argument("--write", action="store_true", help="write the record as a versioned product")
    a = ap.parse_args(argv)

    rec = replay(repo_root(), n=a.n, seed=a.seed, timeout=a.timeout, ref=a.ref)
    print(render(rec))
    if a.write:
        from merlin.common.artifacts import new_product

        prod = new_product(
            "verification",
            version=1,
            sources=[
                f"{rec['population_size']} fix( commits touching {len(OBSERVED_ROOTS)} observed paths",
                f"sample of {rec['sample_size']}, seed {rec['seed']}",
            ],
            notes=(
                "Historical replay: each sampled fix's PARENT files are shadowed over the package and "
                "the layers re-run. Unreplayable commits are reported, never dropped."
            ),
        )
        out = prod.add_artifact("historical_replay.json")
        out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        prod.write_manifest()
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
