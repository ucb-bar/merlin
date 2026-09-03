"""Where THIS experiment is allowed to write, and the frozen things it may only read.

This is a PARALLEL track. The agentic performance experiment (`gemmini_perf_bench`, the
`performance_contract` program) and the certified gemmini backend are live work owned by other
sessions in this shared checkout, so the rule for everything here is:

    READ the frozen artifacts. COPY anything that has to change. WRITE only under paths this
    experiment owns exclusively.

Stated once, in one module, because four scripts had each spelled the paths themselves and one of
them already leaked: passing ``runs_root=out/runs`` made ``certify`` create ``out/runs/runs/`` and put
this track's runs in the SHARED ``gemmini-contract`` suite dir, where a run-id collision with another
session was only avoided by luck.

WHAT IS READ-ONLY HERE, and must stay byte-identical:
  * ``out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0/`` -- the certified champion package. Its
    ``SHA256SUMS`` is the check; :func:`assert_frozen_intact` runs it.
  * the prebuilt GSIM emulator, which another session's campaign is using concurrently.
  * every capsule under ``merlin/contract/capsules/`` -- read as inputs, never written.

WHAT THIS TRACK OWNS:
  * the fork ``gemmini_xdsl_recipe_v0`` -- a copy, so the original is never edited;
  * ``out/runs/gemmini/recipe-select/`` -- its own runs root, via the sanctioned
    ``benchharness.runs_root(target, suite)`` helper rather than a hand-built path;
  * ``out/artifacts/recipe-select/<target>/`` products, ``out/artifacts/cache/recipe_select_*``
    caches, and ``out/build/recipe_select_*`` scratch.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def repo() -> Path:
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "merlin" / "python").is_dir():
            return cand
    raise SystemExit("could not locate repo root (no merlin/python above this file)")


REPO = repo()
if str(REPO / "merlin" / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness import runs_root as _runs_root      # noqa: E402
from merlin.common.paths import artifacts_dir                # noqa: E402

TARGET = "gemmini"
SUITE = "recipe-select"

#: READ-ONLY. The certified champion. Never opened for writing by anything in this experiment.
FROZEN = artifacts_dir() / f"targets/{TARGET}/gemmini_xdsl_rtl_v0"
#: OWNED. A copy of the above with a recipe surface added.
FORK = artifacts_dir() / f"targets/{TARGET}/gemmini_xdsl_recipe_v0"

#: OWNED. This track's runs root. `certify` appends `runs/<suite>/<run_id>` beneath it, so every run
#: this experiment makes is under one directory nobody else writes to.
RUNS = _runs_root(TARGET, SUITE)

GSIM_EMU = Path("/scratch/agustin/tmp/gsim_cert_serialclk_v1/"
                "emu_gemmini_gsim_serialclk_v1_filtered_final")
GSIM_SHA = "fb356ede610fb5f5ecbe2edb61dfd9a5a196293408a5ea02f34f919b5e39916b"
GSIM_CONFIG = "chipyard.harness.TestHarness.GemminiGsimSerialClkConfig"
GSIM_MAXCYCLES = "100000000"

#: Why a cycle number from this track may not be quoted as a Verilator number. MEASURED, not assumed.
ENGINE_NOTE = (
    "cycles describe " + GSIM_CONFIG + ". Its accelerator is identical to stock GemminiRocketConfig "
    "(the two elaborations differ only by ClockSourceAtFreqMHz x2 and one IO cell; Mesh, "
    "MeshWithDelays, PE, Tile and AccumulatorMem module sets match), but the two ENGINES were "
    "measured to disagree -- 302 vs 303 on A2 and 604 vs 610 on PK03_k128 -- so these are not "
    "Verilator-equivalent numbers and must not be compared against the frozen package's recorded "
    "rtl_verilator cycles."
)


def assert_frozen_intact() -> None:
    """Fail before doing work if the certified package's sources are no longer its pinned bytes.

    Cheap, and it protects the one thing this track must not break: if the frozen backend drifts, the
    fork's equivalence gate is comparing against something that is no longer certified.
    """
    sums = FROZEN / "SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"{sums} is missing: cannot establish that the frozen package is intact")
    recorded: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        recorded[rel.strip().lstrip("./")] = digest.strip()
    drift = []
    for rel in ("mlir_oot/lowering/isa.py", "mlir_oot/ir_ingest.py",
                "mlir_oot/gemmini_opt.py", "mlir_oot/transforms.py"):
        want = recorded.get(rel)
        p = FROZEN / rel
        if want is None or not p.exists():
            drift.append(f"{rel}: not declared in SHA256SUMS or absent")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            drift.append(f"{rel}: {got[:12]} != pinned {want[:12]}")
    if drift:
        raise SystemExit("the FROZEN package is not its pinned bytes; refusing to run:\n  "
                         + "\n  ".join(drift))


def gsim_env() -> dict[str, str]:
    """The env that selects the certified GSIM emulator, refusing an emu whose bytes are unconfirmed."""
    if not GSIM_EMU.exists():
        raise SystemExit(f"gsim emulator absent at {GSIM_EMU}")
    got = hashlib.sha256(GSIM_EMU.read_bytes()).hexdigest()
    if got != GSIM_SHA:
        raise SystemExit(f"gsim emu digest {got} != certified {GSIM_SHA}; refusing to cite it")
    return {"MERLIN_GEMMINI_GSIM_EMU": str(GSIM_EMU),
            "MERLIN_GEMMINI_GSIM_MAXCYCLES": GSIM_MAXCYCLES}


# ---------------------------------------------------------------------------------------------
# Per-run package minting.
#
# v0 ran every arm against ONE mutable fork directory. That is fine for a single run and wrong for a
# programme: an edit made while a run is in flight silently changes what that run is measuring, and
# two runs cannot proceed at once. So compiler work happens in the WORKING copy below, and a run
# consumes only a MINTED package -- content-addressed over the sources that decide what is emitted,
# frozen 0444 with its own SHA256SUMS, and never written again.
#
# The content address is the point. Two runs that minted the same bytes share one directory (so a
# re-run is free and provably identical); a run whose sources differ by one character gets its own,
# and every candidate row's ``package_fingerprint`` resolves back to exactly the compiler that
# produced it.
# ---------------------------------------------------------------------------------------------

#: The WORKING copy. Compiler work is done here. A run never measures this directory directly.
WORK = FORK

#: Files whose bytes decide what the package emits. Digesting the whole tree would make the address
#: depend on __pycache__ and build leftovers, which change without changing the compiler.
_SOURCE_ROOT = "mlir_oot"
_SKIP_DIRS = {"__pycache__", "build", ".git", ".pytest_cache"}


def _source_files(pkg: Path) -> list[Path]:
    """Every source file under ``mlir_oot/``, sorted, with generated/cache trees excluded."""
    root = pkg / _SOURCE_ROOT
    if not root.is_dir():
        raise SystemExit(f"{pkg} has no {_SOURCE_ROOT}/ -- not an OOT backend package")
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if _SKIP_DIRS.intersection(part for part in p.relative_to(pkg).parts):
            continue
        out.append(p)
    return out


def source_digest(pkg: Path) -> str:
    """A content address over the package's emitting sources: sha256 of (relpath, bytes) pairs.

    Path-sensitive on purpose -- a file that moves changes what the tool imports, so it must change
    the address even when the bytes are the same.
    """
    h = hashlib.sha256()
    for p in _source_files(pkg):
        rel = p.relative_to(pkg).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _write_sha256sums(pkg: Path) -> None:
    lines = []
    for p in sorted(pkg.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS":
            continue
        rel = p.relative_to(pkg)
        if _SKIP_DIRS.intersection(rel.parts):
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{rel.as_posix()}")
    (pkg / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze(pkg: Path) -> None:
    """0444 on files, 0555 on directories. A run that tries to write its own compiler gets EACCES."""
    for p in sorted(pkg.rglob("*"), reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    pkg.chmod(0o555)


def _thaw(pkg: Path) -> None:
    """Make a minted package writable again -- only ever used to REPLACE a half-written mint."""
    for p in sorted(pkg.rglob("*"), reverse=True):
        p.chmod(0o755 if p.is_dir() else 0o644)
    pkg.chmod(0o755)


def mint_fork(work: Path | None = None, *, label: str = "recipe") -> Path:
    """Freeze the working fork into its own content-addressed package and return it.

    Idempotent: if a package with this content address already exists and still verifies, it is
    returned untouched -- that is what makes a re-run share the identical compiler rather than mint a
    near-copy of it.
    """
    import shutil
    import time

    work = Path(work or WORK)
    digest = source_digest(work)
    pkg_id = f"gemmini_xdsl_{label}_{digest[:12]}"
    dest = artifacts_dir() / f"targets/{TARGET}/{pkg_id}"

    if dest.exists():
        try:
            assert_package_frozen(dest)
            return dest
        except SystemExit:
            # A previous mint died partway. Replace it rather than measure against a torn package.
            _thaw(dest)
            shutil.rmtree(dest)

    staging = dest.with_name(dest.name + ".minting")
    if staging.exists():
        _thaw(staging)
        shutil.rmtree(staging)
    shutil.copytree(work, staging,
                    ignore=shutil.ignore_patterns(*_SKIP_DIRS, "SHA256SUMS"))

    # The manifest must describe THIS package, not the parent it was copied from. The working fork
    # still carries the champion's `package_id` and its `publication:` block; leaving those in place
    # would let a fork's numbers be read as the champion's certification.
    import yaml  # noqa: PLC0415

    man_path = staging / "manifest.yaml"
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    parent_fp = (man.get("publication") or {}).get("fingerprint") or man.get("fingerprint")
    man.pop("publication", None)          # not a champion; it is an experiment fork
    man["package_id"] = pkg_id
    man["lineage"] = {
        "parent_package_id": "gemmini_xdsl_rtl_v0",
        "parent_fingerprint": parent_fp,
        "minted_by": "merlin/experiments/agent_recipe_select_v0",
        "minted_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "source_digest": digest,
        "note": ("experiment fork of the certified champion; its cycles describe this package "
                 "only and are not the champion's certified numbers"),
    }
    man_path.write_text(yaml.safe_dump(man, sort_keys=True), encoding="utf-8")

    _write_sha256sums(staging)
    _freeze(staging)
    staging.rename(dest)
    return dest


def assert_package_frozen(pkg: Path) -> str:
    """Verify every file a package's SHA256SUMS declares. Returns the source digest.

    Called before the first candidate of a run and after the last: a package that changed underneath
    a run invalidates every number the run produced, and the whole point of minting is that this can
    be checked rather than assumed.
    """
    pkg = Path(pkg)
    sums = pkg / "SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"{pkg} has no SHA256SUMS: it was never minted, so nothing pins its bytes")
    drift, missing = [], []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, _, rel = line.partition("  ")
        p = pkg / rel.strip().lstrip("./")
        if not p.exists():
            missing.append(rel.strip())
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != want.strip():
            drift.append(rel.strip())
    if drift or missing:
        raise SystemExit(
            f"minted package {pkg.name} is not its pinned bytes; refusing to cite it:\n"
            + "".join(f"  changed: {r}\n" for r in drift[:10])
            + "".join(f"  missing: {r}\n" for r in missing[:10]))
    return source_digest(pkg)


def snapshot_scripts(run_dir: Path) -> Path:
    """Copy the scripts that produced a run into the run, so it stays reproducible after the tree moves.

    A digest manifest alone would say *that* the scripts differ from today's; the copy says *how*.
    """
    import shutil

    src = Path(__file__).resolve().parent
    dest = Path(run_dir) / "scripts_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in sorted(src.glob("*.py")):
        shutil.copy2(p, dest / p.name)
        manifest.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (dest / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------------------------
# MEASURED HAZARD, 2026-09-03: ``.venv/bin/python`` imports ``merlin`` from a DIFFERENT checkout
# (``oscar-merlin-arm4-v4-functional-final``), because that tree is the one installed into the shared
# venv. The consequence is not an import error -- it is a plausible wrong answer: ``out_dir()``
# resolves to the other checkout, so ``recaptures_dir()`` finds no bundles ("skip: no bundle with a
# model.mlir") and ``new_product`` writes this experiment's artifacts into someone else's tree. One
# census run did exactly that before this guard existed.
#
# Scripts in this directory are safe in-process because they ``sys.path.insert(0, REPO/merlin/python)``
# before importing. SUBPROCESSES ARE NOT: a child launched with ``sys.executable`` inherits the venv's
# view, not the parent's ``sys.path``. Every subprocess that imports merlin must be given ``py_env()``.
# ---------------------------------------------------------------------------------------------

MERLIN_PY = REPO / "merlin" / "python"


def assert_right_merlin() -> None:
    """Fail loudly if the imported ``merlin`` is some other checkout's."""
    import merlin as _m                                                   # noqa: PLC0415

    got = Path(_m.__file__).resolve()
    if REPO not in got.parents:
        raise SystemExit(
            f"imported merlin from {got}\n"
            f"  but this experiment lives in {REPO}\n"
            "  -> set PYTHONPATH={MERLIN_PY} (or use _track.py_env()); otherwise out_dir(), "
            "recaptures_dir() and new_product() all resolve into the other checkout")


def py_env(extra: "dict[str, str] | None" = None) -> dict:
    """Environment for a subprocess that imports merlin: this checkout's package wins the path."""
    import os                                                             # noqa: PLC0415

    env = dict(os.environ)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(MERLIN_PY) + (os.pathsep + prior if prior else "")
    if extra:
        env.update(extra)
    return env
