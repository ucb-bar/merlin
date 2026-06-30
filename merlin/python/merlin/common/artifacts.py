"""A tiny artifact abstraction shared by TargetGen generators.

An :class:`Artifact` is a (relative path, text payload) pair that knows how to write
itself under a base directory. Generators return lists of artifacts; the pipeline writes
them. Keeping this explicit makes generation deterministic and easy to test (you can
inspect the artifacts without touching the filesystem).
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .paths import repo_root
from .yaml import dump_yaml


@dataclass(frozen=True)
class Artifact:
    """A single file to emit: ``relpath`` is relative to a generation base directory."""

    relpath: str
    content: str

    def write(self, base: str | Path) -> Path:
        """Write this artifact under ``base``, creating parent directories."""
        out = Path(base) / self.relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.content, encoding="utf-8")
        return out


def yaml_artifact(relpath: str, obj, header: str | None = None) -> Artifact:
    """Build an :class:`Artifact` whose content is deterministic YAML for ``obj``."""
    text = dump_yaml(obj)
    if header:
        text = f"# {header}\n{text}"
    return Artifact(relpath=relpath, content=text)


def write_all(artifacts: list[Artifact], base: str | Path) -> list[Path]:
    """Write every artifact under ``base`` and return the written paths."""
    return [a.write(base) for a in artifacts]


# ===========================================================================
# Three-root output convention (runs/ artifacts/ build/) — see CLAUDE.md
# "Generated-output convention" and .claude/skills/artifact-layout.
#
# This is the SINGLE place that knows how to name and locate generated output.
# Scripts must call start_run() / new_product() / cache_dir() instead of
# hand-building paths under output/, results/, etc. (those roots are retired).
# aet imports are LAZY (inside functions) so importing this module never pulls
# aet — protecting the ~16 modules that use the Artifact class above.
# ===========================================================================


def utc_stamp() -> str:
    """Canonical timestamp token: UTC, ISO-8601 basic, no ':' (fs/shell/url/tar safe, sortable)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_sha7(root: Path | None = None) -> str:
    """Short (7-char) git SHA of HEAD for provenance; 'nogit' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root or repo_root()), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def _unique_dir(base: Path, name: str) -> Path:
    """base/name, appending _<6hex> on collision (same-second + same-sha parallel arms)."""
    cand = base / name
    if not cand.exists():
        return cand
    return base / f"{name}_{os.urandom(3).hex()}"


# ---- experiment runs: runs/<suite>/<run-id>/ (aet-managed) -----------------


@dataclass(frozen=True)
class RunHandle:
    """Everything a caller needs after start_run(): paths, aet logger, artifact store, provenance."""

    spec: object          # aet RunSpec
    paths: object         # aet RunPaths (.run_path/.logs/.metrics/.artifacts_dir/.generated/...)
    logger: object        # aet EvalRunLogger
    store: object         # aet ArtifactStore
    run_id: str
    run_dir: Path
    git_sha: str
    timestamp: str


def start_run(
    *,
    suite: str,
    method: str,
    seed: int = 0,
    target: str | None = None,
    project: str = "merlin",
    run_id: str | None = None,
    project_root: Path | None = None,
    tracking_mode: str = "local",
    extra: dict | None = None,
    make_subdirs: tuple[str, ...] = ("logs", "metrics", "artifacts_dir", "generated", "contracts"),
) -> RunHandle:
    """Begin an aet-managed experiment run under ``runs/<target>/<suite>/<run-id>/``.

    The **target** is the top folder level so everything for a target groups together and the
    inner file names (run_record.json, metrics/summary_metrics.json, ...) are shared across
    targets for easy cross-target diffing. run-id = ``<TS>_<method>_seed<NNN>_<sha7>``
    (timestamp-first → chronological ``ls``). The folder name is a convenience; ``run_record.json``
    (git_sha/timestamp/...) is the source of truth. Returns a :class:`RunHandle`; pair with
    :func:`finish_run`. ``aet runs --suite <target>/<suite>`` lists them (target is embedded in
    the aet suite so discovery keeps working).
    """
    from aet.core.artifact_store import ArtifactStore
    from aet.core.run_paths import RunPaths
    from aet.core.run_spec import RunSpec
    from aet.tracking import EvalRunLogger

    root = Path(project_root) if project_root else repo_root()
    ts = utc_stamp()
    sha = git_sha7(root)
    # Target at folder level: embed it as the leading suite segment (aet lays runs out at
    # runs/<suite>/<run-id>, and suites may be slash-nested — mirrors gemmini-conformance etc.).
    eff_suite = suite
    if target and not (suite == target or suite.startswith(f"{target}/")):
        eff_suite = f"{target}/{suite}"
    if run_id is None:
        run_id = f"{ts}_{method}_seed{seed:03d}_{sha}"
    spec = RunSpec(
        project=project, suite=eff_suite, method=method, seed=seed, run_id=run_id,
        project_root=root, tracking_mode=tracking_mode, target=target,
        repo_initial_commit=sha, extra=dict(extra or {}),
    )
    paths = RunPaths.from_spec(spec, run_id)
    if paths.run_path.exists():  # collision guard (parallel same-second + same-sha arms)
        run_id = f"{run_id}_{os.urandom(3).hex()}"
        spec = dataclasses.replace(spec, run_id=run_id)
        paths = RunPaths.from_spec(spec, run_id)
    for attr in make_subdirs:
        getattr(paths, attr).mkdir(parents=True, exist_ok=True)
    logger = EvalRunLogger.start(
        project=project, suite=eff_suite, target=target or "", method=method,
        seed=seed, run_id=run_id, run_path=paths.run_path, tracking_mode=tracking_mode,
    )
    logger.write_run_record({"git_sha": sha, "timestamp": ts, **(extra or {})})
    store = ArtifactStore(paths.run_path, run_id)
    return RunHandle(spec=spec, paths=paths, logger=logger, store=store,
                     run_id=run_id, run_dir=paths.run_path, git_sha=sha, timestamp=ts)


def finish_run(h: RunHandle, status: str, summary: dict | None = None) -> None:
    """Close an aet run: write summary metrics (if given), finish + close the logger."""
    if summary is not None:
        try:
            h.logger.write_summary_metrics(summary)
        except Exception:
            pass
    h.logger.finish(status=status)
    h.logger.close()


# ---- versioned products: artifacts/<topic>/v<ver>/<leaf>/ ------------------


@dataclass
class ProductDir:
    """A versioned product directory + its manifest. Use :func:`new_product` to create."""

    path: Path
    manifest_path: Path
    run_id: str
    topic: str
    version: int
    git_sha: str
    timestamp: str
    target: str | None = None
    sources: list | None = None
    notes: str = ""
    _artifacts: list | None = None

    def add_artifact(self, relpath: str) -> Path:
        """Register a relative artifact path (for the manifest) and ensure its parent dir."""
        if self._artifacts is None:
            self._artifacts = []
        self._artifacts.append(relpath)
        out = self.path / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def write_manifest(self) -> Path:
        """Write manifest.yaml (mined_knowledge schema: run_id/timestamp/git_sha/version/...)."""
        manifest = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "target": self.target,
            "version": self.version,
            "topic": self.topic,
            "artifacts": sorted(self._artifacts or []),
            "sources": self.sources or [],
            "notes": self.notes,
        }
        self.manifest_path.write_text(dump_yaml(manifest), encoding="utf-8")
        return self.manifest_path


def new_product(
    generator: str,
    *,
    version: int,
    target: str | None = None,
    sources: list | None = None,
    notes: str = "",
    update_latest: bool = True,
) -> ProductDir:
    """Create a versioned product dir and repoint ``latest``.

    Layout (target at folder level so a target's products group together; inner file names
    are shared across targets for cross-target diffing):
      * with target:  ``artifacts/<topic>/<target>/v<ver>/<topic>_<target>_v<ver>_<TS>_<sha7>/``
      * no target:    ``artifacts/<topic>/v<ver>/<topic>_v<ver>_<TS>_<sha7>/``
    The ``latest`` symlink is RELATIVE (bwrap-safe) and repointed atomically.
    """
    ts = utc_stamp()
    sha = git_sha7()
    base = repo_root() / "artifacts" / generator
    if target:
        base = base / target
    vdir = base / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    leaf = f"{generator}_{target}_v{version}_{ts}_{sha}" if target else f"{generator}_v{version}_{ts}_{sha}"
    pdir = _unique_dir(vdir, leaf)
    pdir.mkdir(parents=True, exist_ok=True)
    if update_latest:
        link = vdir / "latest"
        tmp = vdir / f".latest.{os.urandom(3).hex()}"
        os.symlink(pdir.name, tmp)   # RELATIVE target
        os.replace(tmp, link)        # atomic repoint
    return ProductDir(path=pdir, manifest_path=pdir / "manifest.yaml", run_id=pdir.name,
                      topic=generator, version=version, git_sha=sha, timestamp=ts,
                      target=target, sources=sources, notes=notes, _artifacts=[])


# ---- measurements: artifacts/measurements/<substrate>/<model>/<experiment>_v<ver>_<TS>_<sha7>/ ----


@dataclass
class MeasurementDir:
    """A versioned hardware-measurement run dir + manifest. Use :func:`new_measurement`."""

    path: Path
    manifest_path: Path
    run_id: str
    substrate: str
    model: str
    experiment: str
    version: int
    git_sha: str
    timestamp: str
    notes: str = ""
    _artifacts: list | None = None

    def add_artifact(self, relpath: str) -> Path:
        if self._artifacts is None:
            self._artifacts = []
        self._artifacts.append(relpath)
        out = self.path / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def write_manifest(self) -> Path:
        manifest = {
            "run_id": self.run_id, "timestamp": self.timestamp, "git_sha": self.git_sha,
            "substrate": self.substrate, "model": self.model, "experiment": self.experiment,
            "version": self.version, "artifacts": sorted(self._artifacts or []), "notes": self.notes,
        }
        self.manifest_path.write_text(dump_yaml(manifest), encoding="utf-8")
        return self.manifest_path


def new_measurement(
    substrate: str,
    model: str,
    experiment: str,
    *,
    version: int = 1,
    notes: str = "",
    update_latest: bool = True,
) -> MeasurementDir:
    """Create a hardware-measurement run dir under
    ``artifacts/measurements/<substrate>/<model>/<experiment>_v<ver>_<TS>_<sha7>/``.

    **substrate** = the execution environment that produced the numbers, named
    ``<kind>_<design>`` so identical kernels measured on different bitstreams/designs never
    collide — e.g. ``firesim_<bitstream>``, ``baremetal_<verilator-design>``,
    ``zephyr_<design-or-bitstream>``, ``k1_<board>``, ``spike_<config>``. **model** is the
    workload (bitvla/openvla/…). **experiment** is the campaign (cross_framework/e2e/…).
    Keep inner file names identical across substrates/models so cross-substrate diffs are trivial.
    """
    ts = utc_stamp()
    sha = git_sha7()
    base = repo_root() / "artifacts" / "measurements" / substrate / model
    base.mkdir(parents=True, exist_ok=True)
    leaf = f"{experiment}_v{version}_{ts}_{sha}"
    mdir = _unique_dir(base, leaf)
    mdir.mkdir(parents=True, exist_ok=True)
    if update_latest:
        link = base / f"{experiment}_latest"
        tmp = base / f".{experiment}_latest.{os.urandom(3).hex()}"
        os.symlink(mdir.name, tmp)   # RELATIVE target (bwrap-safe)
        os.replace(tmp, link)
    return MeasurementDir(path=mdir, manifest_path=mdir / "manifest.yaml", run_id=mdir.name,
                          substrate=substrate, model=model, experiment=experiment,
                          version=version, git_sha=sha, timestamp=ts, notes=notes, _artifacts=[])


# ---- caches & recaptures: artifacts/cache/<ns>/, artifacts/recaptures/ -----


def cache_dir(namespace: str, *, ensure: bool = True) -> Path:
    """Large regenerable cache dir under ``artifacts/cache/<namespace>/`` (PURGEABLE)."""
    d = repo_root() / "artifacts" / "cache" / namespace
    if ensure:
        d.mkdir(parents=True, exist_ok=True)
    return d


def recaptures_dir() -> Path:
    """Model-recapture root ``artifacts/recaptures/`` (PURGEABLE; ~150 GB, regenerable via m2m).

    Holds model captures (``<model>_<dtype>_<variant>/model.mlir`` + weights/io) and golden
    reference outputs, consumed by every target backend. The legacy ``output/`` tree is retired.
    """
    return repo_root() / "artifacts" / "recaptures"
