#!/usr/bin/env python3
"""Build a FireSim design from a shared Chipyard checkout without writing into it.

A shared checkout accumulates state that is nobody's to change: other sessions build in it, its
submodules sit at revisions that do not all compile together, and a driver or bitstream build
writes generated sources, outputs, sbt targets and a regenerated accelerator header into it.

An overlay is a directory with the checkout's SHAPE, in which everything read is a symlink back to
the base and everything a build WRITES is a private directory:

  project/{project,target}, target/, .classpath_cache/, .java_tmp/     sbt and the generator jar
  sims/firesim-staging/generated-src/                                 Chisel elaboration
  sims/firesim/sim/{generated-src,output,.java_tmp}                   Golden Gate and the driver
  sims/firesim/sim/midas/src/main/scala/target-symlinks/              links the driver build makes
  sims/firesim/deploy/{logs,results-build,results-workload,built-hwdb-entries,workloads}
  sims/firesim/deploy/config_*.yaml                                   copied, so they can be edited
  sims/firesim/deploy/firesim                                         copied: the manager and the
                                                                      queue launcher root the
                                                                      checkout at its real path
  sims/firesim/platforms/<platform>/                                  real, children linked: a
                                                                      bitstream build adds its own
                                                                      per-design copy beside them

`--exclude-generator NAME` leaves `generators/NAME` an empty directory. Chipyard wires an optional
generator into the build only when `generators/NAME/.git` exists, so an excluded generator is
skipped by the build's own rule and no source of it is patched.

Every sbt project of every `build.sbt` in the base gets a private `target/`, and so does every
`target/` that already exists. sbt writes a project's
classes beside its sources, so a project directory that is merely a symlink would send the whole
compile into the base, where another session may be building the same project.

`--private-dir REL` makes one directory below the base private and copies its files, for a build
step that rewrites a tracked file in place (an accelerator's generated parameter header).

Nothing in the base is modified. Re-running on an existing overlay refuses rather than merges.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PRIVATE_DIRS = (
    "project/project",
    "project/target",
    "target",
    ".classpath_cache",
    ".java_tmp",
    "sims/firesim-staging/generated-src",
    "sims/firesim/sim/generated-src",
    "sims/firesim/sim/output",
    "sims/firesim/sim/.java_tmp",
    "sims/firesim/sim/midas/src/main/scala/target-symlinks",
    "sims/firesim/deploy/logs",
    "sims/firesim/deploy/results-build",
    "sims/firesim/deploy/results-workload",
    "sims/firesim/deploy/built-hwdb-entries",
    "sims/firesim/deploy/workloads",
    "sims/firesim/deploy/generated-topology-diagrams",
)
# Directories a build ADDS CHILDREN to. Each becomes a real directory whose existing children are
# linked one by one, so a new child lands in the overlay. A bitstream build copies its platform
# template to `platforms/<platform>/cl_<design>/` beside the template; with the platform directory
# itself a link, that copy was created in the base.
SHALLOW_GLOBS = ("sims/firesim/platforms/*",)
# Editor and build-server state: sharing it lets two builds fight over one server.
NEVER_LINKED = (".bloop", ".metals", ".bsp", ".vscode")
# Copied, not linked. The configs so they can be edited; the manager's entry script because both
# the manager and the queue launcher find the checkout from that file's REAL path, so a link to it
# sends every run back to the base checkout, where this overlay's driver does not exist.
COPIED_GLOBS = ("sims/firesim/deploy/config_*.yaml", "sims/firesim/deploy/firesim")
SUDO_SCRIPTS = Path("sims/firesim/deploy/sudo-scripts")


def _mirror(root: Path, base: Path, overlay: Path, private: set[Path]) -> None:
    """Link ``base``'s children into ``overlay``, descending only toward a private directory."""
    overlay.mkdir(parents=True, exist_ok=True)
    for child in sorted(base.iterdir()):
        relative = child.relative_to(root)
        if relative in private or child.name in NEVER_LINKED:
            continue
        on_path = any(relative in item.parents for item in private)
        if on_path and child.is_dir() and not child.is_symlink():
            _mirror(root, child, overlay / child.name, private)
        else:
            (overlay / child.name).symlink_to(child)


# Never descended into when looking for sbt builds: environments and payloads, not build trees.
_NOT_BUILD_TREES = (
    ".git",
    ".conda-env",
    ".conda-lock-env",
    "software",
    "toolchains",
    "node_modules",
    "generated-src",
    "output",
    "results-build",
)
_SCAN_DEPTH = 6


def _walk(base: Path, depth: int = 0):
    """Real directories below ``base``, bounded, never through a symlink."""
    if depth > _SCAN_DEPTH:
        return
    for child in sorted(base.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.name in _NOT_BUILD_TREES:
            continue
        yield child
        if child.name != "target":
            yield from _walk(child, depth + 1)


def sbt_project_targets(base: Path) -> set[Path]:
    """Every place an sbt build under ``base`` writes classes.

    Three sources, because each misses what another catches: the projects a `build.sbt` declares
    with `file("...")` (relative to that build), the build's own `target` and `project/target`, and
    every `target` directory that already exists (a project declared some other way, such as
    `file("x") / "src"`, shows up here once the base has been built).
    """
    targets: set[Path] = set()

    def add(project: Path) -> None:
        relative = Path(*[part for part in project.relative_to(base).parts if part != "."])
        targets.add(relative / "target")
        targets.add(relative / "project" / "target")

    builds = [base] + [d for d in _walk(base) if (d / "build.sbt").is_file()]
    for build in builds:
        if not (build / "build.sbt").is_file():
            continue
        add(build)
        for chunk in (build / "build.sbt").read_text(encoding="utf-8").split('file("')[1:]:
            declared = chunk.partition('"')[0]
            if declared and "$" not in declared and (build / declared).is_dir():
                add(build / declared)
    for directory in _walk(base):
        if directory.name == "target":
            targets.add(directory.relative_to(base))
    return {Path(*[part for part in item.parts if part != "."]) for item in targets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", type=Path, required=True, help="the shared Chipyard checkout")
    parser.add_argument("--overlay", type=Path, required=True, help="directory to create")
    parser.add_argument("--exclude-generator", action="append", default=[], metavar="NAME")
    parser.add_argument("--private-dir", action="append", default=[], metavar="REL", type=Path)
    parser.add_argument(
        "--installed-sudo-scripts",
        type=Path,
        metavar="DIR",
        help="directory holding the host's installed FireSim sudo scripts; the "
        "overlay's copies are made identical to them",
    )
    arguments = parser.parse_args(argv)

    base = arguments.base.resolve(strict=True)
    overlay = arguments.overlay.absolute()
    if overlay.exists():
        print(f"refusing to merge into an existing overlay: {overlay}", file=sys.stderr)
        return 2
    for required in ("build.sbt", "env.sh", "sims/firesim/sourceme-manager.sh"):
        if not (base / required).exists():
            print(f"{base} is not a Chipyard checkout with FireSim: missing {required}", file=sys.stderr)
            return 2

    excluded = {Path("generators") / name for name in arguments.exclude_generator}
    copied_dirs = set(arguments.private_dir)
    if arguments.installed_sudo_scripts is not None:
        copied_dirs.add(SUDO_SCRIPTS)
    private = {Path(item) for item in PRIVATE_DIRS} | excluded | copied_dirs | sbt_project_targets(base)
    # A private directory below an excluded generator would recreate what the exclusion removed.
    private = {item for item in private if item in excluded or not any(gone in item.parents for gone in excluded)}
    shallow = {
        path.relative_to(base)
        for pattern in SHALLOW_GLOBS
        for path in base.glob(pattern)
        if path.is_dir() and not path.is_symlink()
    }
    _mirror(base, base, overlay, private | shallow)
    for relative in sorted(private):
        (overlay / relative).mkdir(parents=True, exist_ok=True)
    for relative in sorted(shallow):
        (overlay / relative).mkdir(parents=True, exist_ok=True)
        for child in sorted((base / relative).iterdir()):
            (overlay / relative / child.name).symlink_to(child)
    for relative in sorted(copied_dirs):
        for source in sorted((base / relative).iterdir()):
            if source.is_file():
                shutil.copy2(source, overlay / relative / source.name)
    if arguments.installed_sudo_scripts is not None:
        # The manager refuses to run when a script the host has installed differs from the
        # checkout's copy of it. A host whose administrator changed an installed script (to narrow
        # device permissions, say) then fails every stock checkout. The installed script is the one
        # that runs; the checkout's copy is only compared, so the overlay's copy is made to match.
        for script in sorted((overlay / SUDO_SCRIPTS).iterdir()):
            installed = arguments.installed_sudo_scripts / script.name
            if installed.is_file():
                shutil.copyfile(installed, script)
    for pattern in COPIED_GLOBS:
        for source in sorted(base.glob(pattern)):
            destination = overlay / source.relative_to(base)
            if destination.is_symlink():
                destination.unlink()
            shutil.copy2(source, destination)
    print(overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
