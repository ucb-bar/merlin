"""``merlin-storage`` -- what the generated-output root costs, and what is safe to reclaim.

Disk under ``out/`` grew until it was an incident twice, and both times the diagnosis took longer
than the fix because nothing reported the shape of the growth: a campaign that looked like "runs got
bigger" was really one 12.8 GB input closure copied once per run, and a tree that looked like live
evidence was mostly per-run duplicates of files that still existed at their original path. The
numbers that answer "why is it big" -- how much of the root is *shared* rather than distinct, and
which subtrees are declared regenerable -- are cheap to compute and were simply never printed.

``report`` prints them. ``prune`` acts, and only on classes whose safety is a property rather than a
judgement:

* **store-orphans** -- content-store objects no snapshot links any more. A hard link keeps the bytes
  alive for as long as one snapshot names them, so removing an object with a single link cannot take
  data away from a run, and an object with more than one link is skipped. This is what makes the
  store safe to keep under the declared-purgeable cache root.
* **pending-snapshots** -- ``bundle_inputs.pending`` trees. Materialization unwinds these on failure;
  one that survived is a run killed mid-copy and was never a complete input closure.
* **caches** -- ``out/artifacts/cache/<ns>/``, which the layout convention declares PURGEABLE.

Everything else is left alone and merely reported, because "probably finished" is not a property of
a run directory that this tool can read off the filesystem. Dry-run is the default; ``--apply`` acts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from merlin.common import content_store
from merlin.common.paths import artifacts_dir, build_dir, out_dir, runs_dir

_SNAPSHOT_DIR = "bundle_inputs"
_PENDING_SUFFIX = ".pending"


class Usage:
    """Bytes under a tree, counting each inode once.

    ``apparent`` is what the files claim to add up to; ``occupied`` counts a multiply-linked inode
    once. Their difference is the sharing this root gets for free -- the number that says whether a
    tree is large because it holds a lot, or large because it holds the same thing repeatedly.
    """

    def __init__(self) -> None:
        self.apparent = 0
        self.occupied = 0
        self.files = 0
        self._seen: set[tuple[int, int]] = set()

    def add(self, stat: os.stat_result) -> None:
        self.files += 1
        self.apparent += stat.st_size
        key = (stat.st_dev, stat.st_ino)
        if key not in self._seen:
            self._seen.add(key)
            self.occupied += stat.st_size


def measure(root: Path, *, usage: Usage | None = None) -> Usage:
    """Walk ``root`` without following symlinks, tolerating a tree another session is editing."""
    usage = usage if usage is not None else Usage()
    if not root.exists():
        return usage
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue                          # vanished or unreadable mid-walk; report what we saw
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(Path(entry.path))
                else:
                    usage.add(entry.stat(follow_symlinks=False))
            except OSError:
                continue
    return usage


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:,.1f} TB"


def store_root() -> Path:
    """The store to report on. Disabling sharing does not hide the objects already in the store, so
    an operator who turned it off can still see and reclaim what it holds."""
    return content_store.store_root() or content_store.default_root()


def store_orphans(root: Path | None = None) -> tuple[list[Path], int]:
    """Store objects nothing references any more -- see merlin.common.content_store.orphans."""
    return content_store.orphans(root if root is not None else store_root())


def pending_snapshots(root: Path | None = None) -> tuple[list[Path], int]:
    """Input closures abandoned mid-copy. A complete one is named without the suffix."""
    root = root if root is not None else out_dir()
    found: list[Path] = []
    total = 0
    if not root.is_dir():
        return found, total
    for path in root.rglob(_SNAPSHOT_DIR + _PENDING_SUFFIX):
        if path.is_dir() and not path.is_symlink():
            found.append(path)
            total += measure(path).occupied
    return found, total


def purgeable_caches(store: Path | None = None) -> tuple[list[Path], int]:
    """``out/artifacts/cache/<ns>/`` -- regenerable by the layout convention that created it.

    The content store lives here too and is reported separately, because pruning it wholesale is
    wasteful rather than unsafe: its live objects would be re-copied by the next run that grants
    the same file.
    """
    store = store if store is not None else store_root()
    cache = artifacts_dir() / "cache"
    found: list[Path] = []
    total = 0
    if not cache.is_dir():
        return found, total
    for path in sorted(cache.iterdir()):
        if not path.is_dir() or path.is_symlink() or path.resolve() == store.resolve():
            continue
        found.append(path)
        total += measure(path).occupied
    return found, total


def snapshot_sharing() -> dict:
    """How much the per-run input closures cost, and how much of that is shared."""
    usage = Usage()
    count = 0
    for path in out_dir().rglob(_SNAPSHOT_DIR):
        if path.is_dir() and not path.is_symlink():
            count += 1
            measure(path, usage=usage)        # one Usage across all of them: shared inodes collapse
    return {"snapshots": count, "apparent_bytes": usage.apparent,
            "occupied_bytes": usage.occupied, "files": usage.files}


# --- layout drift ------------------------------------------------------------------------------
# The layout convention names THREE top-level dirs under out/ and a closed set of concerns under
# out/artifacts/. Neither is enforced for UNTRACKED output, which is almost all of it: the tracked
# linter (check_artifact_layout.py) can only see files in the index, and generated output is
# gitignored by design. So the convention held for the tracked tree and drifted freely underneath
# it. Measured 2026-09-15: 4 undeclared top-level dirs and 52 undeclared concerns, against 16
# declared ones. A one-off directory is not a crime; 52 of them means nobody can find anything, and
# a reader cannot tell a live concern from a debugging detour someone left behind.
#
# This reports, it does not move: a product's path is quoted in reports, manifests and docs, and a
# tidy tree that broke those references would be a poor trade.
OUT_ROOTS = ("runs", "artifacts", "build")
DECLARED_CONCERNS = (
    "cache", "capsule-bench", "ceiling", "compare", "design-pressure", "dse", "dse-guidance",
    "kernel-index", "kernel-mining", "measurements", "optimization-surface", "perf-bench",
    "presentation", "recaptures", "selfcheck", "targets",
)


def layout_drift() -> dict:
    """What sits under out/ that the convention does not name, and what it costs."""
    report: dict = {"stray_roots": [], "undeclared_concerns": [], "declared_concerns": []}
    for path in _safe_dirs(out_dir()):
        if path.name not in OUT_ROOTS:
            usage = measure(path)
            report["stray_roots"].append({"name": path.name, "bytes": usage.occupied,
                                          "files": usage.files})
    for path in _safe_dirs(artifacts_dir()):
        usage = measure(path)
        row = {"name": path.name, "bytes": usage.occupied, "files": usage.files,
               "units": len(_safe_dirs(path))}
        key = "declared_concerns" if path.name in DECLARED_CONCERNS else "undeclared_concerns"
        report[key].append(row)
    for key in ("stray_roots", "undeclared_concerns", "declared_concerns"):
        report[key].sort(key=lambda r: -r["bytes"])
    return report


def _print_layout(report: dict, top: int) -> None:
    stray, undeclared = report["stray_roots"], report["undeclared_concerns"]
    declared = report["declared_concerns"]
    print(f"out/ roots: {len(OUT_ROOTS)} declared ({', '.join(OUT_ROOTS)}), "
          f"{len(stray)} undeclared")
    for row in stray:
        print(f"  [stray root]  {_human(row['bytes']):>10}  {row['files']:>7,} files  "
              f"out/{row['name']}")
    if stray:
        print("  -> generated output belongs under one of the three roots; anything else is a root\n"
              "     the convention retired, and the write-guard hook only blocks paths it knows.")

    print(f"\nout/artifacts concerns: {len(declared)} declared, {len(undeclared)} undeclared")
    declared_bytes = sum(r["bytes"] for r in declared)
    undeclared_bytes = sum(r["bytes"] for r in undeclared)
    print(f"  declared   {_human(declared_bytes):>10}")
    print(f"  undeclared {_human(undeclared_bytes):>10}   "
          f"({len(undeclared)} dirs, {sum(r['units'] for r in undeclared):,} units)")
    if undeclared:
        print(f"\n  largest undeclared (top {top}):")
        for row in undeclared[:top]:
            print(f"    {_human(row['bytes']):>10}  {row['units']:>5} units  {row['name']}")
        print("\n  Each needs one of: fold into a declared concern, add it to the convention as a\n"
              "  real concern, or retire it. Reported only -- paths appear in reports and manifests.")


# --- per-experiment accounting ----------------------------------------------------------------
# "How much does one run cost?" had no answer, so every disk conversation was about totals -- and a
# total cannot distinguish a concern that is big because each run is bloated from one that is big
# because nothing ever deletes a finished campaign. Those want opposite fixes: the first is a bug in
# the producer, the second a retention decision. Measured 2026-09-15: a phase-2 perf-bench unit is
# 170 MB and there are 172 of them, while the same concern's 297 other units average 187 MB -- so
# perf-bench is large by accumulation, not by bloat, and no amount of de-duplication would fix it.
_VERSION_PREFIX = "v"


def _is_version_level(name: str) -> bool:
    """``v1``/``v12`` product-version levels sit BETWEEN the axis and the unit.

    Treating one as a unit prices a whole version series as a single experiment -- ``perf-bench``'s
    ``v1`` holds 34 GB -- which is the one number guaranteed to mislead.
    """
    return (name.startswith(_VERSION_PREFIX) and len(name) > 1
            and name[1:].isdigit())


def experiment_units() -> list[tuple[str, str, Path]]:
    """``(group, unit name, path)`` for every directory that represents ONE experiment.

    Two shapes, both from the layout convention: an aet run at
    ``out/runs/<target>/<suite>/<run-id>`` and a product at
    ``out/artifacts/<concern>/<axis>/[v<n>/]<unit>``. Anything shallower is a container and anything
    deeper is a unit's own content, so this is the level at which "per experiment" means something.
    """
    found: list[tuple[str, str, Path]] = []

    def descend(group: str, parent: Path, depth: int) -> None:
        try:
            children = sorted(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            return
        for child in children:
            if _is_version_level(child.name) and depth == 0:
                descend(_label(child), child, depth)
            else:
                found.append((group, child.name, child))

    for target in _safe_dirs(runs_dir()):
        for suite in _safe_dirs(target):
            descend(_label(suite), suite, 0)
    for concern in _safe_dirs(artifacts_dir()):
        if concern.name == "cache":
            continue                          # regenerable by convention; priced as a cache instead
        for axis in _safe_dirs(concern):
            descend(_label(axis), axis, 0)
    return found


def _label(path: Path) -> str:
    """A group's name is its place under the out/ root, derived rather than spelled.

    Writing the root names into the label would also bake in the layout this module is meant to
    REPORT on, and it would go wrong the moment ``MERLIN_OUT_ROOT`` points somewhere else.
    """
    try:
        return path.relative_to(out_dir()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_dirs(parent: Path) -> list[Path]:
    try:
        return sorted(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return []


def experiment_costs(match: str | None = None) -> dict:
    """Per-group experiment cost. ``match`` keeps only units whose name contains it (case-folded),
    which is how a campaign spread across groups -- ``phase2``, a model name, a date -- gets priced
    as one thing."""
    groups: dict[str, dict] = {}
    needle = match.casefold() if match else None
    for group, name, path in experiment_units():
        if needle and needle not in name.casefold():
            continue
        usage = measure(path)
        row = groups.setdefault(group, {"units": 0, "bytes": 0, "files": 0, "largest": ("", 0)})
        row["units"] += 1
        row["bytes"] += usage.occupied
        row["files"] += usage.files
        if usage.occupied > row["largest"][1]:
            row["largest"] = (name, usage.occupied)
    for row in groups.values():
        row["mean_bytes"] = row["bytes"] // row["units"] if row["units"] else 0
    return groups


def _print_experiments(groups: dict, top: int, match: str | None) -> None:
    if not groups:
        print(f"no experiment units{f' matching {match!r}' if match else ''} under the out/ root")
        return
    rows = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
    total_units = sum(r["units"] for _, r in rows)
    total_bytes = sum(r["bytes"] for _, r in rows)
    shown = rows[:top]
    label = f" matching {match!r}" if match else ""
    print(f"{total_units:,} experiment unit(s){label}, {_human(total_bytes)} "
          f"(top {len(shown)} of {len(rows)} groups by total)\n")
    print(f"  {'group':<46} {'units':>6} {'total':>12} {'mean/unit':>12} {'files':>10}")
    for group, row in shown:
        print(f"  {group:<46} {row['units']:>6} {_human(row['bytes']):>12} "
              f"{_human(row['mean_bytes']):>12} {row['files']:>10,}")
    print("\nA group that is large with a SMALL mean is large by accumulation -- that is a retention\n"
          "decision, not a producer bug. A large mean is the producer writing too much per run.")
    biggest = max(rows, key=lambda kv: kv[1]["mean_bytes"])
    print(f"\nheaviest per run: {biggest[0]} at {_human(biggest[1]['mean_bytes'])}/unit "
          f"(largest single unit: {biggest[1]['largest'][0]} "
          f"{_human(biggest[1]['largest'][1])})")


def collect() -> dict:
    roots = {"runs": runs_dir(), "artifacts": artifacts_dir(), "build": build_dir()}
    report: dict = {"out_root": str(out_dir()), "roots": {}, "concerns": {}}
    try:
        disk = shutil.disk_usage(out_dir() if out_dir().exists() else Path.cwd())
        report["filesystem"] = {"total_bytes": disk.total, "used_bytes": disk.used,
                                "free_bytes": disk.free}
    except OSError:
        report["filesystem"] = {}
    for name, path in roots.items():
        usage = measure(path)
        report["roots"][name] = {"apparent_bytes": usage.apparent,
                                 "occupied_bytes": usage.occupied, "files": usage.files}
    if artifacts_dir().is_dir():
        for concern in sorted(p for p in artifacts_dir().iterdir()
                              if p.is_dir() and not p.is_symlink()):
            usage = measure(concern)
            report["concerns"][concern.name] = {"occupied_bytes": usage.occupied,
                                                "files": usage.files}
    report["bundle_snapshots"] = snapshot_sharing()
    store = store_root()
    store_usage = measure(store)
    orphans, orphan_bytes = store_orphans(store)
    pending, pending_bytes = pending_snapshots()
    caches, cache_bytes = purgeable_caches(store)
    report["store"] = {"path": str(store), "occupied_bytes": store_usage.occupied,
                       "objects": store_usage.files, "orphan_objects": len(orphans),
                       "orphan_bytes": orphan_bytes}
    report["reclaimable"] = {
        "store-orphans": {"count": len(orphans), "bytes": orphan_bytes,
                          "why": "no snapshot links these objects any more"},
        "pending-snapshots": {"count": len(pending), "bytes": pending_bytes,
                              "why": "input closures abandoned mid-copy; never complete"},
        "caches": {"count": len(caches), "bytes": cache_bytes,
                   "why": "out/artifacts/cache/<ns> is declared regenerable"},
    }
    return report


def _print_report(report: dict, top: int) -> None:
    fs = report.get("filesystem") or {}
    if fs:
        print(f"filesystem   {_human(fs['used_bytes'])} used, {_human(fs['free_bytes'])} free "
              f"of {_human(fs['total_bytes'])}")
    print(f"out root     {report['out_root']}")
    for name, row in report["roots"].items():
        shared = row["apparent_bytes"] - row["occupied_bytes"]
        note = f"  ({_human(shared)} shared with other names)" if shared else ""
        print(f"  {name:<10} {_human(row['occupied_bytes']):>12}  {row['files']:>9,} files{note}")

    concerns = sorted(report["concerns"].items(), key=lambda kv: -kv[1]["occupied_bytes"])
    if concerns:
        print(f"\nlargest artifact concerns (top {top}):")
        for name, row in concerns[:top]:
            print(f"  {name:<24} {_human(row['occupied_bytes']):>12}  {row['files']:>9,} files")

    snap = report["bundle_snapshots"]
    if snap["snapshots"]:
        saved = snap["apparent_bytes"] - snap["occupied_bytes"]
        share = saved / snap["apparent_bytes"] * 100 if snap["apparent_bytes"] else 0.0
        print(f"\nper-run input closures: {snap['snapshots']} snapshots, "
              f"{_human(snap['apparent_bytes'])} declared, {_human(snap['occupied_bytes'])} on disk "
              f"({_human(saved)} shared, {share:.0f}%)")
        if share < 1.0 and snap["snapshots"] > 1:
            print("  NOTE: these closures share almost nothing. If their grants overlap, they "
                  "predate the content store or were written with MERLIN_BUNDLE_CAS disabled.")
    store = report["store"]
    print(f"content store: {_human(store['occupied_bytes'])} in {store['objects']:,} objects "
          f"({store['orphan_objects']:,} unreferenced, {_human(store['orphan_bytes'])})")

    print("\nreclaimable now (merlin-storage prune --apply <class>):")
    for name, row in report["reclaimable"].items():
        print(f"  {name:<20} {_human(row['bytes']):>12}  {row['count']:>6} items   {row['why']}")


CLASSES = ("store-orphans", "pending-snapshots", "caches")


def _candidates(names: tuple[str, ...]) -> list[tuple[str, Path, int]]:
    out: list[tuple[str, Path, int]] = []
    if "store-orphans" in names:
        for path in store_orphans()[0]:
            try:
                out.append(("store-orphans", path, path.stat(follow_symlinks=False).st_size))
            except OSError:
                continue
    if "pending-snapshots" in names:
        for path in pending_snapshots()[0]:
            out.append(("pending-snapshots", path, measure(path).occupied))
    if "caches" in names:
        for path in purgeable_caches()[0]:
            out.append(("caches", path, measure(path).occupied))
    return out


def _remove(path: Path) -> None:
    """Remove a frozen file or tree, restoring only the write bits removal actually needs.

    A freeze clears write bits, so the DIRECTORIES have to be made writable again -- unlinking an
    entry is a write to its parent. The files must be left alone: a store-backed one shares its
    inode with every other frozen tree holding the same bytes, and chmod follows the inode, so
    "restoring the write bit" here would reach into live snapshots and make their own files
    writable. That is not hypothetical -- it is the same mechanism that made a second perf-bench
    snapshot fail verification, and it is why nothing in the reclaim path touches a file's mode.
    """
    if path.is_dir():
        for child in sorted(path.rglob("*"), key=lambda p: len(p.parts)):
            try:
                if child.is_dir() and not child.is_symlink():
                    child.chmod(child.stat().st_mode | 0o700)
            except OSError:
                continue
        path.chmod(path.stat().st_mode | 0o700)
        shutil.rmtree(path)
        return
    # A single store object: its last link is this name, so its mode is nobody else's business.
    path.chmod(path.stat().st_mode | 0o200)
    path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-storage", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("report", help="what the generated-output root costs and why "
                                        "(walks the whole root; minutes on a large one)")
    show.add_argument("--top", type=int, default=12, help="how many artifact concerns to list")
    show.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    exp = sub.add_parser("experiments", help="what ONE experiment costs, by group")
    exp.add_argument("--top", type=int, default=20, help="how many groups to list")
    exp.add_argument("--match", help="only units whose name contains this (e.g. phase2)")
    exp.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    lay = sub.add_parser("layout", help="what sits under out/ that the convention does not name")
    lay.add_argument("--top", type=int, default=15, help="how many undeclared concerns to list")
    lay.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    prune = sub.add_parser("prune", help="reclaim the provably-safe classes (dry run by default)")
    prune.add_argument("classes", nargs="*", choices=CLASSES,
                       help="which classes to reclaim (default: all of them)")
    prune.add_argument("--apply", action="store_true",
                       help="actually remove; without it nothing is touched")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "report":
        report = collect()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_report(report, args.top)
        return 0

    if args.command == "experiments":
        groups = experiment_costs(args.match)
        if args.json:
            print(json.dumps(groups, indent=2, sort_keys=True))
        else:
            _print_experiments(groups, args.top, args.match)
        return 0

    if args.command == "layout":
        report = layout_drift()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_layout(report, args.top)
        return 0

    chosen = tuple(args.classes or CLASSES)
    candidates = _candidates(chosen)
    total = sum(size for _, _, size in candidates)
    if not candidates:
        print("nothing to reclaim in " + ", ".join(chosen))
        return 0
    verb = "removing" if args.apply else "would remove"
    print(f"{verb} {len(candidates)} items, {_human(total)}")
    failures = 0
    for kind, path, size in candidates:
        if not args.apply:
            print(f"  [{kind}] {_human(size):>12}  {path}")
            continue
        try:
            _remove(path)
            print(f"  [{kind}] {_human(size):>12}  {path}")
        except OSError as exc:
            failures += 1
            print(f"  [{kind}] FAILED {path}: {exc}", file=sys.stderr)
    if not args.apply:
        print("\ndry run — pass --apply to reclaim")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
