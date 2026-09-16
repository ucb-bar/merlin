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
_CONTRACT = "storage.yaml"


def contract() -> dict:
    """The declared storage layout, from ``merlin/contract/storage.yaml``.

    Scan roots, the out/ roots, the concerns allowed under out/artifacts/, where a concern that
    predates the roster belongs, and which freezes verify by file mode. All of these were literals
    in this module, and the two that describe the layout drifted badly, because a roster in code is
    a roster nobody edits: 52 undeclared concerns had accumulated against 16 declared ones by
    2026-09-15. They are data now, for the same reason no other path in this repo is spelled in a
    library module. A missing or unreadable contract degrades to "nothing declared" rather than
    raising, so pricing the root still works in a checkout that has not got the file.
    """
    from merlin.common.paths import merlin_dir  # noqa: PLC0415
    from merlin.common.yaml import load_yaml    # noqa: PLC0415

    try:
        return load_yaml(merlin_dir() / "contract" / _CONTRACT) or {}
    except (OSError, ValueError):
        return {}


def out_roots() -> tuple[str, ...]:
    """The top-level directories under out/ that the convention names."""
    return tuple(contract().get("out_roots") or ())


def declared_concerns() -> tuple[str, ...]:
    """Every concern allowed to own a subtree of out/artifacts/."""
    return tuple(sorted(contract().get("concerns") or {}))


def scan_roots(extra: list[Path] | None = None) -> list[Path]:
    """Every root that can hold generated state: the out/ root, plus the declared workspace roots.

    An agent run freezes its declared input closure as a SIBLING of its workspace, and a
    capsule-bench workspace lives beside its target experiment rather than under out/. So the roots
    this tool must walk are not all reachable from ``out_dir()``, and the first version of it walked
    only that -- it was blind to 40 GB of completed closures and one 8.7 GB closure abandoned
    mid-copy, which is precisely the class it exists to find.

    The extra roots come from ``merlin/contract/storage.yaml`` rather than from a literal here, for
    the reason every other path in that directory is declared: no library module spells a checkout
    directory, so relocating a workspace is an edit to data. A pattern matching nothing is skipped.
    """
    from merlin.common.paths import merlin_dir  # noqa: PLC0415

    base = merlin_dir()
    roots = [out_dir()]
    patterns = contract().get("scan_roots") or []
    for pattern in patterns:
        for path in sorted(base.glob(str(pattern))):
            if path.is_dir() and not path.is_symlink():
                roots.append(path)
    for path in extra or []:
        if path.is_dir():
            roots.append(path)
    # A root nested inside another would be walked twice and double-counted.
    kept: list[Path] = []
    for path in sorted({p.resolve() for p in roots}, key=lambda p: len(p.parts)):
        if not any(parent in path.parents for parent in kept):
            kept.append(path)
    return kept


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


def pending_snapshots(roots: list[Path] | None = None) -> tuple[list[Path], int]:
    """Input closures abandoned mid-copy. A complete one is named without the suffix."""
    found: list[Path] = []
    total = 0
    for root in (roots if roots is not None else scan_roots()):
        if not root.is_dir():
            continue
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


def snapshot_sharing(roots: list[Path] | None = None) -> dict:
    """How much the per-run input closures cost, and how much of that is shared.

    ONE ``Usage`` across every closure on purpose: a store-backed inode is counted once no matter
    how many runs link it, so apparent-minus-occupied IS the saving the store is producing.
    """
    usage = Usage()
    count = 0
    for root in (roots if roots is not None else scan_roots()):
        if not root.is_dir():
            continue
        for path in root.rglob(_SNAPSHOT_DIR):
            if path.is_dir() and not path.is_symlink():
                count += 1
                measure(path, usage=usage)
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
# Reporting it was the first half; ``organize`` is the second. A product's path is quoted in
# manifests, reports and docs this repo does not own, so a fold MOVES the directory and leaves a
# relative symlink at the old name -- the tree gets organized and every existing citation still
# resolves. Where each one belongs is declared in the contract, not decided here.


def layout_drift() -> dict:
    """What sits under out/ that the convention does not name, and what it costs."""
    roots, concerns = out_roots(), declared_concerns()
    report: dict = {"stray_roots": [], "undeclared_concerns": [], "declared_concerns": []}
    for path in _safe_dirs(out_dir()):
        if path.name not in roots:
            usage = measure(path)
            report["stray_roots"].append({"name": path.name, "bytes": usage.occupied,
                                          "files": usage.files})
    for path in _safe_dirs(artifacts_dir()):
        usage = measure(path)
        row = {"name": path.name, "bytes": usage.occupied, "files": usage.files,
               "units": len(_safe_dirs(path))}
        key = "declared_concerns" if path.name in concerns else "undeclared_concerns"
        report[key].append(row)
    for key in ("stray_roots", "undeclared_concerns", "declared_concerns"):
        report[key].sort(key=lambda r: -r["bytes"])
    report["unconventional_units"] = _unit_name_drift()
    report["undeclared_payload"] = _undeclared_payload()
    return report


def _undeclared_payload() -> list[dict]:
    """Bytes sitting inside a product unit that the unit's own manifest does not list.

    A product's ``manifest.yaml`` is its statement of what it contains, so anything in the directory
    and not in that statement is bulk the build left behind rather than a result -- and unlike every
    other number here, this one needs no threshold and no judgement, because the producer already
    declared the answer.

    Measured 2026-09-16: 35.06 of 35.11 GiB of undeclared payload across the whole root sits in ONE
    unit, whose manifest lists 43 small receipts and logs while the directory holds 19 whole-model
    ELFs of ~1.21 GB each, their weight blobs and three multi-GB zips. The same pass reports the
    9.53 GiB delivery bundle beside it at 0% undeclared, which is what makes the number trustworthy:
    a large product that declares its contents does not appear here at all.
    """
    from merlin.common.yaml import load_yaml  # noqa: PLC0415

    rows: list[dict] = []
    for group, name, path in experiment_units():
        manifest = path / "manifest.yaml"
        if not manifest.is_file():
            continue
        try:
            declared = set((load_yaml(manifest) or {}).get("artifacts") or [])
        except (OSError, ValueError):
            continue
        if not declared:
            continue                      # a placeholder manifest declares nothing, so nothing drifts
        total = undeclared = 0
        for directory, _dirs, names in os.walk(path):
            for entry in names:
                candidate = Path(directory) / entry
                try:
                    if candidate.is_symlink():
                        continue
                    size = candidate.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                total += size
                if candidate.relative_to(path).as_posix() not in declared | {"manifest.yaml"}:
                    undeclared += size
        if undeclared:
            rows.append({"group": group, "unit": name, "declared": len(declared),
                         "bytes": undeclared, "unit_bytes": total})
    rows.sort(key=lambda r: -r["bytes"])
    return rows


def _unit_name_drift() -> list[dict]:
    """Groups holding BOTH conventionally-named units and units carrying no timestamp at all.

    The concern roster is half of "where things live"; this is the other half. A unit whose name
    carries no ``<TS>`` token cannot be ordered against its siblings, so no retention depth can place
    it and no reader can tell which of them is current.

    Only MIXED groups are reported, and the restriction is what makes the number mean something. A
    group where nothing is timestamped is following a convention of its own -- a codegen package is
    named by package id, and the layout convention says so -- and calling that drift would bury the
    real signal in false positives. A group where some units are named to the convention and others
    are not is the signal: the same tree is being written both through ``new_product()`` and by
    something that just made a directory. Measured 2026-09-15: 448 of perf-bench's 475, which is why
    that concern reads as an accumulation problem when it is really a working directory that grew
    inside a product tree.
    """
    counts: dict[str, list[int]] = {}
    for group, name, _path in experiment_units():
        row = counts.setdefault(group, [0, 0])
        row[0] += 1
        if unit_timestamp(name) is None:
            row[1] += 1
    rows = [{"group": group, "units": total, "undated": odd}
            for group, (total, odd) in counts.items() if 0 < odd < total]
    rows.sort(key=lambda r: -r["undated"])
    return rows


def _print_layout(report: dict, top: int) -> None:
    stray, undeclared = report["stray_roots"], report["undeclared_concerns"]
    declared = report["declared_concerns"]
    roots = out_roots()
    print(f"out/ roots: {len(roots)} declared ({', '.join(roots)}), "
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
        print("\n  Each needs one of: a fold in merlin/contract/storage.yaml, an entry in that\n"
              "  file's concern roster, or retirement. `merlin-storage organize` applies the folds.")

    odd = report.get("unconventional_units") or []
    if odd:
        total = sum(r["undated"] for r in odd)
        print(f"\ngroups written both ways: {total:,} undated units across {len(odd)} groups "
              f"(top {min(top, len(odd))})")
        for row in odd[:top]:
            print(f"  {row['undated']:>6} of {row['units']:<6} {row['group']}")
        print("\n  These trees hold conventionally-named units AND directories with no <TS> token,\n"
              "  so the same concern is being written through new_product() and by something that\n"
              "  just made a directory. An undated unit cannot be ordered against its siblings, so no\n"
              "  retention depth can place it. That is a producer fix, not a disk decision.")

    payload = report.get("undeclared_payload") or []
    if payload:
        total = sum(r["bytes"] for r in payload)
        print(f"\nbytes inside a product its own manifest does not list: {_human(total)} across "
              f"{len(payload)} units (top {min(top, len(payload))})")
        for row in payload[:top]:
            share = row["bytes"] / row["unit_bytes"] * 100 if row["unit_bytes"] else 0
            print(f"  {_human(row['bytes']):>12} of {_human(row['unit_bytes']):>12} ({share:3.0f}%)  "
                  f"{row['declared']:>4} declared  {row['group']}/{row['unit']}")
        print("\n  A manifest is the product's statement of what it contains, so this needs no\n"
              "  threshold: the producer already declared the answer. Undeclared bulk is build\n"
              "  output that belongs under out/build/, referenced by digest rather than embedded.")


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
        for child in _safe_dirs(parent):
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
    """Real subdirectories, skipping the ones no convention put there.

    ``__pycache__`` beside a run directory was being priced as an experiment, which is a small error
    in bytes and a misleading one in counts -- a group's unit count is what says whether it is big by
    accumulation, and interpreter droppings are not units. Dot-directories are skipped for the same
    reason.
    """
    try:
        return sorted(p for p in parent.iterdir()
                      if p.is_dir() and not p.is_symlink()
                      and not p.name.startswith((".", "__")))
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


# --- folding the tree into its declared shape ---------------------------------------------------


def planned_folds() -> list[tuple[Path, Path, str]]:
    """``(source, destination, state)`` for every fold the contract declares.

    ``state`` is ``"pending"`` when the source is still a real directory, ``"done"`` when it is
    already the symlink a previous run left behind, ``"absent"`` when neither exists, and
    ``"conflict"`` when the destination is occupied by something this cannot safely merge into.
    """
    plan: list[tuple[Path, Path, str]] = []
    for entry in contract().get("folds") or []:
        try:
            source = out_dir() / str(entry["from"])
            destination = out_dir() / str(entry["into"])
        except (KeyError, TypeError):
            continue
        if source.is_symlink():
            state = "done"
        elif not source.is_dir():
            state = "absent"
        elif _holds_tracked_files(source):
            state = "tracked"
        elif not (destination.exists() or destination.is_symlink()):
            state = "pending"
        elif destination.is_dir() and not destination.is_symlink():
            state = "merge"                 # two old names, one concern: move the units across
        else:
            state = "conflict"
        plan.append((source, destination, state))
    return plan


def _holds_tracked_files(source: Path) -> bool:
    """Whether git has anything under this directory in its index.

    A fold leaves a SYMLINK at the old name, and git does not walk one: every tracked file under a
    folded directory immediately reads as deleted from the working tree, even though the bytes are
    intact behind the link. Moving those is a git operation, not a disk one, so a directory holding
    tracked content is reported and skipped -- which happened to two concerns holding five curated
    files the first time this ran.

    No git, or no index, means there is nothing to break, so the fold proceeds.
    """
    import subprocess  # noqa: PLC0415

    try:
        done = subprocess.run(["git", "ls-files", "-z", "--", str(source)],
                              capture_output=True, cwd=source.parent, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and bool(done.stdout.strip(b"\0"))


def fold(source: Path, destination: Path) -> None:
    """Move ``source`` to ``destination`` and leave a relative symlink at the old name.

    A rename is the whole move when the destination is free -- both names are under one root, so it
    is atomic and costs nothing regardless of how many bytes the tree holds. When the destination is
    an existing directory the fold is a MERGE, because more than one old name can belong to one
    concern; the units move across one at a time and a name collision aborts the whole fold rather
    than silently choosing a winner.

    The symlink is what makes the reorganization safe to do at all: a product's path is quoted in
    manifests, reports, figures and docs that this repo does not own and cannot rewrite, and a link
    is the one way to satisfy those readers and a tidy tree at the same time. It is relative, so the
    whole out/ root can still be moved or mounted somewhere else.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    if destination.is_dir() and not destination.is_symlink():
        for child in sorted(source.iterdir()):
            target = destination / child.name
            if target.exists() or target.is_symlink():
                for was, now in reversed(moved):
                    os.rename(now, was)
                raise FileExistsError(f"{target} already holds a unit of that name")
            os.rename(child, target)
            moved.append((child, target))
        source.rmdir()
    else:
        os.rename(source, destination)
        moved.append((source, destination))
    try:
        source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=True)
    except OSError:
        source.mkdir(parents=True, exist_ok=True)   # put it back rather than lose the old name
        for was, now in reversed(moved):
            os.rename(now, was)
        raise


# --- de-duplication ------------------------------------------------------------------------------
# The content store gets the saving at the moment a tree is frozen, which does nothing for the trees
# written before it existed -- and those are most of the root. The bytes are already there under
# several names; re-pointing each name at one store object collapses them without removing anything.
#
# The safety property is the same one that makes the store safe at all: a hard link is a claim, so
# every name still resolves to the same bytes afterwards, verified by digest before the swap. What
# it does change is the file's MODE, which becomes read-only and shared -- so a tree whose own
# integrity check reads the mode must be skipped, and the contract names the seal that marks one.


def _sealed(directory: Path, patterns: tuple[str, ...]) -> bool:
    """Whether this directory carries a seal whose verification reads file modes."""
    for pattern in patterns:
        try:
            if any(directory.glob(pattern)):
                return True
        except OSError:
            continue
    return False


def dedup_candidates(roots: list[Path] | None = None, *, min_bytes: int = 1 << 20) -> dict:
    """Files under ``roots`` that hold bytes some other file already holds.

    Digesting the whole root would cost hours, so only files that SHARE A SIZE with another file are
    digested -- two files with different sizes cannot have the same content, and that prefilter drops
    the work to a fraction of the root. ``min_bytes`` keeps the walk off the long tail of small files
    where the saving cannot repay the inode.
    """
    declared = contract()
    patterns = tuple(str(x) for x in (declared.get("mode_verified_seals") or []))
    excluded = {out_dir() / str(name) for name in (declared.get("dedup_excludes") or [])}
    store = store_root().resolve()
    by_size: dict[int, list[Path]] = {}
    seen: set[tuple[int, int]] = set()
    skipped_sealed = 0
    for root in (roots if roots is not None else scan_roots()):
        stack = [root]
        while stack:
            current = stack.pop()
            if _sealed(current, patterns):
                skipped_sealed += 1
                continue                      # its verifier reads file modes; sharing an inode breaks it
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        child = Path(entry.path)
                        if child.resolve() != store and child not in excluded:
                            stack.append(child)
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.st_size < min_bytes:
                    continue
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue                  # already counted under another name: it is the saving
                seen.add(key)
                by_size.setdefault(stat.st_size, []).append(Path(entry.path))

    groups: list[tuple[int, list[Path]]] = []
    reclaimable = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_digest: dict[str, list[Path]] = {}
        for path in paths:
            try:
                by_digest.setdefault(content_store.digest_file(path)[0], []).append(path)
            except OSError:
                continue
        for same in by_digest.values():
            if len(same) > 1:
                groups.append((size, sorted(same)))
                reclaimable += size * (len(same) - 1)
    groups.sort(key=lambda g: -g[0] * (len(g[1]) - 1))
    return {"groups": groups, "reclaimable_bytes": reclaimable,
            "files": sum(len(g[1]) for g in groups), "sealed_trees_skipped": skipped_sealed}


def _writable_parents(paths: list[Path]) -> dict[Path, int]:
    """Restore write on the DIRECTORIES holding these files, remembering the modes to put back.

    A frozen tree's directories are read-only, and replacing a directory entry is a write to the
    directory. The files are deliberately not touched: a file's mode belongs to its inode, which is
    the very thing being shared here, so restoring a write bit on one would reach into every other
    tree that holds the same content.
    """
    original: dict[Path, int] = {}
    for parent in {p.parent for p in paths}:
        try:
            mode = parent.stat().st_mode
        except OSError:
            continue
        if not mode & 0o200:
            try:
                parent.chmod(mode | 0o700)
                original[parent] = mode
            except OSError:
                continue
    return original


def dedup(groups: list[tuple[int, list[Path]]]) -> dict:
    """Re-point every name in each group at one store object.

    The reclaim is NOT the sum of what the names gave up. The first name in a group that the store
    has not seen puts a copy of those bytes INTO the store, and that copy is as real as the ones it
    replaced -- so counting only the released inodes over-reports by one file per new object, which
    for a group of seven 4.1 GB weight files is a 4.1 GB error. The store is measured before and
    after and the difference subtracted, which is why this returns three numbers rather than one.
    """
    store = content_store.store_root()
    before = measure(store_root()).occupied
    released = changed = 0
    for _size, paths in groups:
        restore = _writable_parents(paths)
        try:
            for path in paths:
                gained = content_store.adopt(path, store)
                if gained:
                    released += gained
                    changed += 1
        finally:
            for parent, mode in restore.items():
                try:
                    parent.chmod(mode)
                except OSError:
                    continue
    stored = max(0, measure(store_root()).occupied - before)
    return {"released_bytes": released, "stored_bytes": stored,
            "reclaimed_bytes": released - stored, "names": changed}


# --- retention ------------------------------------------------------------------------------------
# A concern that is big because each unit is bloated and one that is big because nothing ever drops a
# finished campaign want opposite fixes, and ``experiments`` is what tells them apart. This acts on
# the second: perf-bench holds 475 units averaging 110 MB, which no amount of de-duplication touches.
#
# Ordering is by the timestamp token the naming convention puts in the unit's own name, never by
# mtime. An mtime is not a statement about a run: it changes when anything walks or edits the tree,
# and a purge's own deletions update the mtimes of the units it touched, so an mtime rule reports the
# units you just edited as the live ones. A unit whose name carries no timestamp is never dropped --
# it cannot be placed in the order, so it cannot be shown to be old.
_TS_LEN = 16


def unit_timestamp(name: str) -> str | None:
    """The ``YYYYMMDDTHHMMSSZ`` token in a run or product name, or None if it carries none."""
    for token in name.split("_"):
        if (len(token) == _TS_LEN and token[8] == "T" and token[15] == "Z"
                and token[:8].isdigit() and token[9:15].isdigit()):
            return token
    return None


def retention_plan(keep: int, match: str | None = None) -> dict:
    """Per group: the units a depth of ``keep`` would drop, newest kept, with what they cost."""
    kept_alive: set[Path] = set()
    for _group, _name, path in experiment_units():
        pointer = path.parent / "latest"
        try:
            if pointer.is_symlink():
                kept_alive.add(pointer.resolve())
        except OSError:
            continue
    by_group: dict[str, list[tuple[str, str, Path]]] = {}
    unplaceable: dict[str, int] = {}
    needle = match.casefold() if match else None
    for group, name, path in experiment_units():
        if needle and needle not in name.casefold():
            continue
        stamp = unit_timestamp(name)
        if stamp is None:
            unplaceable[group] = unplaceable.get(group, 0) + 1
            continue
        by_group.setdefault(group, []).append((stamp, name, path))

    plan: dict = {"keep": keep, "groups": {}, "drop_bytes": 0, "drop_units": 0}
    for group, rows in by_group.items():
        rows.sort(reverse=True)
        drops = []
        for _stamp, name, path in rows[keep:]:
            try:
                if path.resolve() in kept_alive:
                    continue                  # a `latest` pointer resolves here; it is the live one
            except OSError:
                continue
            size = measure(path).occupied
            drops.append({"name": name, "path": str(path), "bytes": size})
            plan["drop_bytes"] += size
            plan["drop_units"] += 1
        plan["groups"][group] = {"units": len(rows), "undated": unplaceable.get(group, 0),
                                 "drops": drops}
    for group, count in unplaceable.items():
        plan["groups"].setdefault(group, {"units": 0, "undated": count, "drops": []})
    return plan


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


def _organize(apply: bool) -> int:
    plan = planned_folds()
    pending = [(src, dst, state) for src, dst, state in plan if state in ("pending", "merge")]
    conflicts = [(src, dst) for src, dst, state in plan if state == "conflict"]
    tracked = [src for src, _dst, state in plan if state == "tracked"]
    done = sum(1 for _s, _d, state in plan if state == "done")
    for source, destination in conflicts:
        print(f"  CONFLICT {_label(source)} -> {_label(destination)} already exists",
              file=sys.stderr)
    for source in tracked:
        print(f"  SKIPPED  {_label(source)} holds tracked files; git does not walk the symlink a "
              f"fold leaves behind", file=sys.stderr)
    unresolved = len(conflicts) + len(tracked)
    if not pending:
        print(f"nothing to fold ({done} already folded, {len(conflicts)} conflicts, "
              f"{len(tracked)} held by git)")
        return 1 if unresolved else 0
    print(f"{'folding' if apply else 'would fold'} {len(pending)} directories "
          f"({done} already folded)")
    failures = 0
    for source, destination, state in pending:
        note = "  (merge)" if state == "merge" else ""
        print(f"  {_label(source):<48} -> {_label(destination)}{note}")
        if not apply:
            continue
        try:
            fold(source, destination)
        except OSError as exc:
            failures += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
    if apply:
        print("\nold names are now relative symlinks, so existing citations still resolve")
    else:
        print("\ndry run -- pass --apply to fold")
    return 1 if failures or unresolved else 0


def _dedup(paths: list[Path] | None, min_bytes: int, top: int, apply: bool) -> int:
    roots = [p.resolve() for p in paths] if paths else None
    found = dedup_candidates(roots, min_bytes=min_bytes)
    groups = found["groups"]
    if not groups:
        print("no duplicated content found")
        return 0
    print(f"{len(groups)} content group(s) held under {found['files']} names, "
          f"{_human(found['reclaimable_bytes'])} reclaimable")
    if found["sealed_trees_skipped"]:
        print(f"  ({found['sealed_trees_skipped']} mode-verified tree(s) skipped: sharing an inode "
              f"would break their own integrity check)")
    for size, same in groups[:top]:
        print(f"\n  {_human(size * (len(same) - 1)):>12}  {len(same)}x {_human(size)}")
        for path in same[:4]:
            print(f"                {_label(path)}")
        if len(same) > 4:
            print(f"                ... and {len(same) - 4} more")
    if not apply:
        print("\ndry run -- pass --apply to collapse these onto one copy each")
        return 0
    result = dedup(groups)
    print(f"\nreclaimed {_human(result['reclaimed_bytes'])}: {result['names']} names released "
          f"{_human(result['released_bytes'])} and {_human(result['stored_bytes'])} moved into the "
          f"store as the one remaining copy.\nEvery name still resolves to the same bytes, now "
          f"read-only and shared.")
    return 0


def _retain(keep: int, match: str | None, apply: bool) -> int:
    plan = retention_plan(keep, match)
    if not plan["drop_units"]:
        print(f"keeping {keep} per group drops nothing")
        return 0
    print(f"{'removing' if apply else 'would remove'} {plan['drop_units']} unit(s), "
          f"{_human(plan['drop_bytes'])}, keeping the {keep} newest per group")
    failures = 0
    for group, row in sorted(plan["groups"].items(), key=lambda kv: -sum(
            d["bytes"] for d in kv[1]["drops"])):
        if not row["drops"]:
            continue
        cost = sum(d["bytes"] for d in row["drops"])
        undated = f", {row['undated']} undated kept" if row["undated"] else ""
        print(f"\n  {group}  ({len(row['drops'])} of {row['units']} dated units, "
              f"{_human(cost)}{undated})")
        for drop in row["drops"][:6]:
            print(f"      {_human(drop['bytes']):>12}  {drop['name']}")
        if len(row["drops"]) > 6:
            print(f"      ... and {len(row['drops']) - 6} more")
        if not apply:
            continue
        for drop in row["drops"]:
            try:
                _remove(Path(drop["path"]))
            except OSError as exc:
                failures += 1
                print(f"      FAILED {drop['name']}: {exc}", file=sys.stderr)
    if not apply:
        print("\ndry run -- pass --apply to remove. Ordering is by the timestamp in each unit's\n"
              "name, never by mtime; a unit carrying no timestamp is never dropped.")
    return 1 if failures else 0


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

    org = sub.add_parser("organize", help="fold the tree into the shape the contract declares "
                                         "(dry run by default)")
    org.add_argument("--apply", action="store_true",
                     help="actually move; without it nothing is touched")

    dd = sub.add_parser("dedup", help="collapse files that hold bytes another file already holds "
                                     "(dry run by default)")
    dd.add_argument("paths", nargs="*", type=Path,
                    help="where to look (default: every declared scan root)")
    dd.add_argument("--min-bytes", type=int, default=1 << 20,
                    help="ignore files smaller than this (default 1 MiB)")
    dd.add_argument("--top", type=int, default=15, help="how many duplicate groups to list")
    dd.add_argument("--apply", action="store_true",
                    help="actually re-point the names; without it nothing is touched")

    ret = sub.add_parser("retain", help="what a retention depth would drop (dry run by default)")
    ret.add_argument("--keep", type=int, required=True,
                     help="how many of the newest units to keep per group")
    ret.add_argument("--match", help="only units whose name contains this (e.g. phase2)")
    ret.add_argument("--apply", action="store_true",
                     help="actually remove; without it nothing is touched")

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

    if args.command == "organize":
        return _organize(args.apply)

    if args.command == "dedup":
        return _dedup(args.paths or None, args.min_bytes, args.top, args.apply)

    if args.command == "retain":
        return _retain(args.keep, args.match, args.apply)

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
