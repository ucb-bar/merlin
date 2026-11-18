"""The ``merlin-storage report`` model: what the generated-output root costs and how it is shaped.

Bytes counted once per inode, the content store's orphans and pending input closures, declared
caches, layout drift against the storage contract, and per-experiment unit costs. Nothing here
changes the disk; :mod:`merlin.common.storage.cli` acts on what this reports.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from merlin.common.paths import artifacts_dir, build_dir, out_dir, runs_dir
from merlin.common.storage import content_store, retention
from merlin.common.storage.retention import declared_concerns, out_roots, protection_reasons, scan_roots, unit_timestamp

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
            continue  # vanished or unreadable mid-walk; report what we saw
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
    # content_store.store_root() creates the store for writers. Reports and dry runs must not.
    override = os.environ.get(content_store.LOCATION_ENV)
    return Path(override) if override and override.strip() else content_store.default_root()


def store_orphans(root: Path | None = None) -> tuple[list[Path], int]:
    """Store objects nothing references any more -- see merlin.common.storage.content_store.orphans."""
    found, _ = content_store.orphans(root if root is not None else store_root())
    safe = []
    size = 0
    for path in found:
        try:
            if not protection_reasons(path):
                size += path.stat(follow_symlinks=False).st_size
                safe.append(path)
        except OSError:
            continue  # another session removed an orphan while this read-only report ran
    return safe, size


def pending_snapshot_status(roots: list[Path] | None = None) -> list[dict]:
    """Incomplete closures, distinguishing terminal owners from active or unclassified copies."""
    found: list[dict] = []
    for root in roots if roots is not None else scan_roots():
        if not root.is_dir():
            continue
        for path in root.rglob(_SNAPSHOT_DIR + _PENDING_SUFFIX):
            if path.is_dir() and not path.is_symlink():
                found.append({"path": str(path), "blockers": protection_reasons(path, require_terminal=True)})
    return found


def pending_snapshots(roots: list[Path] | None = None) -> tuple[list[Path], int]:
    """Only incomplete closures whose terminal owner and protections permit reclamation."""
    found = [Path(row["path"]) for row in pending_snapshot_status(roots) if not row["blockers"]]
    return found, sum(measure(path).occupied for path in found)


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
        if protection_reasons(path):
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
    for root in roots if roots is not None else scan_roots():
        if not root.is_dir():
            continue
        for path in root.rglob(_SNAPSHOT_DIR):
            if path.is_dir() and not path.is_symlink():
                count += 1
                measure(path, usage=usage)
    return {
        "snapshots": count,
        "apparent_bytes": usage.apparent,
        "occupied_bytes": usage.occupied,
        "files": usage.files,
    }


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
            report["stray_roots"].append({"name": path.name, "bytes": usage.occupied, "files": usage.files})
    for path in _safe_dirs(artifacts_dir()):
        usage = measure(path)
        row = {"name": path.name, "bytes": usage.occupied, "files": usage.files, "units": len(_safe_dirs(path))}
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
    from merlin.common.record_formats import load_yaml  # noqa: PLC0415

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
            continue  # a placeholder manifest declares nothing, so nothing drifts
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
            rows.append(
                {"group": group, "unit": name, "declared": len(declared), "bytes": undeclared, "unit_bytes": total}
            )
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
    rows = [
        {"group": group, "units": total, "undated": odd} for group, (total, odd) in counts.items() if 0 < odd < total
    ]
    rows.sort(key=lambda r: -r["undated"])
    return rows


def _print_layout(report: dict, top: int) -> None:
    stray, undeclared = report["stray_roots"], report["undeclared_concerns"]
    declared = report["declared_concerns"]
    roots = out_roots()
    print(f"out/ roots: {len(roots)} declared ({', '.join(roots)}), {len(stray)} undeclared")
    for row in stray:
        print(f"  [stray root]  {_human(row['bytes']):>10}  {row['files']:>7,} files  out/{row['name']}")
    if stray:
        print(
            "  -> generated output belongs under one of the three roots; anything else is a root\n"
            "     the convention retired, and the write-guard hook only blocks paths it knows."
        )

    print(f"\nout/artifacts concerns: {len(declared)} declared, {len(undeclared)} undeclared")
    declared_bytes = sum(r["bytes"] for r in declared)
    undeclared_bytes = sum(r["bytes"] for r in undeclared)
    print(f"  declared   {_human(declared_bytes):>10}")
    print(
        f"  undeclared {_human(undeclared_bytes):>10}   "
        f"({len(undeclared)} dirs, {sum(r['units'] for r in undeclared):,} units)"
    )
    if undeclared:
        print(f"\n  largest undeclared (top {top}):")
        for row in undeclared[:top]:
            print(f"    {_human(row['bytes']):>10}  {row['units']:>5} units  {row['name']}")
        print(
            "\n  Each needs one of: a fold in src/merlin/resources/contracts/storage.yaml, an entry in that\n"
            "  file's concern roster, or retirement. `merlin-storage organize` applies the folds."
        )

    odd = report.get("unconventional_units") or []
    if odd:
        total = sum(r["undated"] for r in odd)
        print(
            f"\ngroups written both ways: {total:,} undated units across {len(odd)} groups (top {min(top, len(odd))})"
        )
        for row in odd[:top]:
            print(f"  {row['undated']:>6} of {row['units']:<6} {row['group']}")
        print(
            "\n  These trees hold conventionally-named units AND directories with no <TS> token,\n"
            "  so the same concern is being written through new_product() and by something that\n"
            "  just made a directory. An undated unit cannot be ordered against its siblings, so no\n"
            "  retention depth can place it. That is a producer fix, not a disk decision."
        )

    payload = report.get("undeclared_payload") or []
    if payload:
        total = sum(r["bytes"] for r in payload)
        print(
            f"\nbytes inside a product its own manifest does not list: {_human(total)} across "
            f"{len(payload)} units (top {min(top, len(payload))})"
        )
        for row in payload[:top]:
            share = row["bytes"] / row["unit_bytes"] * 100 if row["unit_bytes"] else 0
            print(
                f"  {_human(row['bytes']):>12} of {_human(row['unit_bytes']):>12} ({share:3.0f}%)  "
                f"{row['declared']:>4} declared  {row['group']}/{row['unit']}"
            )
        print(
            "\n  A manifest is the product's statement of what it contains, so this needs no\n"
            "  threshold: the producer already declared the answer. Undeclared bulk is build\n"
            "  output that belongs under out/build/, referenced by digest rather than embedded."
        )


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
    return name.startswith(_VERSION_PREFIX) and len(name) > 1 and name[1:].isdigit()


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
            continue  # regenerable by convention; priced as a cache instead
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
        return sorted(
            p for p in parent.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith((".", "__"))
        )
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
    print(
        f"{total_units:,} experiment unit(s){label}, {_human(total_bytes)} "
        f"(top {len(shown)} of {len(rows)} groups by total)\n"
    )
    print(f"  {'group':<46} {'units':>6} {'total':>12} {'mean/unit':>12} {'files':>10}")
    for group, row in shown:
        print(
            f"  {group:<46} {row['units']:>6} {_human(row['bytes']):>12} "
            f"{_human(row['mean_bytes']):>12} {row['files']:>10,}"
        )
    print(
        "\nA group that is large with a SMALL mean is large by accumulation -- that is a retention\n"
        "decision, not a producer bug. A large mean is the producer writing too much per run."
    )
    biggest = max(rows, key=lambda kv: kv[1]["mean_bytes"])
    print(
        f"\nheaviest per run: {biggest[0]} at {_human(biggest[1]['mean_bytes'])}/unit "
        f"(largest single unit: {biggest[1]['largest'][0]} "
        f"{_human(biggest[1]['largest'][1])})"
    )


def collect() -> dict:
    roots = {"runs": runs_dir(), "artifacts": artifacts_dir(), "build": build_dir()}
    report: dict = {"out_root": str(out_dir()), "roots": {}, "concerns": {}}
    try:
        disk = shutil.disk_usage(out_dir() if out_dir().exists() else Path.cwd())
        report["filesystem"] = {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}
    except OSError:
        report["filesystem"] = {}
    for name, path in roots.items():
        usage = measure(path)
        report["roots"][name] = {
            "apparent_bytes": usage.apparent,
            "occupied_bytes": usage.occupied,
            "files": usage.files,
        }
    if artifacts_dir().is_dir():
        for concern in sorted(p for p in artifacts_dir().iterdir() if p.is_dir() and not p.is_symlink()):
            usage = measure(concern)
            report["concerns"][concern.name] = {"occupied_bytes": usage.occupied, "files": usage.files}
    report["bundle_snapshots"] = snapshot_sharing()
    report["lifecycle"] = retention.inventory()
    report["protected_pending_snapshots"] = [row for row in pending_snapshot_status() if row["blockers"]]
    store = store_root()
    store_usage = measure(store)
    orphans, orphan_bytes = store_orphans(store)
    pending, pending_bytes = pending_snapshots()
    caches, cache_bytes = purgeable_caches(store)
    report["store"] = {
        "path": str(store),
        "occupied_bytes": store_usage.occupied,
        "objects": store_usage.files,
        "orphan_objects": len(orphans),
        "orphan_bytes": orphan_bytes,
    }
    report["reclaimable"] = {
        "store-orphans": {
            "count": len(orphans),
            "bytes": orphan_bytes,
            "why": "no snapshot links these objects any more",
        },
        "pending-snapshots": {
            "count": len(pending),
            "bytes": pending_bytes,
            "why": "incomplete input closures with terminal owners and no retention protections",
        },
        "caches": {
            "count": len(caches),
            "bytes": cache_bytes,
            "why": "out/artifacts/cache/<ns> is declared regenerable",
        },
    }
    return report


def _print_report(report: dict, top: int) -> None:
    fs = report.get("filesystem") or {}
    if fs:
        print(
            f"filesystem   {_human(fs['used_bytes'])} used, {_human(fs['free_bytes'])} free "
            f"of {_human(fs['total_bytes'])}"
        )
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
        print(
            f"\nper-run input closures: {snap['snapshots']} snapshots, "
            f"{_human(snap['apparent_bytes'])} declared, {_human(snap['occupied_bytes'])} on disk "
            f"({_human(saved)} shared, {share:.0f}%)"
        )
        if share < 1.0 and snap["snapshots"] > 1:
            print(
                "  NOTE: these closures share almost nothing. If their grants overlap, they "
                "predate the content store or were written with MERLIN_BUNDLE_CAS disabled."
            )
    store = report["store"]
    print(
        f"content store: {_human(store['occupied_bytes'])} in {store['objects']:,} objects "
        f"({store['orphan_objects']:,} unreferenced, {_human(store['orphan_bytes'])})"
    )

    print("\nreclaimable now (merlin-storage prune --apply <class>):")
    for name, row in report["reclaimable"].items():
        print(f"  {name:<20} {_human(row['bytes']):>12}  {row['count']:>6} items   {row['why']}")
    lifecycle = report.get("lifecycle", {})
    if lifecycle.get("status") == "unknown":
        print(f"\nstorage ownership UNKNOWN: {lifecycle['reason']}")
    else:
        protected = [
            row
            for row in lifecycle.get("paths", [])
            if row["pins"] or row["state"] in {"running", "unknown", "owner-dead"}
        ]
        print(f"\nlifecycle protects {len(protected)} recorded paths")
    pending = report.get("protected_pending_snapshots", [])
    if pending:
        print(f"  {len(pending)} incomplete input closures kept: active, pinned, tracked, or unclassified")
