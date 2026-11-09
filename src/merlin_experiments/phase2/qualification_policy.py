"""Own the frozen package tool closure, not the host operating system.

The enclosing qualification declaration authenticates this record. V4 owns copied bytes and
support aliases; the extra inventory binds modes and empty directories as execution inputs.
System mounts supplied by bwrap remain runtime dependencies. DNS service state is omitted
because package execution unshares the network.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import asdict
from pathlib import Path

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin_experiments.phase2 import campaign


def _digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inventory(root: Path) -> list[dict]:
    rows = []
    for path in [root, *sorted(root.rglob("*"))]:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)) or mode & 0o222:
            raise campaign.CampaignGateError(f"frozen execution resource is writable, linked, or special: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": "dir" if stat.S_ISDIR(mode) else "file",
                "mode": stat.S_IMODE(mode),
            }
        )
    return rows


def _surface_record(surface: AnswerSurface) -> dict:
    return {**asdict(surface), "path": str(surface.path), "grantable": list(surface.grantable)}


def _captured_surfaces(sources: list[Path], surfaces: tuple[AnswerSurface, ...]) -> list[AnswerSurface]:
    """Capture lexical and canonical answer ownership while original links still exist."""
    owners = [(s, p) for s in surfaces for p in {s.path.absolute(), s.path.resolve()}]
    result = [AnswerSurface(s.label, p, s.kind, s.origin, s.grantable) for s, p in owners]

    def visit(path: Path, ancestry: tuple[Path, ...]) -> None:
        canonical = path.resolve(strict=True)
        if canonical in ancestry:
            raise campaign.CampaignGateError(f"cyclic tool resource alias: {path}")
        for surface, owner in owners:
            if canonical == owner or canonical.is_relative_to(owner):
                if any(canonical == owner / sub or canonical.is_relative_to(owner / sub) for sub in surface.grantable):
                    continue
                result.append(
                    AnswerSurface(
                        surface.label,
                        path,
                        "dir" if path.is_dir() else "file",
                        surface.origin,
                        surface.grantable if canonical == owner else (),
                    )
                )
                if canonical != owner or not surface.grantable:
                    return  # The whole lexical subtree is already withheld, including its aliases.
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child, (*ancestry, canonical))

    for source in sources:
        visit(source, ())
    return result


def freeze(root: Path, target, inputs: campaign.PackageSandboxInputs) -> dict:
    """Freeze the selected tool/harness mounts and all inputs to their sandbox policy."""
    try:
        return _freeze(root, target, inputs)
    except campaign.CampaignGateError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise campaign.CampaignGateError(f"cannot freeze execution policy: {exc}") from exc


def _freeze(root: Path, target, inputs: campaign.PackageSandboxInputs) -> dict:
    root = Path(root).resolve()
    ws = root / "inputs/execution/workspace"
    if BW.bundle_snapshot_root(ws).exists() or BW.bundle_snapshot_root(ws).is_symlink():
        raise campaign.CampaignGateError("execution resources already exist; restore their authenticated policy")
    repo = inputs.paths.repo.absolute()
    selected = {"paths": inputs.paths, "sim": inputs.sim, "harness": inputs.harness}
    explicit = [
        inputs.paths.uv_python,
        inputs.paths.venv,
        inputs.paths.llvm,
        inputs.paths.clang_bin,
        inputs.paths.clang_resource,
        inputs.paths.compat_lib,
        *inputs.sim.bind_paths,
    ]
    if inputs.harness:
        explicit.append(inputs.harness)
    for value in explicit:
        if not value or not Path(value).is_absolute() or ".." in Path(value).parts or not Path(value).exists():
            raise campaign.CampaignGateError(f"selected execution resource is missing or unsafe: {value!r}")
    raw = TC.toolchain_binds(target, **selected, memory_dir="")
    sources: list[Path] = []
    mounts = []
    tail = []
    i = 0
    while i < len(raw):
        if raw[i] == "--ro-bind":
            source, destination = raw[i + 1 : i + 3]
            if source != TC.RESOLVE_DIR:
                path = Path(source).absolute()
                sources.append(path)
                mounts.append(
                    {"lexical": str(path), "owner": str(path.resolve(strict=True)), "destination": destination}
                )
            i += 3
        elif raw[i] == "--unsetenv":
            tail.extend(raw[i : i + 2])
            i += 2
        else:
            raise campaign.CampaignGateError(f"unsupported selected execution mount option: {raw[i]}")
    if not mounts or any(str(Path(value).absolute()) not in {row["lexical"] for row in mounts} for value in explicit):
        raise campaign.CampaignGateError("selected execution resource was omitted from the mount policy")
    captured = _captured_surfaces(sources, inputs.surfaces)
    probes = tuple(TC.required_tool_probes(target, paths=inputs.paths, sim=inputs.sim))
    if not probes:
        raise campaign.CampaignGateError("frozen package sandbox derives zero required tool probes")
    bundle = {"allowed": [{"path": source} for source in sorted({str(path) for path in sources})]}
    ws.mkdir(parents=True, exist_ok=True)
    try:
        marker = BW.materialize_bundle_inputs(
            ws,
            bundle,
            repo=repo,
            descriptor=target if hasattr(target, "backend_package") else None,
            private_sources=tuple({surface.path for surface in captured}),
        )
        snapshot_root = BW.bundle_snapshot_root(ws)
        grants = {row["destination"]: row["snapshot"] for row in marker["grants"]}
        frozen_surfaces = list(captured)
        for mount in mounts:
            source = Path(mount["lexical"])
            mount["source"] = grants[str(source)]
            for surface in captured:
                if surface.path == source or surface.path.is_relative_to(source):
                    frozen_surfaces.append(
                        AnswerSurface(
                            surface.label,
                            snapshot_root / mount["source"] / surface.path.relative_to(source),
                            surface.kind,
                            surface.origin,
                            surface.grantable,
                        )
                    )
                elif source.is_relative_to(surface.path):
                    if any(
                        source == surface.path / sub or source.is_relative_to(surface.path / sub)
                        for sub in surface.grantable
                    ):
                        continue
                    frozen_surfaces.append(
                        AnswerSurface(
                            surface.label,
                            snapshot_root / mount["source"],
                            "dir" if source.is_dir() else "file",
                            surface.origin,
                            (),
                        )
                    )
        frozen_surfaces.extend(BW.snapshot_support_surfaces(ws, marker))
        # Finish alias projection NOW. The generic backend masking helper resolves live paths;
        # replay must use only these captured spellings, even after originals are retargeted.
        source_surfaces = tuple(frozen_surfaces)
        for mount in mounts:
            source = snapshot_root / mount["source"]
            destination = Path(mount["destination"])
            for surface in source_surfaces:
                if surface.path == source or surface.path.is_relative_to(source):
                    frozen_surfaces.append(
                        AnswerSurface(
                            surface.label,
                            destination / surface.path.relative_to(source),
                            surface.kind,
                            surface.origin,
                            surface.grantable,
                        )
                    )
                elif source.is_relative_to(surface.path):
                    if any(
                        source == surface.path / sub or source.is_relative_to(surface.path / sub)
                        for sub in surface.grantable
                    ):
                        continue
                    frozen_surfaces.append(
                        AnswerSurface(
                            surface.label, destination, "dir" if source.is_dir() else "file", surface.origin, ()
                        )
                    )
        argv = [
            part
            for mount in mounts
            for part in ("--ro-bind", str(snapshot_root / mount["source"]), mount["destination"])
        ] + tail
        surface_rows = {_digest(_surface_record(surface)): _surface_record(surface) for surface in frozen_surfaces}
        record = {
            "version": 1,
            "workspace": "inputs/execution/workspace",
            "repo": str(repo),
            "bundle": bundle,
            "snapshot": BW.snapshot_record(ws),
            "inventory": _inventory(snapshot_root),
            "mounts": mounts,
            "argv": argv,
            "tail": tail,
            "probes": [asdict(probe) for probe in probes],
            "env_prefix": TC.sandbox_env(target, ws, **selected),
            "surfaces": list(surface_rows.values()),
            "inputs": {
                "paths": {**asdict(inputs.paths), "repo": str(repo)},
                "sim": asdict(inputs.sim),
                "harness": inputs.harness,
            },
            "runtime_system_mounts": "bwrap base operating-system mounts; network unshared; DNS service state omitted",
        }
        record["sha256"] = _digest(record)
        restore(root, record)
        return record
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise campaign.CampaignGateError(f"cannot freeze execution policy: {exc}") from exc


def restore(root: Path, record: dict) -> campaign.FrozenPackageSandboxInputs:
    """Verify declaration-owned policy and payload without reading original resources."""
    try:
        root = Path(root).resolve(strict=True)
        if record.get("version") != 1 or record.get("workspace") != "inputs/execution/workspace":
            raise ValueError("unsupported frozen execution policy")
        if record.get("sha256") != _digest({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("execution policy digest mismatch")
        ws = root / record["workspace"]
        if ws.resolve(strict=True) != ws:
            raise ValueError("execution workspace path is aliased")
        snapshot_root = BW.bundle_snapshot_root(ws)
        if snapshot_root.resolve(strict=True) != snapshot_root:
            raise ValueError("execution snapshot path is aliased")
        repo = Path(record["repo"])
        if not repo.is_absolute() or ".." in repo.parts:
            raise ValueError("invalid original repository identity")
        marker = BW.verify_snapshot_binding(ws, record["bundle"], record["snapshot"], repo=repo)
        BW.verify_bundle_snapshot(ws, record["bundle"], repo=repo)
        BW.snapshot_support_surfaces(ws, marker)
        if _inventory(snapshot_root) != record["inventory"]:
            raise ValueError("execution resource mode or directory inventory changed")
        grants = {row["destination"]: row["snapshot"] for row in marker["grants"]}
        for mount in record["mounts"]:
            if grants.get(mount["lexical"]) != mount["source"] or mount["destination"] != mount["lexical"]:
                raise ValueError("execution mount source mapping differs from snapshot")
            if not Path(mount["owner"]).is_absolute():
                raise ValueError("execution mount owner is unsafe")
        tail = record["tail"]
        if len(tail) % 2 or any(
            tail[i] != "--unsetenv" or tail[i + 1] not in TC.NESTED_SESSION_VARS for i in range(0, len(tail), 2)
        ):
            raise ValueError("unsafe frozen execution environment argv")
        expected = [
            part
            for mount in record["mounts"]
            for part in ("--ro-bind", str(snapshot_root / mount["source"]), mount["destination"])
        ] + tail
        if expected != record["argv"]:
            raise ValueError("execution argv differs from captured mounts")
        surfaces = tuple(
            AnswerSurface(
                row["label"],
                Path(row["path"]),
                row["kind"],
                "hidden" if row["origin"] == "backend" else row["origin"],
                tuple(row["grantable"]),
            )
            for row in record["surfaces"]
        )
        probes = tuple(TC.ToolProbe(**row) for row in record["probes"])
        if not probes or not isinstance(record["env_prefix"], str):
            raise ValueError("execution probes or environment are absent")
        return campaign.FrozenPackageSandboxInputs(
            root, record, repo, tuple(expected), probes, record["env_prefix"], surfaces
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise campaign.CampaignGateError(f"frozen execution policy verification failed: {exc}") from exc
