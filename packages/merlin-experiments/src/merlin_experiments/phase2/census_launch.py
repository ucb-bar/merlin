"""One-ELF descriptive census under the original normal package sandbox.

This wrapper adds no compiler/model/reference grant. Missing original policy or
native producer is UNKNOWN; it never falls through to a raw host invocation.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.perf import dynamic_census as C
from merlin.runtime import _native_files as F

from . import campaign as PC
from . import normal_resource_launch as R


def _producer(elf, limits):
    expected = F.pin(elf, max_bytes=limits.max_elf_bytes)
    # Native link observations are written in the original ELF build owner.
    # A legacy memo or another artifact cannot supply that actual producer.
    records = elf.parent / "invocations"
    seen = 0
    if records.is_dir():
        for path in F.members(records, max_files=limits.max_bins):
            if path.name != "invocation.json" or not path.is_file():
                continue
            if path.stat().st_size > 1 << 20:
                raise ValueError("census link record exceeds metadata budget")
            row = json.loads(path.read_text())
            if (
                row.get("kind") == "subprocess"
                and row.get("stage") == "elf"
                and row.get("status") == "completed"
                and row.get("returncode") == 0
                and expected in row.get("outputs", ())
            ):
                I.verify(path)
                seen += 1
    if not seen:
        raise ValueError("census ELF has no actual original completed link producer")
    return expected


def _narrow_prefix(policy, workspace):
    """Preserve tools/masks; discard compiler and broad original output mounts."""
    source = list(policy.argv)
    result, index = [source[0]], 1
    while index < len(source):
        flag = source[index]
        count = (
            3
            if flag in {"--setenv", "--ro-bind", "--ro-bind-try", "--bind", "--bind-try"}
            else (2 if flag in {"--unsetenv", "--chdir", "--dir", "--tmpfs", "--proc", "--dev"} else 1)
        )
        group = source[index : index + count]
        if len(group) != count:
            raise ValueError("original census sandbox prefix is malformed")
        if flag in {"--ro-bind", "--ro-bind-try", "--bind", "--bind-try"}:
            mounted = Path(group[1])
            if mounted == policy.workspace or mounted.is_relative_to(policy.workspace) or mounted == policy.package:
                index += count
                continue
            if mounted == Path("/") or policy.workspace.is_relative_to(mounted):
                raise ValueError("original tool mount contains the broad private workspace")
            if flag in {"--bind", "--bind-try"}:
                index += count
                continue  # A readonly trace needs no writable compiler cache.
        if flag == "--chdir":
            group[1] = str(workspace)
        result.extend(group)
        index += count
    result.extend(["--bind", str(workspace), str(workspace)])
    return tuple(result)


def _private_backing(prefix, root):
    """Census-only sampled backing when no aggregate external owner is selected."""
    adjusted, positions, writable = R._scratch_plan(prefix)
    if writable != (root / "workspace",):
        raise ValueError("census namespace has a foreign writable root")
    backing = root / "scratch"
    backing.mkdir(mode=0o700)
    for number, index in reversed(tuple(enumerate(positions))):
        member = backing / str(number)
        member.mkdir(mode=0o700)
        adjusted[index : index + 2] = ["--bind", str(member), adjusted[index + 1]]
    (backing / "shm").mkdir(mode=0o700)
    adjusted.extend(
        [
            "--bind",
            str(backing / "shm"),
            "/dev/shm",
            "--remount-ro",
            "/dev",
            "--remount-ro",
            "/",
            "--unshare-user",
            "--disable-userns",
            "--assert-userns-disabled",
        ]
    )
    return tuple(adjusted), backing


def observe_dynamic_census(elf, *, target, directory, remaining_s, policy=None):
    """Reusable original public Phase 1/2 route, after its numerical L2 gate."""
    root, launch, limits = None, None, None
    try:
        limits = C.CensusLimits.for_timeout(remaining_s)
        if type(policy) is not PC.PackageSandboxPolicy or policy.coverage_gap:
            raise ValueError("original package sandbox policy is unavailable")
        PC._verify_frozen_policy(policy)
        elf, directory = F.canonical(elf), F.canonical(directory, exists=False)
        if not directory.is_relative_to(policy.workspace):
            raise ValueError("census output escaped the original workspace owner")
        before = _producer(elf, limits)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="census-box-", dir=directory))
        workspace = root / "workspace"
        workspace.mkdir(mode=0o700)
        copied = workspace / elf.name
        F.copy_member(elf, copied, budget=[1, limits.max_elf_bytes])
        if F.digest(copied) != before["sha256"]:
            raise ValueError("census streamed ELF copy differs from its original producer")
        prefix = _narrow_prefix(policy, workspace)
        narrow = replace(policy, argv=prefix, workspace=workspace)
        boundary = R.resources.active_resource_boundary()
        if boundary is None:
            prefix, backing = _private_backing(prefix, root)
            launch = C._issue_launch(
                original_owner=narrow,
                elf=elf,
                executed_elf=copied,
                target=target,
                directory=directory,
                prefix=prefix,
                limits=limits,
                writable_roots=(workspace, backing),
                owner_sources=(Path(__file__).resolve(),),
            )
            result = C.profile_elf(elf, target=target, directory=directory, limits=limits, launch=launch)
        else:
            scope = R.bounded_normal_scratch(prefix, owner=root, boundary=boundary)
            bounded = scope.__enter__()
            try:
                _, _, writable = R._scratch_plan(prefix)
                # The original helper has replaced every tmpfs with private
                # capped backing below root; sampling covers that exact root.
                launch = C._issue_launch(
                    original_owner=narrow,
                    elf=elf,
                    executed_elf=copied,
                    target=target,
                    directory=directory,
                    prefix=tuple(bounded),
                    limits=limits,
                    writable_roots=(root, *writable),
                    owner_sources=(Path(__file__).resolve(),),
                )
                result = C.profile_elf(elf, target=target, directory=directory, limits=limits, launch=launch)
            finally:
                if launch is None or launch.closed:
                    scope.__exit__(None, None, None)
                else:
                    # The mandatory BaseException aborts this job. Retain the
                    # original backing owner until actual enclosing cleanup.
                    launch._retained_scratch = scope
        PC._verify_frozen_policy(policy)
        if _producer(elf, limits) != before:
            raise ValueError("original linked census product changed during execution")
        return result
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError, TypeError) as error:
        return C.unknown("original boxed census unavailable: " + type(error).__name__, limits=limits)
    finally:
        # Never remove a live or UNKNOWN native owner's input/scratch backing.
        if root is not None and (launch is None or launch.closed):
            if launch is not None and getattr(launch, "started", False):
                # The actual invocation's one-ELF input stays in its original
                # protected job evidence so later consumers can reopen it.
                for path in root.iterdir():
                    if path != workspace:
                        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
                for path in workspace.iterdir():
                    if path != copied:
                        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
            else:
                shutil.rmtree(root)
        if launch is not None and launch.closed:
            C._ISSUED.pop(launch._issuer, None)
