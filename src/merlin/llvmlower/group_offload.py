"""The whole-model device route: one device call per CLOSED COMPUTE GROUP.

:mod:`.device_offload` moves a *contraction*. That is the right unit for a matrix instruction and the
wrong one for a whole model: a captured layer is a contraction plus the bias, the requantize, the
activation and the pooling its readout absorbs, and routing only the contraction leaves every one of
those on the host. The model then runs as a device call per matmul interleaved with host epilogues --
correct, and nothing like the program a hand-written whole-model schedule emits.

This module routes the unit the planner already forms. :func:`~..xdsl_dialects.lowering.compute_groups.form_groups`
closes a contraction together with the stages the target's readout takes;
:func:`~..xdsl_dialects.lowering.outline.outline_dispatches` already turns each such group into one
``func.func`` plus one ``func.call``; and
:func:`~..xdsl_dialects.lowering.group_command.program` already restates a group as the generator
entry a backend package builds from -- carrying ``epilogue``, ``acc_scale``, the convolution
geometry and the committed output type. All three existed. What did not was the step that puts them
in a line, so the call the outliner emits is answered by a kernel the target's own package emitted
from the group's REAL entry rather than from a bare ``M x K x N``.

**Every group is accounted for, by name.** A group is on the device, on the host, or refused with the
reason it could not be stated -- and :func:`census` reports the three counts separately so a mixed
program can never be read as one mechanism. :func:`require_every_group_accounted` is the raising
form: it is a defect for a group to be in none of the three, because that is what "silently dropped"
looks like from the outside.

Nothing here knows a target. The device is a parameter; the entries are derived from the module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GroupCall",
    "GroupOffload",
    "HostRegion",
    "ON_DEVICE",
    "ON_HOST",
    "REFUSED",
    "Refusal",
    "SCHEMA",
    "capsule_entry_for",
    "census",
    "emit_group_artifacts",
    "plan",
    "require_every_group_accounted",
]

SCHEMA = "group_offload_v1"

#: The three ways a compute group leaves this pass. They are named because they are DIFFERENT
#: MECHANISMS, and a census that summed them would let a model that ran a third of its layers on the
#: device read exactly like one that ran all of them.
ON_DEVICE = "device_group"
ON_HOST = "host_region"
REFUSED = "refused"

#: What a group-derived capsule entry says about where it came from. The generator requires a role
#: and a reference on every entry, and "derived from a model" is the honest one here: these are not
#: synthesized shapes, they are the layers the capture contains.
SOURCE_ROLE = "model_derived"


@dataclass(frozen=True)
class GroupCall:
    """One closed group, the call it became, and the program it asks a backend for."""

    index: int
    symbol: str
    #: The REAL ``GroupProgram.entry`` -- op, extents, epilogue, multiplier, committed dtype.
    entry: dict[str, Any]
    #: ``GroupProgram.to_dict()``: what a prepack step needs to honour the statement.
    program: dict[str, Any]
    placement: str

    @property
    def epilogue(self) -> list[str]:
        return [str(stage) for stage in (self.entry.get("epilogue") or ())]


@dataclass(frozen=True)
class HostRegion:
    """A group the planner placed on the host, and the reason it gave."""

    index: int
    symbol: str | None
    why: str


@dataclass(frozen=True)
class Refusal:
    """A group that could not be stated as a device program. NAMED, never dropped."""

    index: int
    symbol: str | None
    name: str
    why: str


@dataclass(frozen=True)
class GroupOffload:
    """What the route did to every group of one module."""

    device: str
    n_groups: int = 0
    module: Any = None
    dispatches: tuple = ()
    calls: tuple[GroupCall, ...] = ()
    host: tuple[HostRegion, ...] = ()
    refused: tuple[Refusal, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    grouping: Any = field(default=None, repr=False, compare=False)
    materialization_identity: str | None = field(default=None, repr=False)

    @property
    def accounted(self) -> int:
        return len(self.calls) + len(self.host) + len(self.refused)

    def entries(self) -> dict[str, dict[str, Any]]:
        """``symbol -> the group's entry``, the map a device build consumes instead of synthesizing."""
        return {call.symbol: dict(call.entry) for call in self.calls}

    def census(self) -> dict[str, Any]:
        return census(self)

    def require_grouping(self, grouping) -> None:
        """Reopen this materialization's source decision and actual output bytes."""
        if self.grouping is not grouping or self.device != grouping.target:
            raise ValueError("offload materialization belongs to another group plan")
        grouping.require_current(grouping.source, grouping.target, weight_args=grouping.weight_args)
        if self.materialization_identity != _materialization_identity(self.module, self.dispatches):
            raise ValueError("offload materialization changed after grouping")


def _materialization_identity(module, dispatches) -> str:
    from merlin.transforms.certificates import ir_digest

    return json.dumps(
        {"module": ir_digest(module), "dispatches": [asdict(info) for info in dispatches]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _symbol_of(dispatches, index: int) -> str | None:
    for info in dispatches:
        if getattr(info, "group", None) == index:
            return str(info.symbol)
    return None


def plan(
    module,
    device: str,
    *,
    weight_args=None,
    select=None,
    oracle=None,
    model: str = "",
    grouping=None,
) -> GroupOffload:
    """Route every closed compute group of ``module`` onto ``device``, one call per group.

    ``select`` is the placement decision, passed in rather than taken here -- the same contract
    :mod:`.device_offload` keeps, and for the same reason: a pass that decided for itself would be a
    second placement authority, and two authorities disagree. ``None`` routes every group the planner
    already closed on this device, which is the planner's decision and not this pass's.

    The module is OUTLINED, not mutated in place: the returned ``module`` is the driver plus one
    kernel function per dispatch, with each accelerator group's kernel carrying ``merlin.group`` and
    ``merlin.placement``. That is the module a device build compiles against, and the calls in its
    driver are the calls the device objects have to answer.
    """
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    if grouping is None:
        groups = CG.form_groups(module, device, oracle=oracle)
    else:
        from merlin.compiler.planning.groups import GroupPlan

        if type(grouping) is not GroupPlan:
            raise TypeError("offload materialization requires a live group plan")
        grouping.require_current(module, device, weight_args=weight_args)
        if oracle is not None and oracle is not grouping.oracle:
            raise ValueError("offload cannot replace the group plan's selected oracle")
        groups = grouping.groups
    outlined = outline_dispatches(module, groups=groups)

    calls: list[GroupCall] = []
    host: list[HostRegion] = []
    refused: list[Refusal] = []
    for group in groups:
        symbol = _symbol_of(outlined.dispatches, group.index)
        name = _group_name(group, model=model)
        if group.placement == CG.HOST or group.root is None:
            host.append(
                HostRegion(
                    index=group.index,
                    symbol=symbol,
                    why=str(group.reason or group.refusal or "the planner placed this region on the host"),
                )
            )
            continue
        if select is not None and not select(group.root):
            # A REFUSAL BY THE PLACEMENT IS STILL A NAMED OUTCOME. Silently leaving it in the host
            # bucket with the groups nobody could take would make a decision indistinguishable from
            # a gap, which is the whole failure this census exists to end.
            host.append(
                HostRegion(index=group.index, symbol=symbol, why="the placement did not select this group's root")
            )
            continue
        try:
            stated = GC.program(group, weight_args=weight_args, name=name)
        except CG.NoCapsuleForm as error:
            refused.append(Refusal(index=group.index, symbol=symbol, name=name, why=str(error)))
            continue
        if symbol is None:
            # The outliner emits a kernel for every group that holds a compute op; a closed group
            # with no symbol means the two disagree about what this module contains, and building
            # against that would link a call nothing defines.
            refused.append(
                Refusal(
                    index=group.index,
                    name=name,
                    symbol=None,
                    why="the outliner emitted no dispatch for this group, so there is no call to answer",
                )
            )
            continue
        calls.append(
            GroupCall(
                index=group.index,
                symbol=symbol,
                entry=capsule_entry_for(stated.entry, index=group.index, name=name, device=device, model=model),
                program=stated.to_dict(),
                placement=str(group.placement),
            )
        )
    if grouping is not None:
        grouping.require_current(module, device, weight_args=weight_args)
    return GroupOffload(
        device=device,
        n_groups=len(groups),
        module=outlined.module,
        dispatches=tuple(outlined.dispatches),
        calls=tuple(calls),
        host=tuple(host),
        refused=tuple(refused),
        grouping=grouping,
        materialization_identity=(
            _materialization_identity(outlined.module, outlined.dispatches) if grouping is not None else None
        ),
    )


def _group_name(group, *, model: str = "") -> str:
    """A stable, symbol-safe name for one group. Position is part of it: two layers of identical
    shape are two calls in one program, and a name that collapsed them would make a refusal
    ambiguous about WHICH layer refused."""
    stem = f"{model}_" if model else ""
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in stem)
    return f"{safe}group{int(group.index)}"


def capsule_entry_for(entry, *, index: int, name: str, device: str, model: str = "") -> dict[str, Any]:
    """The group's entry, completed with what the corpus generator requires of every entry.

    The entry itself is carried VERBATIM. That is the point of this route: ``epilogue``,
    ``acc_scale``, the convolution geometry and the committed output type are the group's, and a
    build that rebuilt them from the extents alone would emit a kernel that computes a different
    function than the layer it stands for.
    """
    out = dict(entry)
    out["name"] = name
    out.setdefault("kind", "layer")
    out.setdefault("cat", "layers")
    out.setdefault("label", "dev")
    out["source_role"] = SOURCE_ROLE
    out["source_reference"] = f"compute group {int(index)} of {model or 'a captured model'} on {device}"
    return out


def census(offload: GroupOffload) -> dict[str, Any]:
    """Which groups went which way, as three counts that are never summed.

    ``uniform`` is the question a reader actually has: did the whole model take one mechanism? A
    program with one host region and seventy device calls is a DIFFERENT program from seventy-one
    device calls, and a report that printed only "71 groups" would not distinguish them.
    """
    mechanism = {
        ON_DEVICE: len(offload.calls),
        ON_HOST: len(offload.host),
        REFUSED: len(offload.refused),
    }
    return {
        "schema": SCHEMA,
        "device": offload.device,
        "groups": offload.n_groups,
        "accounted": offload.accounted,
        "mechanism": mechanism,
        "uniform": sum(1 for count in mechanism.values() if count) == 1,
        "calls": [
            {
                "group": call.index,
                "symbol": call.symbol,
                "op": call.entry.get("op"),
                "epilogue": call.epilogue,
                "output_dtype": call.entry.get("output_dtype"),
                "placement": call.placement,
            }
            for call in offload.calls
        ],
        "host_regions": [{"group": region.index, "why": region.why} for region in offload.host],
        "refusals": [{"group": r.index, "name": r.name, "why": r.why} for r in offload.refused],
    }


class SilentGroupError(RuntimeError):
    """A group left this pass in none of the three buckets."""


def require_every_group_accounted(offload: GroupOffload) -> None:
    """Raise unless every group is on the device, on the host, or refused by name.

    Not belt-and-braces. The three buckets are built by a loop over the groups, so a ``continue``
    added later to skip an awkward case would drop it out of all three -- and the census would then
    report a smaller model than the one being compiled, with nothing to point at.
    """
    if offload.accounted != offload.n_groups:
        raise SilentGroupError(
            f"{offload.n_groups} compute group(s) were formed but only {offload.accounted} are accounted "
            f"for ({len(offload.calls)} on {offload.device!r}, {len(offload.host)} on the host, "
            f"{len(offload.refused)} refused); the rest left this pass unnamed"
        )


def emit_group_artifacts(
    offload: GroupOffload,
    *,
    package_dir: str | Path,
    binding,
    workdir: str | Path,
    timeout: int = 300,
) -> dict[str, Any]:
    """Ask ``package_dir``'s backend for a device program per routed group.

    This is the question "can this backend emit THIS MODEL", asked one layer at a time. A package
    that declines a group is recorded BY NAME with the reason it gave, because "the model does not
    compile" names nothing a backend author can act on.

    ⚠️ **A returncode of zero is not an answer here, and reading it as one measured 71 of 71.** The
    entrypoint that emits the target artifact reports success and prints an EMPTY entry function for
    a layer it refused; the refusal is in the command buffer, under the ABI's own ``declined`` key,
    and on the lowering entrypoint's stderr. So the verdict below is read off the command buffer --
    a program is emitted when it carries commands -- and a buffer with none is a decline whatever
    the exit status said. Measured on a captured ResNet-50: 54 of 71 groups emit, 16 decline for an
    operation the package has no lowering for, one for an accumulator that cannot hold a fused
    pool's rows. Exit status alone called all 71 a success.
    """
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.package_runtime import load_package, run_entrypoint

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    package = load_package(str(package_dir))

    emitted: list[dict[str, Any]] = []
    declined: dict[str, str] = {}
    for call in offload.calls:
        name = str(call.entry["name"])
        try:
            _capsule, iface = CS.build(dict(call.entry), binding)
        except Exception as error:  # noqa: BLE001 -- recorded per group, never swallowed
            declined[name] = f"interface capsule: {type(error).__name__}: {str(error)[:300]}"
            continue
        source = work / f"{name}.iface.mlir"
        source.write_text(iface, encoding="utf-8")
        buffer_path = work / f"{name}.cmdbuf.json"
        result = run_entrypoint(package, "emit_command_buffer", source, buffer_path, timeout=timeout)
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            declined[name] = f"package exited {result.returncode}: {stderr[-300:]}"
            continue
        if not buffer_path.is_file():
            declined[name] = f"package wrote no command buffer{f'; {stderr[-260:]}' if stderr else ''}"
            continue
        try:
            buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
        except ValueError as error:
            declined[name] = f"command buffer unreadable: {error}"
            continue
        if not buffer.get("commands"):
            refusal = (buffer.get("declined") or {}).get("reason")
            declined[name] = str(
                refusal or f"the command buffer carries no commands{f'; {stderr[-200:]}' if stderr else ''}"
            )
            continue
        emitted.append(
            {
                "group": call.index,
                "name": name,
                "symbol": call.symbol,
                "op": call.entry.get("op"),
                "epilogue": call.epilogue,
                "commands": len(buffer["commands"]),
                "command_buffer": str(buffer_path),
            }
        )
    return {
        "schema": SCHEMA,
        "device": offload.device,
        "routed": len(offload.calls),
        "emitted": emitted,
        "declined_by_package": declined,
    }
