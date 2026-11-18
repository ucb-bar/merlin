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

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xdsl.dialects import bufferization, func, memref
from xdsl.dialects.builtin import MemRefType, NoneAttr, StringAttr, TensorType, UnitAttr

from merlin.common.digest import sha256_file
from merlin.common.process import invocation_records as invocation_record
from merlin.common.record_formats import is_identifier
from merlin.compiler.frontend.linalg_mlir import layout_to_buffer_type, parse_mlir_text
from merlin.compiler.lower.device.sibling_source_route import borrowed_bridge_source
from merlin.runtime.harness.output_observation import ReadbackError

# ---- group_offload -----------------------------------------------------------------------------

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
    from merlin.compiler.transforms.certificates import ir_digest

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
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.compiler.partition.groups import group_command as GC
    from merlin.compiler.partition.outline import outline_dispatches

    if grouping is None:
        groups = _merlin_formation_compute_groups_1.form_groups(module, device, oracle=oracle)
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
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_2
    from merlin.development.targetgen.packaging.loading import load_package, run_entrypoint

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    package = load_package(str(package_dir))

    emitted: list[dict[str, Any]] = []
    declined: dict[str, str] = {}
    for call in offload.calls:
        name = str(call.entry["name"])
        try:
            _capsule, iface = _merlin_model_builders_2.build(dict(call.entry), binding)
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


# ---- group_profile_bindings --------------------------------------------------------------------
# Observe routed calls through structurally checked borrowed writer wrappers.
#
# This establishes a profiling boundary and pointer-call prototype only. It grants
# no writer effect, numerical, ownership, timing or device execution authority.
# Unknown transformations retain the existing profiling diagnostic.


def _functions(module):
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    result = {op.sym_name.data: op for op in functions}
    if len(result) != len(functions):
        raise ReadbackError("profile source contains duplicate function symbols")
    return result


def _borrowed(wrapper, declaration, functions):
    """Recognize the existing fresh-writer wrapper shape, without claiming effects."""
    if (
        len(wrapper.body.blocks) != 1
        or wrapper.function_type != declaration.function_type
        or wrapper.sym_visibility != StringAttr("private")
    ):
        raise ReadbackError("profile requires a private, signature-preserving writer wrapper")
    block = wrapper.body.block
    operations = list(block.ops)
    types = tuple(declaration.function_type.inputs)
    if (
        not types
        or len(types) > 64
        or len(declaration.function_type.outputs) != 1
        or any(
            not isinstance(t, TensorType) or t.encoding != NoneAttr() or any(d <= 0 for d in t.get_shape())
            for t in types
        )
    ):
        raise ReadbackError("profile writer needs complete static tensor arguments")
    attrs = declaration.properties.get("arg_attrs")
    if (
        attrs is None
        or len(attrs) != len(types)
        or wrapper.properties.get("arg_attrs") != attrs
        or any(set(a.data) != {"bufferization.access"} for a in attrs)
    ):
        raise ReadbackError("profile writer argument access declarations changed")
    accesses = [getattr(a.data.get("bufferization.access"), "data", None) for a in attrs]
    if (
        any(a not in ("read", "write") for a in accesses)
        or len(operations) < len(types) + 3
        or any(op.regions for op in operations)
    ):
        raise ReadbackError("profile writer has unknown body or access structure")
    arguments = []
    for index, (typ, access, op) in enumerate(zip(types, accesses, operations)):
        if access == "read":
            if (
                type(op) not in (bufferization.ToBufferOp, layout_to_buffer_type())
                or tuple(op.operands) != (block.args[index],)
                or op.properties.get("read_only") != UnitAttr()
            ):
                raise ReadbackError("profile writer does not retain its original read-only input")
        elif type(op) is not memref.AllocOp or op.operands:
            raise ReadbackError("profile writer does not allocate its declared fresh output")
        if tuple(value.type for value in op.results) != (MemRefType(typ.element_type, typ.get_shape()),):
            raise ReadbackError("profile writer buffer types differ from the source")
        arguments.append(op.results[0])
    # Additional private byte scratch is passed as ordinary descriptor pointers.
    extra, (call, result, returned) = operations[len(types) : -3], operations[-3:]
    for op in extra:
        if (
            type(op) is not memref.AllocOp
            or op.operands
            or len(op.results) != 1
            or not isinstance(op.results[0].type, MemRefType)
            or str(op.results[0].type.element_type) != "i8"
            or len(op.results[0].type.get_shape()) != 1
            or op.results[0].type.get_shape()[0] <= 0
        ):
            raise ReadbackError("profile writer has unknown private workspace structure")
        arguments.append(op.results[0])
    if (
        type(call) is not func.CallOp
        or call.attributes
        or set(call.properties) != {"callee"}
        or call.results
        or tuple(call.arguments) != tuple(arguments)
        or type(result) is not bufferization.ToTensorOp
        or len(result.operands) != 1
        or type(returned) is not func.ReturnOp
        or tuple(returned.arguments) != tuple(result.results)
        or result.operands[0] not in arguments[: len(types)]
        or accesses[arguments.index(result.operands[0])] != "write"
    ):
        raise ReadbackError("profile writer has no single direct borrowed full-result boundary")
    borrowed = call.callee.root_reference.data
    external = functions.get(borrowed)
    if (
        external is None
        or external.body.blocks
        or tuple(external.function_type.inputs) != tuple(a.type for a in arguments)
        or external.function_type.outputs
        or external.attributes.get("llvm.emit_c_interface") != UnitAttr()
        or set(external.attributes) != {"llvm.emit_c_interface"}
        or set(external.properties) - {"sym_name", "sym_visibility", "function_type", "arg_attrs"}
        or not borrowed.isascii()
        or not borrowed.isidentifier()
    ):
        raise ReadbackError("profile borrowed writer has no declared void ranked C interface")
    return dict(
        borrowed_symbol=borrowed, buffers=len(arguments), argument_ranks=[len(a.type.get_shape()) for a in arguments]
    )


def derive_bindings(original, selected, routed):
    """Check all original function bodies, allowing only routed callee retargeting.

    No attribute or supplied report alone selects a source-to-writer mapping.
    Original control flow, types, operands and operations must remain identical.
    New private wrapper implementations are read separately for their real ABI.
    """
    original.verify()
    selected.verify()
    before, after = _functions(original), _functions(selected)
    original_other = [op for op in original.body.block.ops if not isinstance(op, func.FuncOp)]
    selected_other = [op for op in selected.body.block.ops if not isinstance(op, func.FuncOp)]
    if (
        original.attributes != selected.attributes
        or original.properties != selected.properties
        or len(original_other) != len(selected_other)
        or any(not a.is_structurally_equivalent(b) for a, b in zip(original_other, selected_other, strict=True))
    ):
        raise ReadbackError("profile transform changed module-level declarations or attributes")
    symbols = {row["symbol"] for row in routed if row.get("symbol")}
    replacements = {}
    for name, source in before.items():
        if not source.body.blocks:
            continue
        target = after.get(name)
        if target is None:
            raise ReadbackError("profile transform removed an original source function")
        restored = target.clone()
        source_ops, target_ops = list(source.walk()), list(restored.walk())
        if len(source_ops) != len(target_ops):
            raise ReadbackError("profile transform changed source operation/control-flow structure")
        for old, new in zip(source_ops, target_ops, strict=True):
            if type(old) is func.CallOp and old.callee.root_reference.data in symbols:
                if type(new) is not func.CallOp:
                    raise ReadbackError("profile transform removed a routed call")
                origin, wrapper = old.callee.root_reference.data, new.callee.root_reference.data
                if origin in replacements and replacements[origin] != wrapper:
                    raise ReadbackError("profile routed symbol has ambiguous writer replacements")
                replacements[origin] = wrapper
                new.properties["callee"] = old.callee
        if not source.is_structurally_equivalent(restored):
            raise ReadbackError("profile transform changed original source beyond routed call retargeting")
    result = {}
    for origin, wrapper in replacements.items():
        if origin == wrapper:
            continue
        declaration = before.get(origin)
        function = after.get(wrapper)
        if declaration is None or declaration.body.blocks or function is None or wrapper in before:
            raise ReadbackError("profile route has no original declaration and fresh wrapper")
        binding = _borrowed(function, declaration, after)
        borrowed = binding["borrowed_symbol"]
        uses = [op for op in selected.walk() if type(op) is func.CallOp and op.callee.root_reference.data == borrowed]
        if len(uses) != 1 or uses[0].parent_op() is not function:
            raise ReadbackError("profile borrowed entry is shared or called outside its source wrapper")
        if any(b["borrowed_symbol"] == borrowed for b in result.values()):
            raise ReadbackError("profile borrowed entry cannot identify a unique source route")
        result[origin] = dict(binding, wrapper_symbol=wrapper)
    allowed = {b[key] for b in result.values() for key in ("wrapper_symbol", "borrowed_symbol")}
    if set(after) - set(before) - allowed:
        raise ReadbackError("profile transform added an unrecognized function")
    for name, declaration in before.items():
        if not declaration.body.blocks and name not in result:
            if name not in after or not declaration.is_structurally_equivalent(after[name]):
                raise ReadbackError("profile transform changed an unrelated original declaration")
    return result


def read_bindings(receipt, prepared, sidecar):
    """Reopen original source, current prepared IR and immutable routing identity."""
    receipt = Path(receipt)
    if not receipt.is_file():
        return {}
    receipt_bytes = receipt.read_bytes()
    record = json.loads(receipt_bytes)
    snapshot = record.get("source_snapshot")
    if snapshot is None:
        raise ReadbackError("profile transformed route has no retained original source; rebuild")
    original, selected, routing = Path(snapshot).read_bytes(), Path(prepared).read_bytes(), Path(sidecar).read_bytes()
    if (
        hashlib.sha256(original).hexdigest() != record["source_sha256"]
        or hashlib.sha256(routing).hexdigest() != record["routing_sidecar_sha256"]
    ):
        raise ReadbackError("profile original source or routing identity changed; rebuild")
    result = derive_bindings(
        parse_mlir_text(original.decode()), parse_mlir_text(selected.decode()), json.loads(routing).get("routed") or []
    )
    if (
        receipt.read_bytes() != receipt_bytes
        or Path(snapshot).read_bytes() != original
        or Path(prepared).read_bytes() != selected
        or Path(sidecar).read_bytes() != routing
    ):
        raise ReadbackError("profile source or routing changed during binding; rebuild")
    return result


def prepare_profile(work, prepared, directory, routing, arguments, undefined, *, transformed):
    """Generate the existing profiler under the actual source/object boundary.

    The runtime driver supplies the observed undefined symbols and retains its
    own process, compilation and link authority. Returned flags select no target.
    """
    from merlin.compiler.lower.device.offload import SIDECAR_NAME

    rows = list(routing.get("routed") or [])
    missing = any(
        row.get("symbol") not in undefined and f"_mlir_ciface_{row.get('symbol')}" not in undefined for row in rows
    )
    bindings = (
        read_bindings(work / "device_host_abi" / "post_offload_transform.json", prepared, work / SIDECAR_NAME)
        if missing and transformed
        else {}
    )
    return emit_profile(
        directory,
        routing,
        arguments,
        undefined,
        bindings=bindings,
        source=prepared,
        model_object=work / "model.o",
        source_kind="prepared_mlir",
    )


def emit_profile(
    directory, routing, arguments, undefined, *, bindings, source, model_object, source_kind, observations=()
):
    """Emit the existing counter wrappers from an observed source/object ABI.

    Completed-object packaging may observe already compiled pointer bridges.
    This supplies no effects or implementation permission, and labels its
    source independently from the ordinary prepared-MLIR observation.
    """
    from merlin.common.digest import sha256_file
    from merlin.runtime.harness.output_observation import callees, group_names, render_group_profile, wrap_flags

    rows = list(routing.get("routed") or [])
    selected = callees(rows, arguments.get("signatures") or {}, undefined, arguments.get("call_buffers"), bindings)
    # A completed object proves its referenced callees and their ABI, not the
    # dynamic order of source routes. Mixed producer routes may share a callee
    # while their sidecar roster has a different order from actual calls.
    names = group_names(rows) if source_kind == "prepared_mlir" else []
    (directory / "group_profile_bindings.json").write_text(
        json.dumps(
            dict(
                schema="merlin.group_profile_bindings.v1",
                bindings=bindings,
                source_observation=dict(kind=source_kind, path=str(Path(source).resolve()), sha256=sha256_file(source)),
                **({"prepared_sha256": sha256_file(source)} if source_kind == "prepared_mlir" else {}),
                model_object_sha256=sha256_file(model_object),
                undefined_symbols=sorted(undefined),
                observations=list(observations),
                labels="source_route_order" if names else "observed_callee_only",
                boundary="inclusive selected external ABI call; host gaps reported separately",
                scope="source/ABI observation plus actual undefined symbols; no numerical or timing authority",
            ),
            indent=2,
        )
        + "\n"
    )
    source = directory / "merlin_group_profile.c"
    source.write_text(render_group_profile(selected, names), encoding="utf-8")
    return source, wrap_flags(selected)


# ---- group_profile_sibling ---------------------------------------------------------------------
# Observe plural pointer-call profiling ABIs in actual compiled bridges.
#
# These are counter boundaries, not borrowed effects or ownership permissions.
# Every mapping reopens the completed model's linked shim and its observed C
# compilation, then matches the entire original generated forwarding function.


def _reopen(row, pins):
    path = Path(row["path"])
    size = row.get("bytes", row.get("size_bytes"))
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != row["sha256"]
        or size is not None
        and path.stat().st_size != size
    ):
        raise ReadbackError("profile compiled sibling bridge changed")
    pin = dict(path=str(path), sha256=row["sha256"], bytes=path.stat().st_size)
    if str(path) in pins and pins[str(path)] != pin:
        raise ReadbackError("profile sibling bridge has conflicting observations")
    pins[str(path)] = pin
    return path


def _bridge(program, record, pins):
    source = _reopen(record["source"], pins)
    obj = _reopen(record["object"], pins)
    if not any(row["path"] == str(obj) and row["sha256"] == record["object"]["sha256"] for row in program["objects"]):
        raise ReadbackError("profile sibling shim is absent from the completed final link")
    observed = invocation_record.verify(_reopen(record["compilation"], pins))
    if (
        observed.get("stage") != "device_shim_compilation"
        or observed["inputs"] != [dict(path=str(source), sha256=record["source"]["sha256"])]
        or observed["outputs"] != [dict(path=str(obj), sha256=record["object"]["sha256"])]
    ):
        raise ReadbackError("profile sibling bridge lacks its actual C compilation")
    for row in (
        *observed["inputs"],
        *observed["outputs"],
        *observed["dependencies"],
        observed["executable"],
        observed["stdout"],
        observed["stderr"],
    ):
        _reopen(row, pins)
    return source.read_text()


def completed_sibling_bindings(program, routing, undefined):
    """Derive existing ``callees`` bindings without recompiling model code.

    The input program is the original completed-program reader's result. Source
    route names are joined through the complete emitted C bridge, never by a
    suffix convention. The actual model object must reference that void entry.
    Returned pins retain observations only; no serialized lifetime is adopted.
    """
    pins, cache, bindings = {}, {}, {}
    sidecar = program["receipt"]["output"]["device_sidecar"]
    if json.loads(_reopen(sidecar, pins).read_text()) != routing:
        raise ReadbackError("profile sibling routing differs from the completed build")
    kernels = (program["receipt"]["output"].get("device_artifacts") or {}).get("kernels", {})
    for symbol in sorted({row["symbol"] for row in routing.get("routed", ())}):
        custody = kernels.get(symbol, {}).get("custody", {}).get("sibling_source")
        if custody is None:
            continue
        record = custody["descriptor_bridge"]
        key = json.dumps(record, sort_keys=True)
        if key not in cache:
            cache[key] = _bridge(program, record, pins)
        source = cache[key]
        call = routing.get("call_buffers", {}).get(symbol)
        if (
            not isinstance(call, list)
            or not 0 < len(call) <= 64
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("role"), str)
                or not isinstance(row.get("shape"), list)
                or not row["shape"]
                or any(type(d) is not int or d <= 0 for d in row["shape"])
                for row in call
            )
        ):
            raise ReadbackError("profile sibling route lost its complete positional buffers")
        results = [i for i, row in enumerate(call) if row["role"].startswith("out_")]
        if len(results) < 2:
            raise ReadbackError("profile sibling bridge requires its complete plural results")
        candidates = set()
        for line in source.splitlines():
            if line.startswith("void _mlir_ciface_"):
                name, separator, _ = line[len("void _mlir_ciface_") :].partition("(")
                if separator and is_identifier(name):
                    candidates.add(name)
        matches = [
            name for name in candidates if source.count(borrowed_bridge_source(symbol, name, call, results)) == 1
        ]
        if len(matches) != 1:
            raise ReadbackError("profile sibling has no unique complete compiled forwarding bridge")
        borrowed = matches[0]
        if f"_mlir_ciface_{borrowed}" not in undefined:
            raise ReadbackError("profile sibling bridge is not referenced by the actual model object")
        if any(row["borrowed_symbol"] == borrowed for row in bindings.values()):
            raise ReadbackError("profile sibling entry has ambiguous source routes")
        bindings[symbol] = dict(
            borrowed_symbol=borrowed, buffers=len(call), argument_ranks=[len(row["shape"]) for row in call]
        )
    for row in tuple(pins.values()):
        _reopen(row, pins)
    return bindings, list(pins.values())


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
