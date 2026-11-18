"""Move selected contractions onto a DEVICE, as calls a compiled host program can make.

This is the shape the matrix-unit path already proved works end to end: replace the contraction with
a call to a private symbol, record what was minted in a sidecar so the build step (a different
process) can generate the callee, and let the host object and the device object meet in one archive.
What that path cannot do is serve a second device: its symbol stem, its dtype legality and its
operand types are literals for one unit. Here all three come from the named device.

Why a sidecar rather than a return value: the rewrite runs inside the lowering subprocess and the
build step that generates the callee runs outside it, so an in-memory hand-off silently produced an
empty signature set.

**Nothing is moved without a decision.** ``select`` is a parameter; with none, the module is returned
untouched. A pass that decided for itself would duplicate the placement decision, and then the two
would disagree -- which is exactly the state this work exists to end.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.compiler.lower.device.offload import (
    BY_CONTRACTION,
    BY_GROUP,
    CONTRACTION_ACCESS,
    SIDECAR_NAME,
    DeviceRewrite,
    Routed,
)

# ---- device_offload ----------------------------------------------------------------------------


def _kernel_identity(entry: dict[str, Any]) -> str:
    """Two groups share one device kernel only when they ask for the SAME program.

    Built on :func:`merlin.development.targetgen.corpus.group_entries._identity`, which is the corpus's notion of "the
    same demand": it drops the name and the requantize multipliers, because a unit's COMMAND STREAM is
    the same for every positive multiplier. That is the right key for a capsule corpus and the wrong
    one for a kernel, because :func:`merlin.development.targetgen.corpus.model_builders.build` bakes the multiplier INTO the
    artifact the package emits. Two layers that differ only in ``acc_scale`` would then share a kernel
    carrying one of the two multipliers, and the other layer would compute a scaled version of itself
    with nothing in the build saying so. So the baked numbers are appended to the corpus identity
    rather than a second identity being invented here -- one notion of sameness, extended where the
    emission makes a dropped field load-bearing.
    """
    from merlin.compiler.lower.device.offload import _BAKED_INTO_THE_KERNEL
    from merlin.development.targetgen.corpus.group_entries import _IDENTITY_DROPS, _identity

    # Only the NUMBERS the emission bakes come back. The rest of what the corpus drops -- the name,
    # the free-text comment, the source reference naming which layer this was -- is per-layer
    # bookkeeping, and folding it back in would make every call site its own kernel and defeat the
    # sharing this key exists for. Intersected with the corpus's own list so that a field it stops
    # dropping stops being appended here too, rather than being added twice.
    baked = {key: entry[key] for key in _BAKED_INTO_THE_KERNEL if key in entry and key in _IDENTITY_DROPS}
    return _identity(entry) + "|" + json.dumps(baked, sort_keys=True)


def _entry_extents(entry: dict[str, Any]) -> tuple[int, ...]:
    """Declared matrix extents, including the empty reduction of an elementwise residual.

    An extent triple is what a caller that only knows contractions can read off a program. A
    convolution states its geometry instead (``ci``/``kh``/``kw``/``stride``), so it has no triple.
    An elementwise sum declares exactly ``(M, N)`` and no reduction axis. Retaining those two
    extents lets its original source signature join the compiled call without inventing a K.
    Consumers are expected to read the ENTRY when there is one; the triple exists so
    a signature map stays meaningful to the contraction-granular consumers that predate this route.
    """
    if entry.get("op") == "residual_add" and all(k in entry for k in ("M", "N")):
        return tuple(int(entry[k]) for k in ("M", "N"))
    return tuple(int(entry[k]) for k in ("M", "N", "K")) if all(k in entry for k in ("M", "N", "K")) else ()


def _stated_shape(entry: dict[str, Any], shape_class):
    """The extent shape a group's own statement makes, or ``None`` when it declares no triple.

    The placement decides over ``(M, K, N)``; a statement that carries all three can be keyed the
    same way as an observed contraction, and one that cannot (a convolution states its geometry) is
    ``None`` rather than a triple assembled out of the geometry -- two different numbers under one
    name is how a decision gets applied to a shape nobody decided.
    """
    rows, columns = entry.get("M"), entry.get("N")
    if rows is None or columns is None:
        return None
    reduced = entry.get("K")
    # An elementwise sum reduces over NOTHING, and `()` is how `ContractionShape` says that. A
    # selector keyed on a reduction extent then declines it, which is the right answer rather than an
    # invented one: a placement derived from contraction demands never decided about this region.
    return shape_class(
        op=str(entry.get("op") or "matmul"),
        parallel=(int(rows), int(columns)),
        reduction=() if reduced is None else (int(reduced),),
    )


def _element_token(mlir_type) -> str:
    """The MLIR element-type token of a tensor type, as the shim's byte-width table spells it."""
    element = getattr(mlir_type, "element_type", mlir_type)
    return str(element)


def _integer_element(mlir_type) -> bool:
    """Does ``mlir_type`` (a tensor type) hold integer elements?

    Asked structurally of the type rather than of its spelling: the device this route builds for
    computes in the integer datapath its entry declares, so a group whose escaping value is a float
    tensor cannot be answered by that kernel however its extents look.
    """
    from xdsl.dialects.builtin import IntegerType

    element = getattr(mlir_type, "element_type", None)
    return isinstance(element, IntegerType)


def _device_operands(group, stated) -> tuple[tuple[Any, Any] | None, str]:
    """``((activation, stored), "")`` -- the two values the STATED program reads -- or the refusal.

    NOT the group's free values. A closed group reads its scale splats, its zero points and its
    padding constants as well, and a device kernel built from the group's ENTRY takes none of them:
    the multiplier is baked into the entry as ``acc_scale`` and the zero point into its numerics. A
    call carrying them would declare a callee nothing can define -- the kernel ABI a backend contract
    declares is the three-operand one -- so the operands are derived from the statement instead.

    Both operands are taken from BEFORE the dequantize, because the dequantize is what the group
    absorbed: the device reads the stored integers. A group whose operand reaches the contraction as a
    float with no dequantize behind it is refused rather than handed to an integer datapath.
    """
    root = group.root
    if group.window_mean is not None or group.window_pool is not None:
        from merlin.compiler.partition.groups import compute_groups as CG

        dequantizes = [
            member
            for member in group.members
            if (stage := CG.classify(member)) is not None and stage.kind == CG.DEQUANTIZE
        ]
        if len(dequantizes) != 1 or not dequantizes[0].operands:
            return None, "the window mean has no unique integer activation source"
        activation = dequantizes[0].operands[0]
        if not _integer_element(activation.type):
            return None, "the window mean's activation is not stored as integers"
        # The other operand is the source-derived constant-one array supplied
        # by the logical ABI prepack, not the float zero-init of linalg.reduce.
        return (activation, activation), ""
    if len(list(root.operands)) < 2:
        return None, "this group's root reads fewer than two operands, so the kernel ABI has nothing to take"
    from merlin.compiler.partition.groups import compute_groups as CG

    values = []
    for index in (0, 1):
        operand = list(root.operands)[index]
        _adapters, dequantize, _dtype = CG._input_chain(operand)
        value = dequantize.operands[0] if dequantize is not None else operand
        if not _integer_element(value.type):
            return None, f"operand {index} reaches this root as a float, so an integer kernel cannot read it"
        values.append(value)
    stored = stated.stored_operand
    if stored is None:
        # A SUM OF TWO ACTIVATIONS HAS NO STORED SIDE, and that is a statement about the program
        # rather than a gap: `group_command.program` returns `stored_operand=None` for it on purpose.
        # Its two operands go in the order the root reads them, which is the order the entry's
        # ``lhs_scale``/``rhs_scale`` are stated in -- swapping them would apply each scale to the
        # other tensor. A contraction of two activations is the same case: the statement read them as
        # lhs[M, K] @ rhs[K, N] in the root's own order.
        return (values[0], values[1]), ""
    return ((values[1], values[0]) if stated.transposed else (values[0], values[1])), ""


def rewrite_groups_to_device(
    module,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    weight_args=None,
    model: str = "",
    oracle=None,
    capture: str | Path | None = None,
    sidecar_dir: str | Path | None = None,
    require_complete_groups: bool = False,
    blob_weights: bool | None = None,
) -> DeviceRewrite:
    """Replace each SELECTED closed COMPUTE GROUP with ONE call to ``device``'s kernel (see
    :func:`_rewrite_groups_to_device`). Under the logical kernel ABI a framework max-pool (a
    NaN-propagating select-max window) is recognized as the readout's maxpool stage: the device's
    pool takes integer readout values, which cannot be NaN, so it is the same maximum.

    ``blob_weights`` binds each laid-out weight as a blob argument instead of a constant; by default
    it is what the enclosing build requested (:func:`merlin.compiler.lower.c.weights.requested`)."""
    from merlin.compiler.lower.device.shim import kernel_abi_for
    from merlin.compiler.partition import outline as CC

    abi = kernel_abi_for(device)
    if blob_weights is None:
        from merlin.compiler.lower.c.weights import is_requested

        blob_weights = is_requested()
    with CC.select_max_is_max(abi is not None and getattr(abi, "version", 1) == 2):
        return _rewrite_groups_to_device(
            module,
            device,
            select=select,
            weight_args=weight_args,
            model=model,
            oracle=oracle,
            capture=capture,
            sidecar_dir=sidecar_dir,
            require_complete_groups=require_complete_groups,
            blob_weights=blob_weights,
        )


def _rewrite_groups_to_device(
    module,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    weight_args=None,
    model: str = "",
    oracle=None,
    capture: str | Path | None = None,
    sidecar_dir: str | Path | None = None,
    require_complete_groups: bool = False,
    blob_weights: bool = False,
) -> DeviceRewrite:
    """Replace each SELECTED closed COMPUTE GROUP with ONE call to ``device``'s kernel. Mutates ``module``.

    The unit this moves is the layer, not the multiply-accumulate:
    :func:`~merlin.compiler.partition.groups.formation_compute_groups.form_groups` closes a contraction together
    with the stages the target's readout takes, and
    :func:`~merlin.compiler.partition.groups.group_command.program` restates that group as the entry a
    backend package builds from -- epilogue, multiplier, convolution geometry, committed type. Routing
    the contraction alone (:func:`rewrite_contractions_to_device`) leaves every one of those on the
    host, so the model runs as a device call per matmul interleaved with host epilogues: correct, and
    nothing like the program a whole-model schedule emits.

    A group is routed, or it is DECLINED BY NAME. There is no third outcome and no fallback to the
    bare contraction: a build that silently substituted extents for a statement would drop the
    readout and report the same count either way.

    ``select`` is the placement decision, passed in and never taken here -- the contract the
    contraction route keeps, for the same reason. It is offered the group's root as the
    ``ContractionShape`` the placement decided over, because
    :func:`merlin.target.system.place.device_selector` keys on extents and cannot read an op handle.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region

    from merlin.compiler.ir import contraction_shapes
    from merlin.compiler.ir.contraction_shapes import ContractionShape, observe_contractions
    from merlin.compiler.lower.device.group_host_call import (
        _device_result,
        _erase_dead,
        _host_window_gather,
        _logical_group_call,
        _nhwc,
        _nhwc_matrix,
        _owner,
        _prepack_arrays,
        _routed_prepack,
        _shape,
        cancel_relayouts,
    )
    from merlin.compiler.lower.device.group_route import capsule_entry_for
    from merlin.compiler.lower.device.offload import symbol_stem
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.compiler.partition.groups import group_command as GC
    from merlin.target.system import device_dtype_triples

    if type(require_complete_groups) is not bool:
        raise TypeError("complete source group rewriting requires an original Boolean")
    if require_complete_groups and select is None:
        raise ValueError("complete source group rewriting has no original selection")
    if select is None:
        return DeviceRewrite(
            device=device,
            granularity=BY_GROUP,
            skipped=(("all", "no selector supplied, so nothing is routed"),),
        )
    # ONE DATAPATH AUTHORITY, NOT TWO. With an ``oracle`` the caller has supplied the device's own
    # statement of what it admits, and asking `system.offload` for a second one would be asking about
    # a different device. Without one the contract is the only source, and a device that declares no
    # datapath offloads nothing -- fail closed rather than route onto an unknown precision.
    if oracle is None and not device_dtype_triples(device):
        if require_complete_groups:
            raise ValueError("complete source group rewriting has no selected datapath")
        return DeviceRewrite(
            device=device,
            granularity=BY_GROUP,
            skipped=(("all", f"{device!r} declares no derivable datapath"),),
        )

    if weight_args is None and capture is not None:
        # WHICH ARGUMENT IS STORED, from the capture's own weights manifest. Without one a first
        # layer whose two operands are both model arguments cannot be told apart, and the honest
        # outcome is that group refusing BY NAME below rather than a guess about which side is the
        # weight -- so this reads the manifest when there is one and passes None on when there is not.
        from merlin.compiler.partition import memory_plan as _stream_plan

        weight_args = _stream_plan.weight_args_beside(capture)

    from functools import partial

    from merlin.compiler.lower.device.source_compute_groups import form_exact_groups, form_source_groups

    form = partial(
        form_source_groups if require_complete_groups else _merlin_formation_compute_groups_1.form_groups, oracle=oracle
    )
    if capture is None:
        groups = form(module, device)
    else:
        groups, _ = form_exact_groups(module, device, capture, form, weight_args=weight_args)
    # THE SHAPES ARE OBSERVED, NOT FILTERED FOR LEGALITY. `system.offload.offloadable_contractions`
    # answers "could this device take this contraction as it stands" -- and a captured layer does not
    # stand as an integer contraction at all: it arrives as a dequantize into an f32 matmul, which
    # that predicate declines for every layer of a quantized model (measured on a captured ResNet-50:
    # 0 of 54). Legality here is the compute-group ORACLE's answer (the group is not on the host) and
    # the decision is `select`'s; these shapes exist only to key the decision, which
    # `system.place.device_selector` does on extents.
    shape_of = {id(op): shape for op, shape in observe_contractions(module)}

    # Statements use this same original module. Replacements preserve SSA
    # types and the relative order of surviving original operations; neither
    # observation needs a full module/block walk for every routed group.
    extents = contraction_shapes.contraction_extents(module)
    orders = {
        block: {id(op): index for index, op in enumerate(block.ops)}
        for block in dict.fromkeys(g.root.parent for g in groups if g.root is not None)
        if block is not None
    }

    stem = symbol_stem(device)
    # UNDER THE LOGICAL KERNEL ABI THE CALL CARRIES EVERY POINTER OF THE GROUP'S INTERFACE: its
    # constant values (a folded bias, a re-laid weight) are passed from the host like its activation.
    from merlin.compiler.lower.device.shim import kernel_abi_for

    selected_abi = kernel_abi_for(device)
    logical = selected_abi is not None and getattr(selected_abi, "version", 1) == 2
    prepack_rows: dict[int, dict[str, Any]] = {}
    prepack_arrays: dict[str, Any] = {}
    why_no_arrays = ""
    prepack_source = None
    candidates = [g for g in groups if g.placement != CG.HOST and g.root is not None]
    if logical:
        prepack_source, prepack_rows, prepack_arrays, why_no_arrays = _prepack_arrays(candidates, capture)
    else:
        prepack_source, why_no_arrays = _prepack_for(candidates, capture)
    weight_binder = None
    if blob_weights and logical:
        from merlin.compiler.lower.c.weights import WeightArgs

        weight_binder = WeightArgs(module)
    call_buffers: dict[str, list[dict[str, Any]]] = {}
    views: list = []  # the relayouts this route inserts, for cancel_relayouts below
    symbols: dict[str, str] = {}  # kernel identity -> symbol
    minted: dict[str, tuple[int, ...]] = {}
    declared: dict[str, tuple[Any, ...]] = {}  # symbol -> the operand types its declaration carries
    returns: dict[str, Any] = {}  # symbol -> the result type its declaration carries
    programs: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    access: dict[str, tuple[str, ...]] = {}
    routed: list[Routed] = []
    skipped: list[tuple[str, str]] = []
    placed: list = []  # the groups that actually became calls, for the prepack record

    # Layout inference may follow an earlier producer. State every group while
    # the original graph is intact; replacing that producer with a call erases
    # the contraction maps needed to infer a later residual's feature axis.
    stated_groups = []
    for group in groups:
        name = (
            f"{''.join(c if (c.isalnum() or c == '_') else '_' for c in model)}_group{group.index}"
            if model
            else f"group{group.index}"
        )
        if group.placement == CG.HOST or group.root is None:
            continue  # the planner's own decision; `group_offload.census` is where it is counted
        try:
            stated = GC.program(group, weight_args=weight_args, name=name, extents=extents)
        except CG.NoCapsuleForm as error:
            skipped.append((name, f"no device form: {error}"))
            continue
        stated_groups.append((group, name, stated))

    for group, name, stated in stated_groups:
        # The placement is keyed on EXTENTS, so the group is offered to `select` as the shape its root
        # makes. A root the shape observer cannot read (a reduce that the group states as a window
        # mean, say) still HAS extents -- the ones its own statement declares -- so the statement is
        # the fallback rather than a decline: declining there would leave a routable layer on the host
        # for want of a second reading of the same numbers.
        shape = shape_of.get(id(group.root)) or _stated_shape(stated.entry, ContractionShape)
        if shape is None or not select(shape):
            skipped.append((name, "the placement did not select this group's root"))
            continue

        block = group.root.parent
        order = orders.get(block, {})
        if any(member.parent is not block or id(member) not in order for member in group.members):
            skipped.append((name, "the group names an op outside its root's block; it cannot be replaced by one call"))
            continue
        members = sorted(group.members, key=lambda m: order[id(m)])
        from merlin.compiler.lower.device.group_accumulator_cut import prepare as prepare_accumulator_cut

        try:
            accumulator_cut = prepare_accumulator_cut(group, capture)
        except ValueError as error:
            skipped.append((name, f"no exact accumulator/host split: {error}"))
            continue
        if accumulator_cut is not None:
            members = list(accumulator_cut.prefix)
        last = members[-1]
        if not last.results:
            skipped.append((name, "the group's last member produces no value, so a call cannot stand for it"))
            continue
        # EVERY ESCAPING VALUE MUST BE THE ONE THE CALL RETURNS. Formation grows a group only along
        # values with a single consumer, so this holds by construction -- which is exactly why it is
        # checked: a later clause that admitted a second consumer would make the erase below drop a
        # value something outside still reads, and the module would fail to verify far from here.
        inside = {id(member) for member in members}
        escaping = [
            member
            for member in members[:-1]
            for result in member.results
            for use in result.uses
            if id(use.operation) not in inside
        ]
        if escaping:
            skipped.append(
                (name, f"{len(escaping)} value(s) of this group are read outside it, so one call cannot stand for it")
            )
            continue

        result_type = last.results[0].type
        # THE DEVICE COMMITS THE TYPE THE CALL RETURNS. A group that stops before its closing
        # requantize still escapes as the capture's float accumulation, and an integer kernel cannot
        # produce that -- so it is refused by name rather than routed into a call whose result type
        # disagrees with what the kernel computes. (Measured on a captured micro-model: all four of
        # its device groups end at the contraction, because the matmul's result is read twice.)
        if not _integer_element(result_type):
            skipped.append(
                (name, "this group escapes as a float value, which the integer datapath it states cannot commit")
            )
            continue
        if group.window_pool is not None and not logical:
            skipped.append((name, "a standalone window pool needs the single-input logical ABI"))
            continue
        if group.window_mean is not None and not logical:
            skipped.append((name, "a window mean needs the logical ABI's constant-one operand"))
            continue
        pair, why = _device_operands(group, stated)
        if pair is None:
            skipped.append((name, why))
            continue
        if stated.batch_shape:
            batch = stated.batch_shape[0]
            rows, columns, reduced = _entry_extents(stated.entry)
            expected = ((batch, rows, reduced), (batch, reduced, columns), (batch, rows, columns))
            actual = tuple(tuple(t.get_shape()) for t in (pair[0].type, pair[1].type, result_type))
            if actual != expected:
                skipped.append((name, "the batch slice ABI cannot reinterpret operand or result views"))
                continue

        region_attr = group.root.attributes.get("prov.region_id")
        nodes_attr = group.root.attributes.get("prov.source_node_ids")
        region_id = region_attr.data if isinstance(region_attr, StringAttr) and region_attr.data else None
        node_ids = (
            tuple(sorted(value.data for value in nodes_attr.data))
            if isinstance(nodes_attr, ArrayAttr)
            and nodes_attr.data
            and all(isinstance(value, StringAttr) and value.data for value in nodes_attr.data)
            else ()
        )
        if len(node_ids) != len(set(node_ids)):
            node_ids = ()

        entry = capsule_entry_for(stated.entry, index=group.index, name=name, device=device, model=model)
        operands, constants, roles = [pair[0], pair[1]], [], ["input_0", "weight_0"]
        gather, relayout, call_type = None, None, result_type
        if logical:
            # A CONVOLUTION THE CAPTURE GATHERED ON THE HOST IS ROUTED FROM ITS IMAGE. The patch matrix
            # is the host's lowering of the convolution, not the layer's input: the device is handed the
            # image before the gather, in the interface's declared layout, and forms its windows however
            # it chooses. The gather left behind is removed below once nothing reads it.
            restated = False
            if entry.get("op") in ("matmul", "conv2d") and stated.stored_operand is not None and not stated.batch_shape:
                # The statement names a host gather as a convolution (group_command.program, from the
                # same recognition); the image is read back from the activation's own [positions,
                # depth] -- not from the entry's output rows, which a fused pool shrinks.
                activation_shape = _shape(pair[0])
                try:
                    gather = _host_window_gather(pair[0], *activation_shape) if len(activation_shape) == 2 else None
                except ValueError as why:
                    skipped.append((name, f"interface value input_0 has no host source: {why}"))
                    continue
            if gather is not None and entry.get("op") == "matmul":
                entry = {key: value for key, value in entry.items() if key not in ("M", "K")}
                entry.update({"op": "conv2d", **gather["window"]})
                restated = True
            from merlin.compiler.lower.host.layout import (
                constant_panel_minimum_input_channels,
                constant_panel_request,
            )

            operands, constants, roles, why = _logical_group_call(
                group,
                stated,
                entry,
                pair,
                prepack_rows.get(int(group.index)),
                prepack_arrays,
                why_no_arrays,
                weight_rows=gather["rows"] if restated else None,
                bind_weight=weight_binder,
                constant_panel_columns=constant_panel_request(module),
                constant_panel_minimum_input_channels=constant_panel_minimum_input_channels(module),
            )
            if operands is None:
                skipped.append((name, why))
                continue
            # Exact source-owned matrix views returned with logical operands
            # participate in the same inverse-view cancellation as results.
            views.extend(op for op in constants if isinstance(op, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)))
            if gather is not None:
                image_ops = _nhwc(gather["image"])
                views += image_ops
                constants = [*image_ops, *constants]
                operands[0] = image_ops[-1].results[0]
            elementwise = False
            if entry.get("op") in ("residual_add", "maxpool2d") and all(
                len(_shape(v)) == 4
                and _shape(v)[0] == 1
                and _shape(v)[1] == int(entry.get("N", -1))
                and _shape(v)[2] * _shape(v)[3] == int(entry.get("M", -1))
                for v in operands
            ):
                # AN ELEMENTWISE SUM OF NCHW TENSORS against an interface stated over [positions,
                # channels]: both inputs go over in that layout and the result comes back from it.
                for index, value in enumerate(list(operands)):
                    matrix = _nhwc_matrix(value)
                    views += matrix
                    constants = [*constants, *matrix]
                    operands[index] = matrix[-1].results[0]
                elementwise = True
            relayout, call_type, why = _device_result(
                group,
                members,
                entry,
                result_type,
                elementwise=elementwise,
                batch=tuple(stated.batch_shape),
                transposed=stated.transposed,
            )
            if why:
                skipped.append((name, f"interface value out_0 has no host destination: {why}"))
                continue
        param_types = (*(value.type for value in operands), call_type)
        identity = _kernel_identity(entry)
        if stated.batch_shape:
            # Different batch counts are different monomorphic host wrappers,
            # while their stated per-slice device programs remain identical.
            identity += "|batch_abi:" + json.dumps([str(t) for t in param_types])
        sym = symbols.get(identity)
        if sym is None:
            sym = f"{stem}_{len(symbols)}"
            symbols[identity] = sym
            declared[sym] = param_types
            returns[sym] = call_type
            minted[sym] = (*stated.batch_shape, *_entry_extents(entry))
            programs[sym] = stated.to_dict()
            entries[sym] = entry
            access[sym] = ("read",) * len(operands) + ("write",) if logical else CONTRACTION_ACCESS
            if logical:
                call_buffers[sym] = [
                    {"role": role, "shape": [int(v) for v in t.get_shape()], "dtype": _element_token(t)}
                    for role, t in zip((*roles, "out_0"), param_types, strict=True)
                ]
        elif (declared[sym], returns[sym]) != (param_types, call_type):
            # MLIR function types are monomorphic. Two groups that ask for the same PROGRAM but hand
            # it different operand types are two callees; sharing one symbol would emit a call whose
            # types disagree with its declaration, which fails in the parser far from here.
            skipped.append((name, f"this group states the same program as @{sym} but with a different function type"))
            continue

        # DESTINATION-PASSING, like the contraction route's call: the callee writes the committed
        # result into the buffer it is handed and returns it. The destination is minted here rather
        # than reusing the contraction's zero fill, because the fill holds ACCUMULATOR elements and
        # what the group commits is what its readout narrows them to.
        destination = tensor.EmptyOp((), call_type)
        call = func.CallOp(sym, [*operands, destination.results[0]], [call_type])
        for constant in constants:
            last.parent.insert_op_before(constant, last)
        last.parent.insert_op_before(destination, last)
        last.parent.insert_op_before(call, last)
        produced = call.results[0]
        for op in relayout(produced) if relayout is not None else ():
            last.parent.insert_op_before(op, last)
            views.append(op)
            produced = op.results[0]
        last.results[0].replace_all_uses_with(produced)
        feeding = [_owner(value) for member in members for value in member.operands]
        if accumulator_cut is not None:
            feeding.extend(accumulator_cut.apply())
        for member in reversed(members):
            member.detach()
            member.erase()
        if logical:
            # What fed only this group -- the weight's views, a host gather, the members' own inits --
            # is now read by nothing, and is removed rather than left for the host to compute.
            _erase_dead(feeding)
        # THE DATAPATH THE CALL ACTUALLY CARRIES, read off the three types the call was built with --
        # not off the capture's contraction shape, whose element types are the FLOAT ones the
        # dequantize produced. The shim sizes its staging buffers from this triple, so recording the
        # capture's f32 there would stage four bytes per int8 element of every routed layer.
        routed.append(
            Routed(
                symbol=sym,
                parallel=tuple(int(v) for v in shape.parallel),
                reduction=tuple(int(v) for v in shape.reduction),
                # The legacy datapath triple repeats the input format for a unary
                # operation; its actual arity is in signatures/arg_access.
                dtypes=tuple(  # type: ignore[arg-type]
                    _element_token(t) for t in (operands[0].type, operands[min(1, len(operands) - 1)].type, result_type)
                ),
                fqn=str(entry.get("source_reference", "")),
                group=int(group.index),
                source_region_id=region_id,
                source_node_ids=node_ids,
            )
        )
        placed.append(group)

    if logical:
        # ONE RELAYOUT PER TRUE HOST BOUNDARY. Between device groups the device's own layout is carried:
        # a result read back to NCHW and handed straight over in NHWC is the same tensor twice.
        cancel_relayouts(views)
    if weight_binder is not None:
        weight_binder.finish()
    body = module.body.block
    for sym in minted:
        # `bufferization.access` is load-bearing: without it one-shot-bufferize defensively copies the
        # weight operand, which for a many-block transformer is a large amount of pointless memcpy. It
        # is a PROPERTY on FuncOp, not a discardable attribute, so it goes through the constructor.
        attrs = ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in access[sym]])
        body.add_op(
            func.FuncOp(
                sym,
                (declared[sym], (returns[sym],)),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
        )

    if require_complete_groups and (skipped or len(placed) != sum(g.placement != CG.HOST for g in groups)):
        details = "; ".join(f"{name}: {reason}" for name, reason in skipped)
        raise ValueError(
            "complete source group rewrite has omitted or refused accelerator computation"
            + (f": {details}" if details else f" ({len(placed)} groups placed)")
        )
    # Roots have now been erased. Re-reading them can lose operands/extents and
    # independently re-fold a different bias. Keep the actual preparation used
    # by the calls and select only the groups that were routed.
    prepack = _routed_prepack(prepack_source, placed)
    why_no_prepack = why_no_arrays if placed else ""
    if why_no_prepack:
        skipped.append(("prepack", why_no_prepack))
    if require_complete_groups and skipped:
        raise ValueError("complete source group rewrite lacks its required preparation")

    out = DeviceRewrite(
        device=device,
        routed=tuple(routed),
        signatures=minted,
        skipped=tuple(skipped),
        granularity=BY_GROUP,
        programs=programs,
        entries=entries,
        prepack=prepack,
        arg_access=access,
        call_buffers=call_buffers,
        blob_weights=weight_binder,
    )
    if sidecar_dir is not None:
        out.write_sidecar(sidecar_dir)
    return out


def _prepack_for(groups, capture) -> tuple[dict[str, Any] | None, str]:
    """The routed groups' prepack record, or ``None`` and the reason there is none.

    A device kernel built from a group's stated program reads its bias ALREADY FOLDED into the
    accumulator's integer domain and its weight ALREADY LAID OUT the way the program holds it. Which
    of those the build has to do is a property of the statement, so it is recorded beside it -- a
    build that had the statement and not the prepack would stage a float bias and a capture-ordered
    weight into a kernel that expects neither, and the numbers would simply be wrong.

    ``None`` is a real answer and it is never silent: with no capture beside the module there is no
    weights manifest, and a manifest is what says which argument is stored.
    """
    if not groups:
        return None, ""
    if capture is None:
        return None, "no capture was given, so no weights manifest could be read and no bias folded"
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    from merlin.compiler.lower.c.weights import weight_files_beside

    manifest, weights = weight_files_beside(capture)
    if manifest is None or weights is None:
        return None, f"no weights manifest and safetensors pair beside {beside}"
    from merlin.compiler.partition.groups import weights as GP

    try:
        return GP.prepack(groups, manifest, weights, device_layout=True)["record"], ""
    except Exception as error:  # noqa: BLE001 -- named, never a silently absent prepack
        return None, f"prepack: {type(error).__name__}: {error}"


def rewrite_prepared_file(
    prepared: str | Path,
    work: str | Path,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    exact_selection=None,
    granularity: str = BY_CONTRACTION,
    weight_args=None,
    model: str = "",
    capture: str | Path | None = None,
    catalog_manifest: str | Path | None = None,
    catalog_object: str | Path | None = None,
    require_complete_groups: bool = False,
    offload_selection=None,
    routing=None,
) -> DeviceRewrite:
    """Rewrite a prepared module ON DISK in place and record what it minted.

    This is the seam a whole-model build uses: read the module the preparation passes produced, move
    the selected contractions onto ``device``, print it back, repair the declarations the printer
    drops, and write the sidecar the build's device side reads.

    ``granularity`` chooses WHAT is moved -- :data:`BY_CONTRACTION` (the multiply-accumulate alone,
    epilogue left on the host) or :data:`BY_GROUP` (the closed compute group, readout and all). It is
    a parameter and not a default because the two build different programs, and the sidecar records
    which one ran so a build cannot be read as the other.

    It REFUSES to write a module whose declarations lost their access attributes, and that is not
    belt-and-braces. xDSL stores ``arg_attrs`` correctly but prints them only for a function WITH a
    body; for a bodyless declaration it prints the types alone, so the attributes never reach the text
    that gets parsed, and one-shot-bufferize then defensively copies the weight operand of every
    routed contraction. A large amount of pointless memcpy in a shipped model is exactly the kind of
    regression nothing would attribute back to here.

    The repair itself is shared with the matrix-unit path rather than reimplemented: both need the
    same fixup for the same printer, and two copies would drift.
    """
    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.lower.device.offload import rewrite_contractions_to_device
    from merlin.compiler.lower.device.outline import patch_declaration_arg_attrs, unpatched_declarations

    if granularity not in (BY_CONTRACTION, BY_GROUP):
        raise ValueError(f"{granularity!r} is neither {BY_CONTRACTION!r} nor {BY_GROUP!r}")
    if type(require_complete_groups) is not bool or require_complete_groups and granularity != BY_GROUP:
        raise ValueError("complete source group selection requires its original group rewrite")
    prepared, work = Path(prepared), Path(work)
    model_sha256 = hashlib.sha256(prepared.read_bytes()).hexdigest()
    if exact_selection is not None and exact_selection.model_sha256 != model_sha256:
        raise ValueError("prepared model bytes changed after exact operation selection")
    catalog_selection = None
    catalog_bindings = None
    catalog_manifest_data = None
    if (catalog_manifest is None) != (catalog_object is None):
        raise ValueError("catalog rewrite needs both manifest and object")
    if catalog_manifest is not None:
        if select is not None or exact_selection is not None:
            raise ValueError("catalog source selection cannot be combined with another placement selector")
        data = json.loads(Path(catalog_manifest).read_text(encoding="utf-8"))
        if data.get("source_sha256") != model_sha256 or not data.get("coverage_complete"):
            raise ValueError("external catalog does not cover these exact prepared model bytes")
        rows = data.get("bindings") or ()
        catalog_selection = {row["region"]: tuple(row["tensor_types"]) for row in rows}
        catalog_bindings = {row["region"]: row for row in rows}
        catalog_manifest_data = data
        if (
            len(catalog_selection) != len(rows)
            or not catalog_selection
            or any(not region for region in catalog_selection)
        ):
            raise ValueError("external catalog needs unique nonempty source regions")
    module = parse_mlir_file(prepared)
    if catalog_selection is not None and granularity != BY_CONTRACTION:
        raise ValueError("source catalogs bind contractions; group routing requires its own program contract")
    if exact_selection is not None and granularity != BY_CONTRACTION:
        raise ValueError("exact operation selection binds contractions; a group route cannot honour it")
    if offload_selection is not None and granularity != BY_CONTRACTION:
        raise ValueError("source-bound economics cannot replace a complete group route")
    rewrite = (
        rewrite_contractions_to_device(
            module,
            device,
            select=select,
            exact_selection=exact_selection,
            model_sha256=model_sha256,
            catalog_selection=catalog_selection,
            catalog_bindings=catalog_bindings,
            catalog_manifest_data=catalog_manifest_data,
            offload_selection=offload_selection,
            routing=routing,
        )
        if granularity == BY_CONTRACTION
        else rewrite_groups_to_device(
            module,
            device,
            select=select,
            weight_args=weight_args,
            model=model,
            capture=capture,
            require_complete_groups=require_complete_groups,
        )
    )
    if offload_selection is not None and hashlib.sha256(prepared.read_bytes()).hexdigest() != model_sha256:
        raise ValueError("device economics original prepared source changed during rewriting")
    if rewrite.moved:
        # A contraction's callee has one fixed shape; a group's call takes the free values its members
        # read, so each symbol is patched with the accesses the rewrite DERIVED for it.
        text = to_text(module)
        by_access: dict[tuple[str, ...], list[str]] = {}
        for sym in rewrite.signatures:
            by_access.setdefault(tuple(rewrite.arg_access.get(sym) or CONTRACTION_ACCESS), []).append(sym)
        for access, symbols in by_access.items():
            text = patch_declaration_arg_attrs(text, symbols, argument_access=access)
        missing = unpatched_declarations(text, rewrite.signatures)
        if missing:
            raise RuntimeError(
                f"declarations {list(missing)} carry no bufferization.access attributes, so "
                "one-shot-bufferize would copy the weight operand of every contraction routed to "
                "them; refusing to write the module"
            )
        prepared.write_text(text, encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    rewrite.write_sidecar(work)
    if catalog_manifest is not None:
        sidecar = work / SIDECAR_NAME
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["catalog_manifest"] = str(Path(catalog_manifest).resolve())
        data["catalog_object"] = str(Path(catalog_object).resolve())
        sidecar.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return rewrite
