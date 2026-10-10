"""Select whole-model placement and device routing before a compiler build."""

from __future__ import annotations

from pathlib import Path


def plan_before_build(
    target: str,
    linalg_mlir: str,
    *,
    datapath: str,
    device_package: str | None = None,
    model: str = "",
    capture: str | Path | None = None,
    granularity: str = "contraction",
    linked_elf_admission=None,
) -> dict:
    """Everything the EMISSION needs decided BEFORE it runs: where each op goes, and what the device
    is asked to build.

    This function exists because of an ordering defect, not for tidiness. ``compile_model`` used to
    compute its ``Placement`` *after* ``compile_rvv`` had already lowered, built and run the model,
    so the decision could not reach the emission however it came out: a routing that put every
    contraction on an accelerator was recorded beside an image that ran all of them on the host, and
    the two read as one result. Placement is an INPUT to a build or it is a commentary on one.

    Returns the plan, the placement record, the ``DeviceRouting`` the build should be given (or None
    with the reason it could not be derived), and the group-by-group offload census. Optional placement
    modelling gaps are recorded as named ``why`` entries. A structurally incomplete contraction
    inventory is different: it raises rather than constructing a misleading placement or coverage claim.
    """
    from merlin.llvmlower import group_offload as GO
    from merlin.targetgen import capsule_source as CSRC
    from merlin.targetgen import routing as _routing

    record: dict = {"target": target}
    # Route on the EXACT registry format name, not the compile-mode token -- see the note at the
    # call site in `compile_model`.
    # The placement and its coverage denominator must describe every parsed
    # contraction.  The tag-only reader can silently lose generic-printed
    # regions, yielding a zero-mesh plan beside successful device dispatch.
    demands = CSRC.model_op_demands_checked(linalg_mlir, datapath)
    shadow = _routing.route_plan(demands, target)
    record["plan"], record["authority"] = shadow, "routing.route_plan"
    placement = None
    try:
        from merlin.system.derive import system_for_experiment as _sysfor
        from merlin.system.place import measured_cost_for as _cost_for
        from merlin.system.place import place as _place

        system, host_why = _sysfor(target)
        placement = _place(demands, system, cost=_cost_for(system))
        projected = placement.as_route_plan()
        divergence = {
            key: {"placement": len(projected[key]), "route_plan": len(shadow[key])}
            for key in ("mesh", "fallback", "scalar_rvv")
            if len(projected[key]) != len(shadow[key])
        }
        record["plan"] = projected
        record["authority"] = "system.place (routing.route_plan is the cross-check)"
        record["placement"] = {
            **placement.to_dict(),
            "host": host_why,
            "authority": record["authority"],
            "divergence": divergence or None,
        }
    except Exception as exc:  # noqa: BLE001 -- a modelling gap must not fail a compile
        record["placement"] = {
            "status": "unavailable",
            "authority": "routing.route_plan (placement unavailable)",
            "why": f"{type(exc).__name__}: {exc}",
        }

    # ONE DEVICE CALL PER CLOSED GROUP. Not per contraction: a captured layer is a contraction plus
    # the readout stages the unit absorbs, and routing only the contraction leaves the bias, the
    # requantize and the activation on the host -- a different program from the one a whole-model
    # schedule emits, reported under the same name.
    try:
        from merlin.common import mlir_query as _mq
        from merlin.xdsl_dialects.lowering import stream_plan as _stream_plan

        # WHICH ARGUMENT IS STORED, from the capture's own weights manifest. Without it a first
        # layer whose two operands are both model arguments cannot be told apart -- and the honest
        # outcome is that group refusing BY NAME, not a guess about which side holds the weight.
        # Hence `capture`: a compile pinned to a bundle can read the manifest beside it, where one
        # handed a module as text has nothing to read and says so per group.
        offload = GO.plan(
            _mq.parse(linalg_mlir),
            target,
            weight_args=_stream_plan.weight_args_beside(capture if capture is not None else linalg_mlir),
            model=model,
        )
        GO.require_every_group_accounted(offload)
        record["offload"] = offload
        record["device_program"] = offload.census()
    except Exception as exc:  # noqa: BLE001 -- the route reports on a compile, it never fails one
        record["device_program"] = {"status": "unavailable", "why": f"{type(exc).__name__}: {exc}"}

    # The routing the build is handed. Derived from the placement, never declared: the operand and
    # accumulate formats are what the router matched against, and a build that assumed them emits
    # kernels in a precision the placement never chose.
    if placement is None:
        record["device_routing_why"] = "no placement was derivable, so there is no routing to build against"
    elif not device_package:
        record["device_routing_why"] = (
            "no backend package was named, so the device side cannot be built; pass mesh_package="
        )
    else:
        from merlin.llvmlower.device_build import routing_for_placement

        devices = sorted({p.device for p in placement.placed if p.on_device})
        if len(devices) != 1:
            record["device_routing_why"] = (
                f"the placement names {len(devices)} device(s) ({devices}); one image carries one "
                "device datapath, so it has to be split before it can be built"
            )
        else:
            try:
                record["device_routing"] = routing_for_placement(
                    placement,
                    devices[0],
                    device_package,
                    granularity=granularity,
                    capture=capture,
                    model=model,
                    linked_elf_admission=linked_elf_admission,
                )
            except Exception as exc:  # noqa: BLE001 -- named, never silently absent
                record["device_routing_why"] = f"{type(exc).__name__}: {exc}"
    return record
