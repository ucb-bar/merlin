"""Which device seams this repo can emit, and which of them this pipeline can build as objects.

Split out of :mod:`merlin.llvmlower.device_build` (which re-exports every name) to keep that module
under the repository's module-size limit.
"""

from __future__ import annotations

#: Transports whose package artifact THIS pipeline can turn into an object. `host_instruction` is a
#: device the host drives with its own instructions (the artifact is LLVM-dialect MLIR); None is a
#: unit that lowers through a stock LLVM target and has no separate device artifact at all.
#:
#: NOT the set of transports whose boundary this repo can emit -- see :func:`boundary_buildable`. A
#: `device_native` boundary is not an object at all (it is a DRAM address contract), so it is emitted
#: by :mod:`merlin.llvmlower.device_native` and would be wrong to list here: this constant gates
#: `build_device_objects`, which really can only build the ones named.
BUILDABLE_TRANSPORTS: frozenset = frozenset({"host_instruction", None})

#: Transports with an emitter somewhere in this repo, and where it lives. Keyed on the derived
#: transport, never on a target: two devices sharing a transport share an emitter by construction.
_SEAM_EMITTERS: dict = {
    "device_native": "merlin.llvmlower.device_native",
}


def boundary_buildable(device: str) -> str | None:
    """Why no path in this repo can emit ``device``'s host/device boundary, or None when one can.

    PUBLIC because the composition axis needs it. ``targetgen.boundary`` classifies a capsule's
    accelerator/host seam from ELIGIBILITY alone, which silently claims a crossing on a target whose
    seam nothing can compile -- so it has to be able to ask, and a private cross-package import is how
    that drifts. Same predicate, one name.

    A BOUNDARY IS NOT ALWAYS AN OBJECT, which is what this used to assume. Asking only whether
    ``build_device_objects`` could compile the artifact answered "no" for every ``device_native``
    device -- correctly at the time, because nothing emitted that boundary, and wrongly once something
    did. A transport with its own emitter is delegated to it, and that emitter answers with its own
    fail-closed reasons (an underivable DRAM window, an operand placement that is not an address
    contract). So a target still reports UNKNOWN when its seam genuinely cannot be emitted; it stops
    reporting UNKNOWN for the reason "this is not the transport I know how to build".
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest

        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception:  # noqa: BLE001 -- an unresolvable device is caught by the package load
        return None
    # DELEGATED TO `_SEAM_EMITTERS` ONLY BECAUSE A SEAM HAS NOW BEEN EMITTED. This line was held back on
    # purpose while nothing had built one: turning it on flips the composition axis from UNDETERMINABLE
    # to a shape for every `device_native` capsule, and doing that on the strength of a module that
    # merely exists is exactly the move `boundary.profile_linalg` exists to prevent -- "the composition
    # numbers would improve by argument alone". What changed is an artifact, not an argument:
    # `merlin/tests/infra/test_device_native_seam_emits.py` builds a complete seam for a device_native
    # target from its own backend package -- the package's emitted directives ASSEMBLED to the bytes the
    # device fetches, the address contract read off the command buffer the same package emitted for the
    # same capsule and converted to offsets in the device's own derived DRAM window, and the host stager
    # compiled to an object -- then LINKS that object against a harness generated from the contract and
    # runs it, checking each operand lands at its declared offset and each result is collected from
    # its own. The emitter keeps its own fail-closed refusals (an underivable window, an operand
    # placement that is not an address contract), so a target whose seam genuinely cannot be emitted
    # still reports UNKNOWN; it stops reporting UNKNOWN for the reason "this is not the transport I
    # know how to build".
    emitter = _SEAM_EMITTERS.get(link.command_transport)
    if emitter is not None:
        from importlib import import_module

        return import_module(emitter).seam_emittable(device)
    if link.command_transport not in BUILDABLE_TRANSPORTS:
        return (
            f"{device!r} is reached by {link.command_transport!r}; this path compiles a device "
            f"whose artifact is LLVM-dialect MLIR, and that transport's package emits "
            f"{link.emitted_artifact or 'another artifact'} instead"
        )
    return None


def objects_buildable(device: str) -> str | None:
    """Why THIS pipeline cannot turn ``device``'s package artifact into linkable objects, or None.

    Split from :func:`boundary_buildable` the moment a second transport gained an emitter. The two
    questions had one answer while there was one emitter, and collapsing them again is a real hazard
    in the direction that matters: a device whose boundary IS emittable (as an address contract) would
    otherwise be let into the loop below, hand a stream of ``.word`` directives to ``mlir-translate``,
    and fail obscurely -- or worse, produce a shim declaring extern kernel symbols nothing defines.
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest

        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception:  # noqa: BLE001 -- an unresolvable device is caught by the package load
        return None
    if link.command_transport not in BUILDABLE_TRANSPORTS:
        return (
            f"{device!r} is reached by {link.command_transport!r}; this path compiles a device "
            f"whose artifact is LLVM-dialect MLIR, and that transport's package emits "
            f"{link.emitted_artifact or 'another artifact'} instead"
            + (
                f". Its boundary IS emittable -- as a DRAM address contract, by "
                f"{_SEAM_EMITTERS[link.command_transport]} -- just not as an object"
                if link.command_transport in _SEAM_EMITTERS
                else ""
            )
        )
    return None
