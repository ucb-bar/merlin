"""Original live target-pass custody, without semantic qualification or grants.

This companion preserves the configuration route and carries an independently
selected schedule owner. No declaration or saved receipt can create that owner.
"""

import hashlib
import json
from pathlib import Path


def freeze_configuration(selected, target):
    if selected is None:
        return None
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

    if type(selected) is not ConfigurationIRSelection:
        raise ValueError("target configuration requires an original ConfigurationIRSelection")
    selected.revalidate()
    if selected.selected_contracts.target != target:
        raise ValueError("target configuration selection names a different target")
    return selected, selected.record()


def require_configuration(device, frozen):
    """A later package or preparation callback cannot replace the selection."""
    selected = getattr(device, "target_configuration", None)
    if frozen is None:
        if selected is not None:
            raise ValueError("target configuration selection changed during build")
        return None
    original, record = frozen
    if selected is not original:
        raise ValueError("target configuration selection changed during build")
    selected.revalidate()
    if selected.selected_contracts.target != device.device or selected.record() != record:
        raise ValueError("target configuration context changed during build")
    return selected


def configuration_artifacts(device, selected, artifacts):
    """Reopen original receipts after later provider and final-image callbacks."""
    if selected is None:
        return
    if not artifacts or not artifacts.get("kernels"):
        raise ValueError("selected target configuration has no consumed package IR in this route")
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen.package_runtime import load_package

    package = load_package(str(device.package_dir))
    for kernel in artifacts["kernels"].values():
        pin = kernel["custody"]["target_configuration"]
        CC.revalidate_target_configuration(
            package,
            Path(pin["path"]).parent,
            selected,
            expected_receipt_sha256=pin["sha256"],
        )


def _sources():
    return tuple(
        (path, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("device_build.py").resolve(),
            Path(__file__).with_name("device_routing_placement.py").resolve(),
            Path(__file__).with_name("device_seam_buildability.py").resolve(),
            Path(__file__).with_name("device_preparation.py").resolve(),
        )
    )


def _methods(selection):
    methods = []
    for name in ("record", "revalidate"):
        method = getattr(selection, name)
        function = getattr(type(selection), name)
        if getattr(method, "__self__", None) is not selection or getattr(method, "__func__", None) is not function:
            raise ValueError("target schedule requires original bound selection methods")
        methods.append((method, function, function.__code__, function.__defaults__, function.__kwdefaults__))
    return tuple(methods)


def freeze_schedule(selected, target):
    if selected is None:
        return None
    from .target_schedule import TargetScheduleSelection

    if type(selected) is not TargetScheduleSelection:
        raise ValueError("target schedule requires an original TargetScheduleSelection")
    methods, sources = _methods(selected), _sources()
    selected.revalidate()
    if selected.selected_contracts.target != target:
        raise ValueError("target schedule selection names a different target")
    record = json.dumps(selected.record(), sort_keys=True, allow_nan=False)
    frozen = selected, record, methods, sources
    require_schedule(selected, target, frozen)
    return frozen


def require_schedule(selected, target, frozen):
    if frozen is None:
        if selected is not None:
            raise ValueError("target schedule selection changed during build")
        return None
    original, record, methods, sources = frozen
    if selected is not original or _sources() != sources:
        raise ValueError("target schedule selection or transport owner changed during build")
    current = _methods(selected)
    if any(a[1] is not b[1] or a[2] is not b[2] or a[3:] != b[3:] for a, b in zip(current, methods, strict=True)):
        raise ValueError("target schedule selection methods changed during build")
    methods[1][0]()
    if (
        selected.selected_contracts.target != target
        or json.dumps(methods[0][0](), sort_keys=True, allow_nan=False) != record
    ):
        raise ValueError("target schedule context changed during build")
    if _sources() != sources or any(
        a[1] is not b[1] or a[2] is not b[2] or a[3:] != b[3:] for a, b in zip(_methods(selected), methods, strict=True)
    ):
        raise ValueError("target schedule owner changed during original verification")
    return selected


def schedule_artifacts(device, selected, artifacts):
    if not artifacts or not artifacts.get("kernels"):
        raise ValueError("selected target schedule has no consumed package IR in this route")
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen.package_runtime import load_package

    pins = tuple(kernel["custody"].get("target_schedule") for kernel in artifacts["kernels"].values())
    if any(type(pin) is not dict for pin in pins):
        raise ValueError("selected target schedule was not consumed by the object builder")
    package = load_package(str(device.package_dir))
    paths = []
    for pin in pins:
        document = CC.revalidate_target_schedule(
            package, Path(pin["path"]).parent, selected, expected_receipt_sha256=pin["sha256"]
        )
        paths.extend(
            (Path(pin["path"]).absolute(), *(Path(row["path"]).absolute() for row in document["files"].values()))
        )
    return tuple(dict.fromkeys(paths))
