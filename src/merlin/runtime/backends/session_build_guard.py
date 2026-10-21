"""Invocation-local snapshots for the existing session build boundaries."""

import inspect
from dataclasses import _HAS_DEFAULT_FACTORY, fields
from types import FunctionType

from merlin.common import invocation_record
from merlin.compile.scheduling import schedule_pipeline
from merlin.llvmlower import (
    compilation_recipe,
    device_build,
    device_preparation,
    lower,
    session_bundle,
    session_freestanding,
    session_objects,
    toolchain,
)
from merlin.llvmlower.source_multioutput import _callback, _check_callback
from merlin.runtime import elf_audit, execution_memory
from merlin.targetgen.contract import elf_admission, execution_service


def _defaults(fn):
    # Generated dataclass initializers use this immutable stdlib identity sentinel.
    defaults = tuple(
        ("dataclass_factory", id(value)) if value is _HAS_DEFAULT_FACTORY else value
        for value in (fn.__defaults__ or ())
    )
    return defaults, fn.__kwdefaults__


def snapshot(modules, devices):
    reader, code = schedule_pipeline._rewrite_snapshot, schedule_pipeline._rewrite_snapshot.__code__
    modules = (
        *modules,
        invocation_record,
        elf_admission,
        execution_service,
        elf_audit,
        execution_memory,
        compilation_recipe,
        device_build,
        device_preparation,
        lower,
        session_bundle,
        session_freestanding,
        session_objects,
        toolchain,
    )
    functions, paths, callbacks = [], {schedule_pipeline.__file__}, []
    for module in modules:
        paths.add(module.__file__)
        owners = [
            module,
            *(
                value
                for value in vars(module).values()
                if isinstance(value, type) and value.__module__ == module.__name__
            ),
        ]
        for owner in owners:
            for name, value in vars(owner).items():
                if type(value) is not FunctionType or not value.__module__.startswith("merlin."):
                    continue
                functions.append((owner, name, value, value.__code__, reader(_defaults(value))))
                while hasattr(value, "__wrapped__"):
                    wrapped = value.__wrapped__
                    functions.append(
                        (
                            value,
                            "__wrapped__",
                            wrapped,
                            wrapped.__code__,
                            reader(_defaults(wrapped)),
                        )
                    )
                    value = wrapped
                path = inspect.getsourcefile(value)
                if path and not path.startswith("<"):
                    paths.add(path)
    routes = []
    for device in devices:
        if device is None:
            routes.append(None)
            continue
        routes.append(
            tuple(
                (
                    field.name,
                    getattr(device, field.name),
                    reader(getattr(device, field.name)) if field.name == "numeric_policy" else None,
                )
                for field in fields(device)
            )
        )
        for name in (
            "select",
            "prepared_transform",
            "catalog_builder",
            "post_offload_transform",
            "final_elf_audit",
            "descriptor_effects",
        ):
            callback = getattr(device, name)
            if callback is not None:
                callbacks.append(
                    (
                        device,
                        name,
                        callback,
                        _callback(callback),
                        reader(
                            (
                                getattr(callback, "__func__", callback).__defaults__,
                                getattr(callback, "__func__", callback).__kwdefaults__,
                            )
                        ),
                    )
                )
                paths.add(callbacks[-1][3][3])
        service = device.linked_elf_admission
        if service is not None:
            for name in ("verify", "evaluate", "revalidate"):
                callback = getattr(service, name)
                fn = getattr(callback, "__func__", callback)
                callbacks.append(
                    (service, name, callback, _callback(callback), reader((fn.__defaults__, fn.__kwdefaults__)))
                )
                paths.add(callbacks[-1][3][3])
    return dict(
        reader=(reader, code), functions=tuple(functions), callbacks=tuple(callbacks), routes=tuple(routes), paths=paths
    )


def check(snapshot, devices):
    reader, code = snapshot["reader"]
    if schedule_pipeline._rewrite_snapshot is not reader or reader.__code__ is not code:
        raise ValueError("session implementation snapshot reader changed")
    for owner, name, value, code, defaults in snapshot["functions"]:
        if getattr(owner, name) is not value or value.__code__ is not code or reader(_defaults(value)) != defaults:
            raise ValueError("session original implementation function or defaults changed")
    for device, route in zip(devices, snapshot["routes"], strict=True):
        if route is None:
            if device is not None:
                raise ValueError("session device routing appeared")
            continue
        for name, value, state in route:
            if getattr(device, name) is not value or (name == "numeric_policy" and reader(value) != state):
                raise ValueError("session original device routing changed")
    for owner, name, callback, original, defaults in snapshot["callbacks"]:
        current = getattr(owner, name)
        _check_callback(current, original)
        fn = getattr(current, "__func__", current)
        if reader((fn.__defaults__, fn.__kwdefaults__)) != defaults:
            raise ValueError("session original routing callback defaults changed")
