"""Live custody for an actual original-profile analysis invocation.

The receipt observes one original executor, including its temporary input and
output before their deletion. It cannot be reconstructed from saved JSON and
does not grant numerical, hardware, timing or final-stage qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import MethodType

from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compiler_library import active_compiler_library
from merlin.targetgen.compiler_pass_selection import _pin, _tree

from . import compiler_profile_runtime as R

SCHEMA = "merlin.original_profile_compiler_emission.v2"
_PENDING = weakref.WeakSet()
_ISSUED = weakref.WeakSet()


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _copy(source, destination):
    source = Path(source)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 64 << 20:
        raise ValueError("original profile product is absent, linked or oversized")
    pin = _pin(source)
    destination.write_bytes(source.read_bytes())
    if _pin(source) != pin or _pin(destination)["sha256"] != pin["sha256"]:
        raise ValueError("original profile product changed while retaining it")
    return _pin(destination)


def _sources():
    return [_pin(Path(__file__)), _pin(Path(R.__file__)), _pin(Path(P.__file__))]


def _methods():
    functions = (
        R._Executor.run_entrypoint,
        R._Executor._invoke,
        R._Executor._verify,
        R.original_compiler_execution,
        R.CommonTransformService._verify,
        P.run_entrypoint,
    )
    return tuple((function, function.__code__) for function in functions)


class OriginalProfileEmission:
    """Only ``emit_analysis`` may issue this original live observation."""

    def __init__(self, *args, **kwargs):
        raise TypeError("original profile emission requires an actual original executor")

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)

    def revalidate(self):
        if self not in _ISSUED or _digest(self._record) != self._identity:
            raise ValueError("original profile emission lacks its unchanged live issuer")
        if _sources() != self._source_pins or _methods() != self._method_pins:
            raise ValueError("original profile emission implementation changed")
        R.CommonTransformService._verify(self._service)
        record = self._record
        if (None if self._library is None else self._library.record()) != record["compiler_library"]:
            raise ValueError("original profile emission selected library changed")
        if (
            self._service.snapshot_digest != record["profile"]["snapshot_sha256"]
            or self._service.profile["sha256"] != record["profile"]["profile_sha256"]
            or _pin(self._service.bundle_manifest) != record["profile"]["archive"]
            or _tree(Path(record["compiler_package"]["path"])) != record["compiler_package"]["sha256"]
        ):
            raise ValueError("original profile emission selection or compiler changed")
        for key in ("source", "copied_input", "copied_output", "command_buffer", "lowered", "invocation"):
            pin = record[key]
            if _pin(pin["path"]) != pin:
                raise ValueError("original profile emission artifact changed: " + key)
        observed = invocation_record.verify(Path(record["invocation"]["path"]))
        if observed != self._invocation:
            raise ValueError("original profile invocation changed")
        return record["status"]

    def check_native_artifacts(self, owner, arm):
        """Bind this live producer to current complete native/source artifacts."""
        self.revalidate()
        binding = getattr(owner, "check_compiler_binding", None)
        if type(binding) is not MethodType or binding.__self__ is not owner:
            raise ValueError("profile emission requires its original bound compiler checker")
        record = self._record
        if arm.get("compiler_package") != record["compiler_package"]:
            raise ValueError("profile emission selected another compiler package")
        binding(Path(record["compiler_package"]["path"]))
        for arm_key, record_key in (("command_buffer_artifact", "command_buffer"), ("lowered", "lowered")):
            if arm.get(arm_key) != record[record_key]:
                raise ValueError("profile emission selected another native input")
        native_pin = arm["native_result"]
        if _pin(native_pin["path"]) != native_pin:
            raise ValueError("profile emission native artifact changed")
        native = json.loads(Path(native_pin["path"]).read_bytes())
        declaration, interface = (native["source"][key] for key in ("capsule_declaration", "interface"))
        if any(_pin(pin["path"]) != {key: pin[key] for key in ("path", "sha256")} for pin in (declaration, interface)):
            raise ValueError("profile emission native source custody changed")
        source = Path(declaration["path"]).parent / "capsule.interface.mlir"
        if _pin(source) != record["source"] or source.read_bytes() != Path(interface["path"]).read_bytes():
            raise ValueError("profile emission native source differs from its actual original input")
        owner.revalidate()
        self.revalidate()
        return self.record()


def retain_original_products(observation, executor, pkg, name, original, incoming, outgoing, result, directory):
    """Called only while the original executor still owns its exact temp files."""
    if (
        type(observation) is not OriginalProfileEmission
        or observation not in _PENDING
        or type(executor) is not R._Executor
        or P.active_package_executor() is not executor
        or executor.owner is not observation._service
        or name != "emit_analysis_bundle"
        or outgoing is None
        or result.returncode != 0
        or Path(directory) != observation._invocation_directory
        or Path(original) != observation._input
        or pkg.directory != observation._package
    ):
        raise ValueError("profile emission did not observe its original selected invocation")
    if getattr(observation, "_products", None) is not None:
        raise ValueError("profile emission cannot observe more than one invocation")
    executor._verify(pkg)
    original_pin = _pin(original)
    if original_pin != observation._input_pin or Path(incoming).read_bytes() != Path(original).read_bytes():
        raise ValueError("profile emission copied input differs from its original selected source")
    paths = tuple(Path(directory).rglob("invocation.json"))
    if len(paths) != 1:
        raise ValueError("profile emission requires its one original process invocation")
    invocation = invocation_record.verify(paths[0])
    expected = P._resolve_argv(pkg, name, incoming, outgoing)
    if P._needs_interpreter(pkg, expected):
        expected = [sys.executable, *expected]
    if (
        invocation["kind"] != "subprocess"
        or invocation["stage"] != "original_profile_package"
        or invocation["status"] != "completed"
        or invocation["returncode"] != 0
        or invocation["inputs"] != [original_pin]
        or invocation["stdout"]["sha256"] != hashlib.sha256(result.stdout.encode()).hexdigest()
        or invocation["argv"][-len(expected) :] != expected
    ):
        raise ValueError("profile emission products differ from the original actual process")
    observation._products = dict(
        copied_input=_copy(incoming, observation._root / "boxed-input.mlir"),
        copied_output=_copy(outgoing, observation._root / "boxed-output.json"),
        invocation=_pin(paths[0]),
    )
    observation._invocation = invocation


def emit_analysis(
    *,
    service,
    target_experiment,
    package,
    input_mlir,
    command_buffer,
    lowered,
    work_root,
    sandbox_inputs=None,
    timeout=60,
):
    """Run the existing original box and retain one live source/product receipt.

    No outer CPU lease: the original executor owns metadata and command loans.
    This operation does not restore a pass factory or change the selected SDK.
    """
    if type(service) is not R.CommonTransformService or R._EMISSION_OBSERVATION.get() is not None:
        raise ValueError("profile emission requires one exact original service and no nested observation")
    root, package, source = map(Path, (work_root, package, input_mlir))
    command_buffer, lowered = map(Path, (command_buffer, lowered))
    if root.exists() or root.is_symlink() or not root.is_absolute() or root.resolve() != root:
        raise ValueError("profile emission needs a fresh canonical private root")
    if root.is_relative_to(package) or any(path.exists() or path.is_symlink() for path in (command_buffer, lowered)):
        raise ValueError("profile emission cannot overwrite an artifact or stage inside the candidate")
    root.mkdir(parents=True, mode=0o700)
    observation = OriginalProfileEmission.__new__(OriginalProfileEmission)
    observation._root, observation._service = root, service
    observation._package, observation._input = package, source
    observation._invocation_directory = root / "original-invocation"
    with shared_compute_pool():
        R.CommonTransformService._verify(service)
        observation._input_pin = _pin(source)
        observation._source_pins = _sources()
        observation._method_pins = _methods()
        observation._library = active_compiler_library()
        library = None if observation._library is None else observation._library.record()
        package_pin = dict(path=str(package), sha256=_tree(package))
        profile = dict(
            archive=_pin(service.bundle_manifest),
            profile_sha256=service.profile["sha256"],
            snapshot_sha256=service.snapshot_digest,
        )
        pkg = P.load_package(package)
        P.integrity_scan(pkg)
    _PENDING.add(observation)
    token = R._EMISSION_OBSERVATION.set(observation)
    try:
        with R.original_compiler_execution(
            service,
            target_experiment=target_experiment,
            candidate=package,
            work_root=root / "execution",
            sandbox_inputs=sandbox_inputs,
        ):
            result = P.run_entrypoint(
                pkg,
                "emit_analysis_bundle",
                source,
                command_buffer,
                timeout=timeout,
                invocation_directory=observation._invocation_directory,
            )
        with shared_compute_pool():
            if result.returncode or getattr(observation, "_products", None) is None:
                raise ValueError("profile emission has no successfully observed original invocation")
            if active_compiler_library() is not observation._library:
                raise ValueError("profile emission changed the original selected library owner")
            lowered.write_text(result.stdout)
            products = observation._products
            if _pin(command_buffer)["sha256"] != products["copied_output"]["sha256"]:
                raise ValueError("profile emission published a different command buffer")
            observation._record = dict(
                schema=SCHEMA,
                status="checked_original_profile_compiler_emission",
                profile=profile,
                compiler_library=library,
                compiler_package=package_pin,
                source=observation._input_pin,
                command_buffer=_pin(command_buffer),
                lowered=_pin(lowered),
                **products,
                authority="original live emission only; numerical/hardware/timing/final gates unchanged",
            )
            observation._identity = _digest(observation._record)
            _ISSUED.add(observation)
            observation.revalidate()
        return result, observation
    except BaseException:
        _ISSUED.discard(observation)
        raise
    finally:
        R._EMISSION_OBSERVATION.reset(token)
        _PENDING.discard(observation)
