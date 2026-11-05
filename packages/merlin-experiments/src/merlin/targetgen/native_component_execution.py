"""Private independent-component inputs and the shared ordinary native route.

The package sees only its source interface through the caller's scoped executor.
The trusted renderer gets the independent input projection. Answers remain in this
owner for complete output comparison under the capsule's original policy.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

from merlin.common import execution_deadline as _deadline_owner
from merlin.common import invocation_record
from merlin.common.execution_deadline import ExecutionDeadline
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import native_component_inputs as input_binding
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.native_component_inputs import (
    NativeComponentExecutionError,
)
from merlin.targetgen.native_component_inputs import (
    _bind as _bind,
)
from merlin.targetgen.native_component_inputs import (
    _match as _match,
)
from merlin.targetgen.native_component_inputs import (
    _source_signature as _source_signature,
)


class NativeComponentAdmissionRefusal(NativeComponentExecutionError):
    """An actually evaluated linked-artifact gate refused before execution."""

    def __init__(self, result):
        self.result = result
        super().__init__("linked ELF admission refused before simulator dispatch")


def _digest(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _build_artifacts(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _plain(path: Path) -> Path:
    path = Path(path).absolute()
    if any(member.is_symlink() for member in (path, *path.parents)) or path.resolve() != path:
        raise NativeComponentExecutionError(f"independent native input is indirect: {path}")
    return path


def _tree(root: Path) -> dict:
    _plain(root)
    if not root.is_dir():
        raise NativeComponentExecutionError(f"independent native source directory is absent: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        _plain(path)
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _digest(path)
        elif not path.is_dir():
            raise NativeComponentExecutionError(f"independent native source contains a special file: {path}")
    if not files:
        raise NativeComponentExecutionError("independent native source directory is empty")
    return files


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _publish_result(output, record, deadline):
    # Closing failed evidence remains mandatory after expiry. Recheck both
    # sides of publication so a late result cannot retain completed status.
    for write in (False, True):
        if write:
            _write(output / "result.json", record)
        if "failure" not in record:
            try:
                deadline.remaining()
            except TimeoutError as error:
                record.update(status="unavailable", failure={"type": type(error).__name__, "detail": str(error)})
                _write(output / "result.json", record)
                raise


def execute_component(
    *,
    package_dir: Path,
    capsule_dir: Path,
    contract_root: Path,
    target: str,
    out_dir: Path,
    build_service,
    execution_service,
    source_verifier,
    readback_policy,
    timeout_s: int,
    elf_admission=None,
    memory_readback=None,
    compiler_library=None,
    compiler_library_root=None,
    original_member=None,
) -> dict:
    from merlin.targetgen import package_runtime as P
    from merlin.targetgen.compiler_library import selected_library_record
    from merlin.targetgen.contract.compile import run_on_oracle
    from merlin.targetgen.contract.compile_only import require_pointer_entry

    from . import capsule_golden as CG

    if original_member is not None:
        from merlin_experiments.phase1.component_original_members import OriginalCandidateMember

        if type(original_member) is not OriginalCandidateMember:
            raise NativeComponentExecutionError("original execution requires the actual live original candidate member")
        try:
            original_before = original_member.verify(Path(capsule_dir).absolute())
        except Exception:
            raise NativeComponentExecutionError(
                "original candidate source/reference selection is unavailable"
            ) from None

    if (
        type(timeout_s) is not int
        or not 0 < timeout_s <= 600
        or type(build_service) is not BuildOnlyService
        or type(execution_service) is not FunctionalExecutionService
        or not callable(source_verifier)
        or type(readback_policy) is not RB.ReadbackPolicy
    ):
        raise NativeComponentExecutionError(
            "independent native route requires explicit bounded build/execution/source services and full readback"
        )
    memory = readback_policy.transport in RB.MEMORY_TRANSPORTS
    reader_pins, reader_callbacks = [], {}
    if memory:
        for name in ("prepare", "decode"):
            callback = getattr(memory_readback, name, None)
            try:
                owner = inspect.getsourcefile(callback) if callable(callback) else None
            except TypeError:
                owner = None
            if owner is None:
                raise NativeComponentExecutionError("independent memory readback requires a source-bound reader")
            path = _plain(Path(owner))
            pin = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
            if pin not in execution_service.source_pins:
                raise NativeComponentExecutionError("memory reader is outside the selected functional source closure")
            reader_pins.append(pin)
            function = getattr(callback, "__func__", callback)
            reader_callbacks[name] = (
                function,
                getattr(callback, "__self__", None),
                getattr(function, "__code__", None),
            )
    elif memory_readback is not None:
        raise NativeComponentExecutionError("memory reader requires an explicitly selected memory policy")

    def verify_reader():
        for name, (function, instance, code) in reader_callbacks.items():
            actual = getattr(memory_readback, name, None)
            current = getattr(actual, "__func__", actual)
            if (
                current is not function
                or getattr(actual, "__self__", None) is not instance
                or getattr(current, "__code__", None) is not code
            ):
                raise NativeComponentExecutionError("selected memory reader callback changed")

    if P.active_package_executor() is None:
        raise NativeComponentExecutionError("independent native route requires an explicit scoped package executor")
    deadline = ExecutionDeadline.start(timeout_s)
    package_dir, capsule_dir, contract_root, output = map(_plain, (package_dir, capsule_dir, contract_root, out_dir))
    if output.exists() or any(output.is_relative_to(root) for root in (package_dir, capsule_dir, contract_root)):
        raise NativeComponentExecutionError("independent native evidence destination must be fresh and separate")
    frozen = {"package": _tree(package_dir), "capsule": _tree(capsule_dir), "contract": _tree(contract_root)}
    binder = _digest(_plain(Path(input_binding.__file__)))
    library = selected_library_record(compiler_library, compiler_library_root)
    library_sources = (
        tuple(compiler_library_root / member.path for member in compiler_library.members) if library else ()
    )
    build_service.verify(target)
    execution_before = execution_service.verify(target, execution_service.simulator)
    output.mkdir(parents=True, mode=0o700)
    generated = output / "generated"
    record = {
        "schema": "merlin.independent_component_execution.v1",
        "status": "unavailable",
        "scope": "ordinary source/build/full-output diagnostic; "
        "semantic stages, effects, hardware and cost remain unqualified",
        "target": target,
        "readback_policy": readback_policy.record(),
        **({"memory_reader_source_pins": sorted(set(reader_pins))} if memory else {}),
        "inputs": frozen,
        "input_binding_source": binder,
        **({"compiler_library": library} if library is not None else {}),
    }

    frozen_projection = None

    def unchanged():
        if _digest(_plain(Path(input_binding.__file__))) != binder:
            raise NativeComponentExecutionError("independent native input binder changed")
        if original_member is not None and original_member.verify(capsule_dir) != original_before:
            raise NativeComponentExecutionError("original candidate source/reference member changed")
        if selected_library_record(compiler_library, compiler_library_root) != library:
            raise NativeComponentExecutionError("independent native selected compiler library changed")
        current = {"package": _tree(package_dir), "capsule": _tree(capsule_dir), "contract": _tree(contract_root)}
        if current != frozen:
            raise NativeComponentExecutionError("independent native source/candidate/contract membership changed")
        build_service.verify(target)
        if execution_service.verify(target, execution_service.simulator) != execution_before:
            raise NativeComponentExecutionError("independent native functional transport changed")
        verify_reader()
        if any(_digest(_plain(Path(row["path"]))) != row for row in record.get("emission", {}).values()):
            raise NativeComponentExecutionError("independent native emitted source products changed")
        if frozen_projection is not None and (bound, inputs) != frozen_projection:
            raise NativeComponentExecutionError("independent native bound source projection changed")

    try:
        capsule = (
            {**original_before["envelope"], "__dir__": str(capsule_dir)}
            if original_member is not None
            else CC.load_capsule(capsule_dir, contract=contract_root)
        )
        if original_member is not None:
            record["original_member"] = {
                "owner_sha256": original_member.owner.sha256,
                "source_slot": original_member.source_slot,
                "binding": original_before,
            }
        source = _plain(capsule_dir / capsule.get("interface_mlir", "capsule.interface.mlir"))
        if not source.is_relative_to(capsule_dir) or not source.is_file():
            raise NativeComponentExecutionError("independent native source interface escaped its frozen capsule")
        package = P.load_package(package_dir, contract=contract_root)
        if package.manifest.get("target") != target:
            raise NativeComponentExecutionError("independent native package differs from selected target")
        deadline.remaining()
        P.integrity_scan(
            package,
            **(
                {"compiler_library": compiler_library, "compiler_library_root": compiler_library_root}
                if library
                else {}
            ),
        )
        P.build_package(package, timeout=deadline.remaining())
        deadline.remaining()
        generated.mkdir()
        products = tuple(
            generated / name
            for name in ("input.interface.mlir", "command_buffer.json", "lowered.target.mlir", "lowered.llvm.mlir")
        )
        with invocation_record.observe_call(
            generated,
            stage="component_source_lowering",
            function=CC.lower_interface,
            arguments={"target": target, "contract_root": str(contract_root), "timeout_s": timeout_s},
            inputs=(source,),
            outputs=products,
            dependencies=(
                *tuple(Path(row["path"]) for kind in ("package", "contract") for row in frozen[kind].values()),
                *library_sources,
            ),
        ) as observation:

            def invoke(*args, **kwargs):
                kwargs["timeout"] = deadline.remaining()
                kwargs["invocation_directory"] = generated
                result = P.run_entrypoint(*args, **kwargs)
                deadline.remaining()
                return result

            cb, artifact = CC.lower_interface(
                package, source, generated, contract=contract_root, timeout=deadline.remaining(), invoke=invoke
            )
            observation.returned(stdout=artifact)
        deadline.remaining()
        unchanged()
        entry = build_service.recipe.require_kernel_stack_frame().entry_symbol
        if (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeComponentExecutionError("independent native artifact has no exact whole-program entry ABI")
        try:
            require_pointer_entry(
                artifact, entry_symbol=entry, pointer_arity=len((cb.get("kernel_abi") or {}).get("args") or ())
            )
        except ValueError as error:
            raise NativeComponentExecutionError(
                "independent native artifact changes its C pointer entry ABI"
            ) from error
        bound, inputs, bindings = _bind(capsule, cb, source, original_member)
        frozen_projection = copy.deepcopy((bound, inputs))
        deadline.remaining()
        expected = (
            {slot["name"]: None for slot in capsule["ordered_abi"]["outputs"]}
            if original_member is not None
            else CG.golden(capsule, capsule_dir)
        )
        deadline.remaining()
        if set(expected) != set(bindings["outputs"]):
            raise NativeComponentExecutionError(
                "independent native golden does not cover the complete declared output roster"
            )
        policy = capsule.get("numeric_policy")
        if original_member is None and (
            not isinstance(policy, dict) or policy.get("compare") not in ("exact_int", "tolerance_float")
        ):
            raise NativeComponentExecutionError("independent native numerical policy is unavailable")
        _write(generated / "command_buffer.bound.json", bound)
        _write(output / "input_projection.json", {"inputs": inputs, "bindings": bindings})
        record["emission"] = {
            name: _digest(path)
            for name, path in {
                "source_interface": source,
                "command_buffer": generated / "command_buffer.json",
                "bound_command_buffer": generated / "command_buffer.bound.json",
                "lowered_mlir": generated / "lowered.llvm.mlir",
                "target_mlir": generated / "lowered.target.mlir",
                "input_projection": output / "input_projection.json",
            }.items()
        }
        deadline.remaining()
        record["source_correspondence"] = source_verifier(
            source=source,
            command_buffer=bound,
            lowered_mlir=generated / "lowered.llvm.mlir",
            target_mlir=generated / "lowered.target.mlir",
            input_projection=output / "input_projection.json",
            package_root=package_dir,
            generated_root=generated,
        )
        deadline.remaining()
        if type(record["source_correspondence"]) is not dict:
            raise NativeComponentExecutionError("independent native source verifier returned no closed observation")
        unchanged()

        def selected_execution():
            unchanged()
            return execution_service.verify(target, execution_service.simulator)

        with invocation_record.observe_call(
            output,
            stage="component_native_execution",
            function=run_on_oracle,
            arguments={
                "target": target,
                "engine": execution_service.simulator,
                "timeout_s": timeout_s,
                "readback_policy": readback_policy.record(),
            },
            inputs=(
                generated / "command_buffer.bound.json",
                generated / "lowered.llvm.mlir",
                output / "input_projection.json",
            ),
            dependencies=(
                *tuple(Path(path) for path, _ in (*build_service.source_pins, *execution_service.source_pins)),
                Path(_deadline_owner.__file__),
                Path(input_binding.__file__),
            ),
        ) as observation:
            native = run_on_oracle(
                bound,
                artifact,
                simulator=execution_service.simulator,
                target=target,
                workdir=output / "build",
                timeout=timeout_s,
                inputs=inputs,
                readback_policy=readback_policy,
                _build_service=build_service,
                _execution_service=execution_service,
                execution_revalidate=selected_execution,
                _elf_admission=elf_admission,
                execution_deadline=deadline,
                **({"memory_readback": memory_readback, "oracle_revalidate": selected_execution} if memory else {}),
            )
            observation.returned(stdout=native["console"])
        deadline.remaining()
        unchanged()
        if native.get("status") == "refused_before_execution":
            record.update(status="refused_before_execution", native=native, elf=_digest(Path(native["elf"])))
            raise NativeComponentAdmissionRefusal(native)
        observed = {
            source_name: native["outputs"][emitted_name] for source_name, emitted_name in bindings["outputs"].items()
        }
        deadline.remaining()
        report = (
            original_member.compare_values(observed)
            if original_member is not None
            else CG.compare(expected, observed, policy, golden_source=CG.golden_source(capsule, capsule_dir))
        )
        deadline.remaining()
        native.pop("console")  # Exact raw bytes remain in oracle_console, bound below.
        record.update(
            status="numeric_match_diagnostic" if report["status"] == "pass" else "numeric_mismatch_diagnostic",
            numeric_report=report,
            native=native,
            elf=_digest(Path(native["elf"])),
            console=_digest(
                output
                / "build"
                / ("oracle_console.bin" if readback_policy.transport == RB.FULL_VALUES_BIN else "oracle_console.log")
            ),
        )
        record["invocations"] = [
            {"record": _digest(path), "stage": invocation_record.verify(path)["stage"]}
            for path in sorted(output.rglob("invocation.json"))
        ]
        deadline.remaining()
        return record
    except Exception as error:
        if isinstance(error, TimeoutError):
            record["status"] = "unavailable"
        record["failure"] = {"type": type(error).__name__, "detail": str(error)}
        if original_member is not None and not isinstance(error, NativeComponentAdmissionRefusal):
            if isinstance(error, TimeoutError):
                raise TimeoutError("original candidate execution exceeded its selected wall budget") from None
            raise NativeComponentExecutionError(
                "original candidate execution failed; inspect candidate code and declared ABI"
            ) from None
        raise
    finally:
        record["build_artifacts"] = _build_artifacts(output)
        _publish_result(output, record, deadline)
