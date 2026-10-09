"""Private independent-component inputs and the shared ordinary native route.

The package sees only its source interface through the caller's scoped executor.
The trusted renderer gets the independent input projection. Answers remain in this
owner for complete output comparison under the capsule's original policy.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

from merlin.common import execution_deadline as _deadline_owner
from merlin.common import invocation_record
from merlin.common.execution_deadline import ExecutionDeadline
from merlin.targetgen import capsule_common as CC
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.execution_service import FunctionalExecutionService


class NativeComponentExecutionError(RuntimeError):
    """The explicit diagnostic route cannot establish its declared input joins."""


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


def _match(spec, emitted):
    from merlin.targetgen.contract.tensor_types import match_tensor_spec

    try:
        match_tensor_spec(spec, emitted)
    except ValueError as error:
        raise NativeComponentExecutionError(
            f"independent native input/output changes declared shape or dtype: {spec['name']}"
        ) from error


def _source_signature(source: Path, inputs: list, outputs: list) -> None:
    """Bind ordered static tensor types before accepting positional ABI names."""
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.func import FuncOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    from merlin.xdsl_dialects._common import make_context

    module = Parser(make_context(Arith, Tensor, Linalg, Math), source.read_text()).parse_module()
    module.verify()
    functions = list(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp or not functions[0].body.blocks:
        raise NativeComponentExecutionError("positional independent native source has no unique tensor function")
    signature = functions[0].function_type
    for expected, observed in ((inputs, signature.inputs.data), (outputs, signature.outputs.data)):
        if len(expected) != len(observed):
            raise NativeComponentExecutionError("positional independent native source tensor arity differs")
        for spec, value_type in zip(expected, observed, strict=True):
            if not isinstance(value_type, TensorType):
                raise NativeComponentExecutionError("positional independent native source is not tensor-valued")
            _match(spec, {"shape": list(value_type.get_shape()), "dtype": str(value_type.get_element_type())})


def _bind(capsule: dict, cb: dict, source: Path) -> tuple[dict, dict, dict]:
    """Project independently selected inputs, retaining compiler and harness names."""
    from merlin.runtime.commandbuffer import whole_program_entry_bindings

    from . import capsule_golden as CG

    inputs = [row for row in capsule["inputs"] if row.get("role") in ("input", "weight", "bias")]
    outputs = [row for row in capsule["inputs"] if row.get("role") == "output"]
    if not outputs:
        outputs = (capsule.get("component_program") or {}).get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise NativeComponentExecutionError("independent native capsule has no complete typed source output roster")
    names = [row["name"] for row in inputs]
    output_names = [row["name"] for row in outputs]
    if len(set(names)) != len(names) or len(set(output_names)) != len(output_names):
        raise NativeComponentExecutionError("independent native input/output roster repeats a name")
    declared_order = ((capsule.get("operation") or {}).get("attributes") or {}).get("arg_order", names)
    if (
        not isinstance(declared_order, list)
        or len(declared_order) != len(set(declared_order))
        or set(declared_order) not in (set(names), set(names + output_names))
    ):
        raise NativeComponentExecutionError("independent native source argument order is incomplete")
    declared_order = [name for name in declared_order if name in names]
    inputs = [next(row for row in inputs if row["name"] == name) for name in declared_order]
    names = declared_order
    tensors = cb.get("tensors") or {}
    leaves = whole_program_entry_bindings(cb)
    if leaves is None:
        leaves = [name for name, spec in tensors.items() if spec.get("role") in ("input", "weight", "bias")]
    emitted_outputs = (cb.get("kernel_abi") or {}).get("outputs")
    if not isinstance(emitted_outputs, list) or len(emitted_outputs) != len(set(emitted_outputs)):
        raise NativeComponentExecutionError("independent native kernel output roster is absent or repeated")
    if len(leaves) != len(names) or len(emitted_outputs) != len(output_names):
        raise NativeComponentExecutionError("independent native ABI does not cover every input/output")
    positional = cb.get("operand_naming") == "positional" or cb.get("interface") == "linalg_positional"
    if positional:
        _source_signature(source, inputs, outputs)
    elif set(leaves) != set(names) or set(emitted_outputs) != set(output_names):
        raise NativeComponentExecutionError("independent native ABI changes named source inputs/outputs")
    else:
        leaves, emitted_outputs = names, output_names
    for spec, name in zip((*inputs, *outputs), (*leaves, *emitted_outputs), strict=True):
        _match(spec, tensors.get(name))
    values = CG.canonical_input_values(capsule, capsule["__dir__"])
    if names and not values:
        if CG.is_independent_float_golden(capsule, capsule["__dir__"]):
            raise NativeComponentExecutionError("independent floating source has no complete selected input projection")
        values = CG.materialized_input_values(capsule)
    if set(values) != set(names):
        raise NativeComponentExecutionError("independent native canonical input roster is incomplete")
    bound = copy.deepcopy(cb)
    projected, bindings = {}, []
    raws = CG.canonical_input_raws(capsule, capsule["__dir__"])
    for spec, name in zip(inputs, leaves, strict=True):
        value = values[spec["name"]]
        if value.get("shape") != spec["shape"] or len(value.get("values", [])) != math.prod(spec["shape"]):
            raise NativeComponentExecutionError("independent native canonical input values have the wrong shape")
        projected[name] = value
        raw = raws.get(spec["name"])
        if raw is not None:
            bound["tensors"][name]["preload_b64"] = base64.b64encode(raw).decode()
        bindings.append(
            {
                "source": spec["name"],
                "harness": name,
                "shape": spec["shape"],
                "dtype": spec["dtype"],
                "raw_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            }
        )
    return bound, projected, {"inputs": bindings, "outputs": dict(zip(output_names, emitted_outputs, strict=True))}


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
) -> dict:
    from merlin.targetgen import package_runtime as P
    from merlin.targetgen.contract.compile import run_on_oracle
    from merlin.targetgen.contract.compile_only import require_pointer_entry

    from . import capsule_golden as CG

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
    }

    def unchanged():
        current = {"package": _tree(package_dir), "capsule": _tree(capsule_dir), "contract": _tree(contract_root)}
        if current != frozen:
            raise NativeComponentExecutionError("independent native source/candidate/contract membership changed")
        build_service.verify(target)
        if execution_service.verify(target, execution_service.simulator) != execution_before:
            raise NativeComponentExecutionError("independent native functional transport changed")
        verify_reader()

    try:
        capsule = CC.load_capsule(capsule_dir, contract=contract_root)
        source = _plain(capsule_dir / capsule.get("interface_mlir", "capsule.interface.mlir"))
        if not source.is_relative_to(capsule_dir) or not source.is_file():
            raise NativeComponentExecutionError("independent native source interface escaped its frozen capsule")
        package = P.load_package(package_dir, contract=contract_root)
        if package.manifest.get("target") != target:
            raise NativeComponentExecutionError("independent native package differs from selected target")
        deadline.remaining()
        P.integrity_scan(package)
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
            dependencies=tuple(Path(row["path"]) for kind in ("package", "contract") for row in frozen[kind].values()),
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
        bound, inputs, bindings = _bind(capsule, cb, source)
        deadline.remaining()
        expected = CG.golden(capsule, capsule_dir)
        deadline.remaining()
        if set(expected) != set(bindings["outputs"]):
            raise NativeComponentExecutionError(
                "independent native golden does not cover the complete declared output roster"
            )
        policy = capsule.get("numeric_policy")
        if not isinstance(policy, dict) or policy.get("compare") not in ("exact_int", "tolerance_float"):
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
        report = CG.compare(expected, observed, policy, golden_source=CG.golden_source(capsule, capsule_dir))
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
        raise
    finally:
        record["build_artifacts"] = _build_artifacts(output)
        _publish_result(output, record, deadline)
