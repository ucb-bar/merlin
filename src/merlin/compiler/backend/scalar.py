"""Ordinary scalar MLIR lowering and explicitly targeted LLVM object emission.

Reuses the existing numerical preprocessing, scalar pass pipeline and native
receipt machinery. An explicit scalar.call-entry admits observed synchronous
static tensor regions to the shared C ABI. Objects still require linking and
independent numerical verification. RVV providers are never selected here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from merlin.compiler.artifact import Artifact, Compilation, digest

name = "merlin.scalar"
_OPTIONS = {
    "scalar.data-layout",
    "scalar.index-bits",
    "scalar.emit",
    "scalar.target-flags",
    "scalar.call-entry",
    "scalar.workspace-bytes",
}
_TARGET_FLAGS = {"--target", "-march", "-mabi", "-mcpu", "-mtune", "-mcmodel"}
_SCALAR_FLAGS = ("-O2", "-fno-vectorize", "-fno-slp-vectorize")


def _parameters(request):
    if request.source.kind not in {"mlir", "upstream-mlir"}:
        raise ValueError("scalar provider requires MLIR or explicitly prepared upstream-mlir")
    if any(
        value is not None
        for value in (
            request.options.system,
            request.options.platform,
            request.options.harts,
        )
    ):
        raise ValueError("scalar object emission does not build a deployment or schedule harts")
    values = request.options.parameters
    if set(values) - _OPTIONS:
        raise ValueError(f"unsupported scalar parameters: {sorted(set(values) - _OPTIONS)}")
    if not isinstance(values.get("scalar.data-layout"), str) or not values["scalar.data-layout"].strip():
        raise ValueError("scalar emission requires explicit scalar.data-layout")
    if type(values.get("scalar.index-bits")) is not int or values["scalar.index-bits"] not in (32, 64):
        raise ValueError("scalar.index-bits must explicitly select 32 or 64")
    emission = values.get("scalar.emit", "llvm-ir")
    if not isinstance(emission, str) or emission not in {"llvm-ir", "object"}:
        raise ValueError("scalar.emit must select llvm-ir or object")
    entry = values.get("scalar.call-entry")
    workspace = values.get("scalar.workspace-bytes", 0)
    if type(workspace) is not int or workspace < 0 or "scalar.workspace-bytes" in values and entry is None:
        raise ValueError("scalar.workspace-bytes requires a nonnegative ceiling and explicit callable entry")
    if "scalar.call-entry" in values and (type(entry) is not str or not entry or emission != "object"):
        raise ValueError("scalar.call-entry requires a nonempty selected entry and object emission")
    flags = values.get("scalar.target-flags")
    if emission == "llvm-ir":
        if flags is not None:
            raise ValueError("scalar.target-flags requires object emission")
        return emission, ()
    if not isinstance(flags, list) or not flags or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("object emission needs explicit scalar.target-flags")
    selected = set()
    for flag in flags:
        key, separator, value = flag.partition("=")
        if (
            key not in _TARGET_FLAGS
            or not separator
            or not value
            or key in selected
            or value == "native"
            or any(char.isspace() for char in value)
        ):
            raise ValueError("scalar target flags need distinct explicit --target/-m selections")
        selected.add(key)
    if "--target" not in selected:
        raise ValueError("object emission needs an explicit --target triple")
    return emission, (*flags, *_SCALAR_FLAGS)


def _tool(path):
    import shutil

    executable = Path(shutil.which(str(path)) or path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"selected scalar compiler tool is unavailable: {path}")
    return executable.resolve()


def _tools(emission):
    from merlin.compiler.lower.model import toolchain

    driver = toolchain._env("MERLIN_COMPILER_DRIVER", "python")
    if driver == "stock-cli":
        selected = {"mlir-opt": toolchain.llvm_install() / "bin/mlir-opt", "mlir-translate": toolchain.mlir_translate()}
    elif driver == "python":
        selected = {"compiler-python": toolchain.compiler_python()}
    else:
        raise ValueError(f"unknown selected scalar native driver: {driver}")
    if emission == "object":
        selected["clang"] = toolchain.clang()
    return {
        key: {"path": str(path), "sha256": digest(path)} for key, value in selected.items() for path in (_tool(value),)
    }


def _source(request):
    from xdsl.dialects.builtin import VectorType

    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(request.source.text)
    module.verify()
    if any(
        op.name.startswith("vector.")
        or any(
            isinstance(value.type, VectorType)
            for value in (
                *op.operands,
                *op.results,
                *(arg for region in op.regions for block in region.blocks for arg in block.args),
            )
        )
        for op in module.walk()
    ):
        raise ValueError("scalar source cannot contain vector operations or values")


def _call_selection(request):
    entry = request.options.parameters.get("scalar.call-entry")
    if entry is None:
        return None
    from merlin.compiler.lower.llvm.call import prepare
    from merlin.compiler.lower.model import toolchain

    if request.source.kind != "mlir":
        raise ValueError("scalar call requires the original tensor MLIR source, not prepared buffer IR")
    if toolchain._env("MERLIN_COMPILER_DRIVER", "python") != "stock-cli":
        raise ValueError("scalar call currently requires the observed stock-cli LLVM dialect boundary")
    # This namespace changes linking identity only; computation selection depends
    # on source semantics. Separate invocations cannot collide on public symbols.
    identity = request.source.sha256 + "\0" + str(request.output)
    symbol = "merlin_scalar_" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    return prepare(
        request.source.text, entry, symbol, workspace_bytes=request.options.parameters.get("scalar.workspace-bytes", 0)
    )


def validate(request):
    emission, _ = _parameters(request)
    _source(request)
    _call_selection(request)
    _tools(emission)


def _call_artifacts(call, native, output, values, tools):
    from merlin.compiler.lower.llvm import call as owner
    from merlin.compiler.lower.llvm import copies as memory_owner
    from merlin.compiler.lower.model import codegen

    original = native / "model.llvm.mlir"
    before = digest(original)
    owner_source, owner_digest = Path(owner.__file__), digest(owner.__file__)
    memory_source, memory_digest = Path(memory_owner.__file__), digest(memory_owner.__file__)
    generic = output / "call.native.mlir"
    codegen._run(
        [tools["mlir-opt"]["path"], original, "--mlir-print-op-generic", "-o", generic],
        inputs=(original,),
        outputs=(generic,),
        stage="scalar_call_native_abi",
    )
    generic_text = generic.read_text()
    initialization = call.initialization
    if initialization is not None:
        for key in ("implementation", "policy_implementation"):
            initializer_owner = initialization[key]
            if digest(initializer_owner["path"]) != initializer_owner["sha256"]:
                raise ValueError("scalar destination initialization implementation changed during lowering")
    abi, wrapper, memory = owner.admit(
        generic_text, call, index_bits=values["scalar.index-bits"], data_layout=values["scalar.data-layout"]
    )
    if digest(memory_source) != memory_digest or digest(owner_source) != owner_digest:
        raise ValueError("native scalar memory admission source changed during inspection")
    generic_digest = hashlib.sha256(generic_text.encode()).hexdigest()
    if digest(generic) != generic_digest:
        raise ValueError("actual scalar interface changed during ABI inspection")
    source = output / "call.mlir"
    source.write_text(wrapper, encoding="utf-8")
    translated = output / "call.ll"
    kernel = output / "call.kernel.ll"
    pins = {
        original: before,
        generic: generic_digest,
        source: hashlib.sha256(wrapper.encode()).hexdigest(),
        owner_source: owner_digest,
        memory_source: memory_digest,
    }
    if call.workspace is not None:
        from merlin.compiler.lower.llvm import workspace as workspace_owner

        path = Path(workspace_owner.__file__)
        if call.workspace_implementation != {"path": str(path), "sha256": digest(path)}:
            raise ValueError("source tensor workspace implementation changed during lowering")
        pins[path] = call.workspace_implementation["sha256"]
    if initialization is not None:
        for key in ("implementation", "policy_implementation"):
            pins[Path(initialization[key]["path"])] = initialization[key]["sha256"]

    def revalidate():
        if any(digest(path) != expected for path, expected in pins.items()):
            raise ValueError("actual scalar interface/source changed during ABI materialization")

    for src, dst in ((generic, kernel), (source, translated)):
        revalidate()
        codegen._run(
            [tools["mlir-translate"]["path"], "--mlir-to-llvmir", src, "-o", dst],
            inputs=(src,),
            outputs=(dst,),
            stage="scalar_call_translation",
        )
        revalidate()
    products = tuple(
        Artifact.from_file(path, kind=kind, role="call-abi")
        for path, kind in (
            (generic, "llvm-dialect"),
            (source, "llvm-dialect"),
            (translated, "llvm-ir"),
            (kernel, "llvm-ir"),
        )
    )
    record = {
        "source_entry": call.source_entry,
        "compute_entry": call.compute,
        "symbol": call.symbol,
        "native_module_sha256": before,
        "generic_module_sha256": digest(generic),
        "wrapper_source_sha256": digest(source),
        "kernel_llvm_sha256": digest(kernel),
        "index_bits": values["scalar.index-bits"],
        "memory": memory,
        "abi_owner": {"path": str(owner_source), "sha256": owner_digest},
        "memory_owner": {"path": str(memory_source), "sha256": memory_digest},
        "descriptor_storage": "LLVM-layout-owned invocation-local ABI bookkeeping; no tensor workspace",
        "numerical_correctness": "not_established",
        "initialization": initialization,
    }
    if call.workspace is not None:
        record.update(
            workspace=call.workspace.record(),
            workspace_owner={"path": str(path), "sha256": pins[path]},
            descriptor_storage="LLVM-layout-owned descriptors; source tensor workspace is caller-owned",
        )
    return abi, products, kernel.read_text(), record


def compile(request):
    from merlin.compiler.build.objects import compile_llvm_object
    from merlin.compiler.lower.model import codegen
    from merlin.compiler.lower.model.target_data_layout import observe_index_width
    from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir

    emission, flags = _parameters(request)
    _source(request)
    call = _call_selection(request)
    tools = _tools(emission)
    values, output = request.options.parameters, request.output
    if output.exists():
        raise FileExistsError(f"scalar compilation requires fresh output: {output}")
    layout = None
    if emission == "object":
        layout = observe_index_width(tools["clang"]["path"], flags)
        if layout["data_layout"] != values["scalar.data-layout"] or layout["index_bits"] != values["scalar.index-bits"]:
            raise ValueError("selected scalar layout/index width differs from actual object compiler")
    output.mkdir(parents=True)
    try:
        native = output / "native"
        if request.source.kind == "mlir":
            from merlin.compiler.lower.model.lower import lower_model

            lowered = lower_model(
                call.source if call is not None else request.source.text,
                native,
                targets=(),
                vectorize=False,
                parallel=False,
                features=frozenset(),
                data_layout=values["scalar.data-layout"],
                index_bits=values["scalar.index-bits"],
            )
            llvm_text, stats = lowered.ll_path.read_text(), lowered.stats
        else:
            selection = {}
            llvm_text = lower_to_llvm_ir(
                request.source.text,
                workdir=native,
                features=frozenset(),
                vectorize=False,
                parallel=False,
                data_layout=values["scalar.data-layout"],
                index_bits=values["scalar.index-bits"],
                lowering_selection=selection,
            )
            stats = {"index_lowering": selection}
        abi, call_products, call_record = None, (), None
        if call is not None:
            abi, call_products, llvm_text, call_record = _call_artifacts(call, native, output, values, tools)
        llvm = output / "module.ll"
        llvm.write_text(llvm_text, encoding="utf-8")
        artifacts = [Artifact.from_file(llvm, kind="llvm-ir", role="region"), *call_products]
        if emission == "object":
            obj = compile_llvm_object(
                llvm,
                output / "module.o",
                compiler=tools["clang"]["path"],
                flags=flags,
                run=codegen._run,
            )
            artifacts.append(Artifact.from_file(obj, kind="object", role="region"))
            if call is not None:
                wrapper = compile_llvm_object(
                    output / "call.ll",
                    output / "call.o",
                    compiler=tools["clang"]["path"],
                    flags=flags,
                    run=codegen._run,
                )
                artifacts.append(Artifact.from_file(wrapper, kind="object", role="call-abi"))
        if _tools(emission) != tools:
            raise ValueError("selected scalar compiler changed during compilation")
        record = output / "scalar.json"
        record.write_text(
            json.dumps(
                {
                    "schema": "merlin.scalar.v1",
                    "source_sha256": request.source.sha256,
                    "source_kind": request.source.kind,
                    "tools": tools,
                    "target_flags": list(flags),
                    "selected_layout": layout,
                    "stats": stats,
                    "scope": (
                        "scalar_lowering; explicit_target_object; no_linked_image_or_execution_qualification"
                        if call_record is not None
                        else "scalar_lowering; object_requires_explicit_target; no_linked_image_or_dispatch_ABI"
                    ),
                    **({"call": call_record} if call_record is not None else {}),
                    "execution_observed": False,
                    "numerical_verification": "not_established",
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(Artifact.from_file(record, kind="json", role="provider-record"))
        artifacts.extend(
            Artifact.from_file(path, kind="native-artifact", role="trace")
            for path in sorted(native.rglob("*"))
            if path.is_file()
        )
        return Compilation(
            output,
            name,
            request.source.sha256,
            tuple(artifacts),
            metadata={
                "stage": emission,
                "execution_observed": False,
                **({"c_call": abi.record()} if abi is not None else {}),
            },
        )
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"type": type(error).__name__, "reason": str(error)}) + "\n")
        raise
