"""Route caller-selected device regions through the ordinary shared host compiler.

This compatibility route preserves the source forms admitted by its callbacks.
It does not choose a target, discover a backend, or approve implementation effects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from merlin.compiler.partition.callback_regions import (
    DeviceArtifact,
    DeviceRegion,
    DeviceRequest,
    WholeProgramError,
    WholeProgramResult,
    _source_function,
    typed_device_function_identity,
)


def select_host_optimization(selection: str) -> str:
    """Validate an explicitly selected existing scalar lowering alternative."""
    if selection not in ("baseline", "scalar_loops", "bounded_rne", "bounded_rne_fused"):
        raise WholeProgramError("unknown selected scalar host optimization")
    return selection


def lower_whole_program(
    source_text: str,
    workdir: str | Path,
    compile_device: Callable[[DeviceRequest], DeviceArtifact],
    *,
    regions: Callable[[Any], Sequence[DeviceRegion]] | None = None,
    retain_llvm_dialect: bool = True,
    host_driver: str = "stock-cli",
    lower_options: dict[str, Any] | None = None,
    host_optimization: str | None = None,
    reuse_device_kernels: bool = False,
    emit_device_shim: Callable,
    mlir_install: Path,
    report_metadata: dict[str, Any],
) -> WholeProgramResult:
    """Route exact device regions, then lower remaining source through Merlin.

    ``compile_device`` receives standalone original linalg text and a unique
    function symbol. It must return the corresponding compiled pointer-ABI
    function. No package loader or reference compiler is consulted. ``regions``
    can use a caller-selected capability oracle to return larger fused groups;
    outputs are checked against all live uses before any source is mutated.

    ``reuse_device_kernels`` opts a content-driven callback into sharing compiled
    code for identical typed functions after stripping provenance and local
    names. Each invocation keeps its own source operands and descriptor shim;
    tensor values and preparations are never cached by this adapter.

    Returned C glue comes from the selected ABI callback and must be compiled
    and linked together with both the host LLVM and every device artifact.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType
    from xdsl.ir import Region

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.lower.device.device_shim import LogicalBuffer
    from merlin.compiler.lower.device.outline import patch_declaration_arg_attrs, unpatched_declarations
    from merlin.compiler.lower.model.lower import lower_model

    host_optimization = select_host_optimization(host_optimization)
    if type(reuse_device_kernels) is not bool:
        raise WholeProgramError("device code sharing requires an explicit Boolean selection")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    module = parse_mlir_text(source_text)
    source_functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    candidates = [op for op in source_functions if op.sym_name.data == "forward"] or source_functions
    if len(candidates) != 1:
        raise WholeProgramError("whole-program source needs one unambiguous public tensor entry")
    entry = candidates[0]
    abi_types = tuple(entry.function_type.inputs) + tuple(entry.function_type.outputs)
    source_abi = dict(
        entry=entry.sym_name.data,
        inputs=len(entry.function_type.inputs),
        slots=[dict(shape=list(t.get_shape()), dtype=str(t.element_type)) for t in abi_types],
    )
    original_source_path = work / "original.mlir"
    original_source_path.write_text(source_text)
    existing_symbols = {getattr(getattr(op, "sym_name", None), "data", "") for op in module.body.block.ops}
    if regions is None or not callable(regions) or not callable(emit_device_shim):
        raise WholeProgramError("mixed host lowering requires explicit region and device ABI callbacks")
    if type(report_metadata) is not dict:
        raise WholeProgramError("mixed host lowering requires explicit report metadata")
    selected = tuple(regions(module))
    ordinal = {op: i for i, op in enumerate(module.walk())}
    owned = set()
    plans = []
    artifacts = []
    shims = {}
    shim_kernels = {}
    shared_artifacts = {}
    rows = []
    for index, region in enumerate(selected):
        members = set(region.operations)
        if not members:
            raise WholeProgramError("device region is empty")
        if any(
            op in owned and op.name not in ("arith.constant", "tensor.empty", "tensor.splat", "linalg.fill")
            for op in members
        ):
            raise WholeProgramError("device arithmetic regions overlap")
        if not region.outputs or len(set(region.outputs)) != len(region.outputs):
            raise WholeProgramError("device region needs distinct ordered returned tensors")
        output_roots = {value.owner for value in region.outputs}
        first = region.operations[0]
        block = first.parent
        if block is None or any(op.parent is not block for op in members) or not output_roots.issubset(members):
            raise WholeProgramError("device region must be in one block with retained result producers")
        # Place one call after all region inputs are available. A plural result
        # group is legal only when every escaping use follows that call.
        positions = {op: i for i, op in enumerate(block.ops)}
        root = max(members, key=positions.__getitem__)

        def enclosing_operation(operation):
            while operation.parent is not block:
                operation = operation.parent_op()
                if operation is None:
                    return None
            return operation

        for value in region.outputs:
            for use in value.uses:
                user = enclosing_operation(use.operation)
                if user is not None and user not in members and positions[user] < positions[root]:
                    raise WholeProgramError("device group has an escaping result before its call anchor")
        for op in members:
            for operand in op.operands:
                if operand.owner not in members and operand not in region.inputs:
                    raise WholeProgramError("region omits a live input dependency")
            for result in op.results:
                if (
                    any(enclosing_operation(use.operation) not in members for use in result.uses)
                    and result not in region.outputs
                ):
                    # Shared initializer producers may remain in the source;
                    # every other escaping result must be part of the ABI.
                    if op.name not in ("arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"):
                        raise WholeProgramError("device region omits a live output")
        for value in (*region.inputs, *region.outputs):
            if not isinstance(value.type, TensorType) or any(d <= 0 for d in value.type.get_shape()):
                raise WholeProgramError("device boundary needs static ranked tensors")
        symbol = f"merlin_device_group_{index}"
        kernel_symbol = f"{symbol}_kernel"
        if symbol in existing_symbols or kernel_symbol in existing_symbols:
            raise WholeProgramError("generated device symbol collides with source symbol")
        source = _source_function(region, kernel_symbol)
        subdir = work / symbol
        subdir.mkdir(exist_ok=True)
        (subdir / "source.mlir").write_text(source)
        request = DeviceRequest(
            source,
            kernel_symbol,
            hashlib.sha256(source.encode()).hexdigest(),
            tuple(str(v.type) for v in region.inputs),
            tuple(str(v.type) for v in region.outputs),
            tuple(ordinal[o] for o in region.operations),
            subdir,
        )
        code_identity = None
        if reuse_device_kernels:
            code_identity = typed_device_function_identity(source)
        reused = code_identity is not None and code_identity in shared_artifacts
        artifact = shared_artifacts[code_identity] if reused else compile_device(request)
        if (
            not isinstance(artifact, DeviceArtifact)
            or (not reused and artifact.symbol != kernel_symbol)
            or not Path(artifact.path).is_file()
            or Path(artifact.path).stat().st_size == 0
        ):
            raise WholeProgramError(f"{symbol}: callback did not supply the requested compiled device artifact")
        if artifact.command_buffer is not None and not artifact.command_buffer.get("commands"):
            raise WholeProgramError(f"{symbol}: device callback returned an empty or declined command buffer")
        if not reused:
            artifacts.append(artifact)
            if code_identity is not None:
                shared_artifacts[code_identity] = artifact
        shim_kernels[symbol] = artifact.symbol
        buffers = []
        call = []
        for i, value in enumerate((*region.inputs, *region.outputs)):
            kind = "input" if i < len(region.inputs) else "output"
            role = f"arg{i}"
            shape, dtype = tuple(value.type.get_shape()), str(value.type.element_type)
            buffers.append(LogicalBuffer(name=role, kind=kind, role=role, shape=shape, dtype=dtype))
            call.append(dict(role=role, shape=list(shape), dtype=dtype))
        shims[symbol] = (buffers, call)
        plans.append((region, root, symbol))
        owned.update(members)
        rows.append(
            dict(
                symbol=symbol,
                kernel_symbol=artifact.symbol,
                kernel_reused=reused,
                code_identity=code_identity,
                source_ordinals=list(request.source_ordinals),
                source_sha256=request.source_sha256,
                placement="device",
                reason=region.reason,
                input_types=list(request.input_types),
                output_types=list(request.output_types),
            )
        )

    # Callbacks all succeed before altering source: no partially routed host fallback.
    routed_values = {}
    for region, root, symbol in plans:
        block = root.parent
        destinations = [tensor.EmptyOp([], value.type) for value in region.outputs]
        operands = (
            *(routed_values.get(value, value) for value in region.inputs),
            *(destination.tensor for destination in destinations),
        )
        call = func.CallOp(symbol, operands, [value.type for value in region.outputs])
        for destination in destinations:
            block.insert_op_before(destination, root)
        block.insert_op_before(call, root)
        for original, replacement in zip(region.outputs, call.results, strict=True):
            routed_values[original] = replacement
            original.replace_all_uses_with(replacement)
        # Erase only dead source members; shared initialization remains available.
        for op in reversed(region.operations):
            if op.name not in ("arith.constant", "tensor.empty", "tensor.splat", "linalg.fill") and all(
                not result.uses for result in op.results
            ):
                op.detach()
                op.erase()
        accesses = ["read"] * len(region.inputs) + ["write"] * len(region.outputs)
        attrs = ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in accesses])
        module.body.block.add_op(
            func.FuncOp(
                symbol,
                ([v.type for v in operands], [value.type for value in region.outputs]),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
        )
    routed = text(module)
    for region, _, symbol in plans:
        routed = patch_declaration_arg_attrs(
            routed, [symbol], argument_access=["read"] * len(region.inputs) + ["write"] * len(region.outputs)
        )
    if unpatched_declarations(routed, shims):
        raise WholeProgramError("device declarations lost their required bufferization access")
    numerical_preparation = None
    if host_optimization in ("bounded_rne", "bounded_rne_fused"):
        # Use the same checked source representation as Merlin's original host
        # numerics route. Extension lowering preserves the original typed
        # quantization; its independent expression checker admits only closed
        # round/clamp/cast chains. Device calls remain opaque boundaries.
        from merlin.compiler.lower.host.automatic_host import optimize_host_quantization
        from merlin.compiler.quantization.linalg import lower_quant_ext

        prepared = parse_mlir_text(routed)
        extension_count = lower_quant_ext(prepared)
        prepared, numerical_preparation = optimize_host_quantization(
            prepared, numerical_policy="typed_source:" + hashlib.sha256(source_text.encode()).hexdigest()
        )
        routed = text(prepared, generic=True)
        numerical_preparation["quantization_extensions_lowered"] = extension_count
        numerical_preparation["selected_source_sha256"] = hashlib.sha256(routed.encode()).hexdigest()
        (work / "host-numerics.json").write_text(json.dumps(numerical_preparation, indent=2))
    source_path = work / "routed.mlir"
    source_path.write_text(routed)
    shim_path = None
    if shims:
        shim = emit_device_shim(shims, kernel_symbol_for=shim_kernels.__getitem__)
        if shim.skipped or set(shim.symbols) != set(shims):
            raise WholeProgramError(f"device ABI adaptation refused: {shim.skipped}")
        shim_path = work / "device_shim.c"
        shim_path.write_text(shim.text)
    options = dict(lower_options or {})
    # These shared passes preserve exact source arithmetic and independently
    # check every reuse. No input argument is presumed immutable.
    from merlin.compiler.lower.options.tensor_provenance_cse import FEATURE as preparation_cse

    options["features"] = sorted(set(options.get("features") or ()) | {preparation_cse})
    options.setdefault("static_arena", True)
    if "targets" in options or "retain_llvm_dialect" in options:
        raise WholeProgramError("targets and retention are owned by the whole-program adapter")
    if host_driver == "stock-cli":
        # Configure only the child. Other compiler requests in this process keep
        # their own explicit transport and pass selections.
        import os
        import subprocess
        import sys
        from types import SimpleNamespace

        host_work = (work / "host").resolve()
        host_work.mkdir(exist_ok=True)
        response = host_work / "adapter-result.json"
        request_path = host_work / "adapter-request.json"
        request_path.write_text(
            json.dumps(
                dict(source=str(source_path.resolve()), work=str(host_work), result=str(response), options=options)
            )
        )
        env = dict(os.environ)
        env.update(
            MERLIN_COMPILER_DRIVER="stock-cli",
            MERLIN_FUSION_GUARD="0",
            MERLIN_SPLIT_FRESH_DEALLOCS="0",
            MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT="0",
        )
        if host_optimization == "bounded_rne_fused":
            # Existing scalar-pipeline alternative: expose named pointwise
            # operations to the ordinary tensor fusion before bufferization.
            # This switch is consumed by pipeline._splice in the child, unlike
            # MERLIN_FUSE_POST, which affects only the vector pipeline.
            env["MERLIN_GENERALIZE_BEFORE_FUSE"] = "1"
        install = Path(mlir_install)
        env.setdefault("MERLIN_MLIR_INSTALL", str(install))
        run = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("_host_regions_worker.py")), str(request_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if run.returncode:
            raise WholeProgramError("shared stock host lowering failed: " + run.stderr[-6000:])
        response_data = json.loads(response.read_text())
        lowered = SimpleNamespace(ll_path=Path(response_data["ll_path"]), stats=response_data["stats"])
        if host_optimization == "bounded_rne_fused":
            recipe_path = host_work / "lowering_recipe.json"
            recipe = json.loads(recipe_path.read_text())
            pipelines = [
                token
                for command in recipe["commands"]
                for token in command["argv"]
                if token.startswith("--pass-pipeline=")
            ]
            if len(pipelines) != 1 or not (
                pipelines[0].find("linalg-generalize-named-ops") >= 0
                and pipelines[0].find("linalg-generalize-named-ops") < pipelines[0].find("linalg-fuse-elementwise-ops")
            ):
                raise WholeProgramError("selected stock scalar fusion order did not reach the actual lowering command")
            lowered.stats["adapter_stock_host_pipeline"] = dict(
                selection="MERLIN_GENERALIZE_BEFORE_FUSE=1",
                pipeline=pipelines[0],
                contract=recipe["sources"]["pipeline_driver"],
                recipe=dict(path=str(recipe_path), sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest()),
                source=recipe["sources"]["prepared_mlir"],
                returned_llvm_ir=recipe["returned_llvm_ir"],
                cost_evidence="not_measured",
            )
    elif host_driver == "python":
        if host_optimization == "bounded_rne_fused":
            raise WholeProgramError("bounded_rne_fused requires the explicitly selected stock scalar transport")
        options["features"] = frozenset(options["features"])
        lowered = lower_model(routed, work / "host", targets=(), retain_llvm_dialect=retain_llvm_dialect, **options)
    else:
        raise WholeProgramError("host_driver must be stock-cli or python")
    lowered.stats["adapter_device_code_sharing"] = dict(
        enabled=reuse_device_kernels,
        source_calls=len(selected),
        compiled_kernels=len(artifacts),
        reused_calls=len(selected) - len(artifacts),
    )
    lowered.stats["adapter_device_calls"] = rows
    lowered.stats["adapter_abi"] = source_abi
    lowered.stats["adapter_host_source_optimization"] = host_optimization
    if numerical_preparation is not None:
        lowered.stats["adapter_host_numerical_preparation"] = numerical_preparation
    lowered.stats["adapter_original_source"] = str(original_source_path.resolve())
    product = lowered.stats.get("llvm_dialect_product")
    dialect_path = None
    if host_driver == "stock-cli" and retain_llvm_dialect:
        dialect_path = (work / "host" / "model.llvm.mlir").resolve()
    if product:
        receipt = json.loads(Path(product["path"]).read_text())
        dialect_path = Path(receipt["llvm_dialect"]["path"])
    report_path = work / "placement.json"
    report_path.write_text(
        json.dumps(
            dict(
                schema="submission.whole_program.v1",
                device_regions=rows,
                source_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                static_boundary_bytes=sum(
                    int(__import__("math").prod(v.type.get_shape())) * max(1, int(str(v.type.element_type)[1:]) // 8)
                    for r in selected
                    for v in (*r.inputs, *r.outputs)
                ),
                **report_metadata,
                measured_host_cycles=None,
            ),
            indent=2,
        )
    )
    return WholeProgramResult(
        lowered.ll_path, dialect_path, shim_path, tuple(artifacts), source_path, report_path, lowered.stats
    )
