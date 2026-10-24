"""Preserve result descriptors and avoid unsafe upstream allocation hoisting."""

from __future__ import annotations

import inspect


def _safe_result_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    # Inspect after CSE and bufferization: independent tensor results may now
    # share an allocation. Upstream's hoister erases that allocation once for
    # each returned operand, dereferencing it after its first erasure.
    passes = pipeline.split(",")
    cut = next((i for i, p in enumerate(passes) if p.startswith("buffer-results-to-out-params")), -1)
    if cut < 0:
        return _result_original_run_stages(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    _result_original_run_stages(ctx, module, ",".join(passes[:cut]), 0, (), (), (), pre_generalize)

    def aliased_allocation(op):
        if op.operation.name == "func.return":
            seen = []
            for value in op.operands:
                if value in seen and getattr(value.owner, "name", None) == "memref.alloc":
                    return True
                seen.append(value)
        return any(
            aliased_allocation(child) for region in op.regions for block in region.blocks for child in block.operations
        )

    if aliased_allocation(module.operation):
        passes[cut] = passes[cut].replace(" hoist-static-allocs", "")
    _result_original_run_stages(ctx, module, ",".join(passes[cut:]), erase, mid, late, post_openmp, ())


def inject(source: str) -> str:
    """Bind the guard at the final runner call, after feature wrappers exist."""
    marker = "src_path, out_path, pipeline ="
    if source.count(marker) != 1:
        raise ValueError("lowering runner must declare one final stage invocation")
    binding = inspect.getsource(_safe_result_stages)
    binding += "\n_result_original_run_stages = _run_stages\n_run_stages = _safe_result_stages\n"
    return source.replace(marker, binding + marker)


def descriptor_results(pipeline: str, mlir: str, *, preserve: bool = False) -> str:
    """Return actual memrefs for dynamic extents or explicit boundary observation."""
    if not preserve and ("?" not in mlir or ("@forward(" not in mlir and 'sym_name = "forward"' not in mlir)):
        return pipeline  # No dynamic extent, or no @forward entry, so no forward result to return by descriptor.
    from merlin.common.mlir_query import forward_signature

    try:
        _, results = forward_signature(mlir)
    except ValueError as exc:
        if str(exc).startswith("no func.func @forward"):
            return pipeline
        raise
    if preserve or any(dim < 0 for shape, _ in results for dim in shape):
        stages = [p for p in pipeline.split(",") if not p.startswith("buffer-results-to-out-params")]
        finalize = next((i for i, p in enumerate(stages) if p.startswith("finalize-memref-to-llvm")), None)
        if finalize is None:
            raise ValueError("descriptor result lowering requires finalize-memref-to-llvm in the pipeline")
        from merlin.llvmlower.result_layout import read_layout

        if read_layout(mlir) is not None:
            # Layout reconstruction needs ownership-normalized memrefs at its seam.
            bufferized = [
                i
                for i, p in enumerate(stages)
                if p.startswith("one-shot-bufferize") or p == "bufferization-lower-deallocations"
            ]
            at = max(bufferized, default=-1) + 1
        else:
            # Preserve the ordinary dynamic-result pipeline's existing pass order.
            at = finalize
        stages.insert(at, "convert-bufferization-to-memref")
        return ",".join(stages)
    return pipeline
