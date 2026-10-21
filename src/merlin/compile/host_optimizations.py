"""Reusable scalar host transformations without an accelerator implementation.

Native tensor rewrites use the same source as ordinary Merlin lowering. Callers
supply their existing MLIR context and module; no compiler, runtime, device or
reference implementation is discovered. LLVM utilities require their original
explicit CPU and numerical/effect choices.
"""

from merlin.llvmlower.host_tensor_rewrites import (
    BROADCAST_SOURCE,
    CONTRACTION_SOURCE,
    SQUARED_SUM_SOURCE,
)

_COMPOSE = r"""
def _merlin_host_ordinary_scope(operation):
    ancestor = operation
    while ancestor is not None:
        for name in ancestor.attributes:
            if ("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
                    or "fenv" in name):
                return False
        ancestor = ancestor.parent
    return True


def _merlin_optimize_host_tensors(ctx, module, choices):
    counts = dict.fromkeys(choices, 0)
    # Explicit environment scopes anywhere in this module are unsupported.
    # The existing individual rewrites also check all typed body operations.
    def scoped(operation):
        if not _merlin_host_ordinary_scope(operation):
            return False
        return all(scoped(inner.operation) for region in operation.regions
                   for block in region.blocks for inner in block.operations)
    if not scoped(module.operation):
        return counts
    if choices['broadcast_math']:
        counts['broadcast_math'] = _hoist_broadcast_source_math(ctx, module)
    if choices['contraction']:
        selected = []
        def collect(operation):
            for region in operation.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        spec = _scalar_contraction_spec(inner)
                        if spec is not None and spec[1][-1] > 1:
                            selected.append(inner.operation)
                        else:
                            collect(inner.operation)
        collect(module.operation)
        counts['contraction'] = _scalarize_tensor_contractions(ctx, module, selection=selected)
    if choices['squared_sum']:
        counts['squared_sum'] = _scalarize_squared_sums(ctx, module)
    module.operation.verify()
    return counts
"""


def tensor_rewrite_source() -> str:
    """Exact native matcher/rewriter source for an existing MLIR interpreter.

    This excludes runner subprocesses, pass-manager orchestration and toolchain
    lookup. The returned function `_merlin_optimize_host_tensors` accepts the
    selected context, module and three explicit Boolean choices.
    """
    return CONTRACTION_SOURCE + SQUARED_SUM_SOURCE + BROADCAST_SOURCE + _COMPOSE


def rewrite_tensor_module(context, module, *, contractions=True, squared_sums=True, broadcast_math=True):
    """Rewrite current pure static source tensors; preserve ordered arithmetic.

    Unsupported source stays on its original route. This API supplies no
    approximation, physical alias, device support or measured-cost permission.
    Named operations must already have their ordinary generic representation.
    """
    choices = dict(contraction=contractions, squared_sum=squared_sums, broadcast_math=broadcast_math)
    if any(type(value) is not bool for value in choices.values()):
        raise ValueError("explicit Boolean host rewrite choices required")
    namespace = {}
    exec(compile(tensor_rewrite_source(), "<merlin-host-tensor-rewrites>", "exec"), namespace)
    return namespace["_merlin_optimize_host_tensors"](context, module, choices)


def legalize_bounded_rounding(source, *, host_isa=None, combine_clamp=False):
    """Legalize closed LLVM integer observations under an explicit CPU policy."""
    from merlin.llvmlower.late_quant_rne import rewrite

    return rewrite(source, host_isa=host_isa, combine_clamp=combine_clamp)


def legalize_constant_clamps(source):
    """Retain IEEE values while simplifying complete finite-bound clamp calls."""
    from merlin.llvmlower.constant_float_clamp import rewrite

    return rewrite(source)


def rewrite_fma_packets(source, *, capability, width, ordinary_nontrapping, exception_flags_unobserved):
    """Schedule source FMA calls using an explicit host CPU emitter and effects."""
    from merlin.llvmlower.constant_fma_packet import rewrite
    from merlin.runtime.host_fma_packet import RV64GCScalarFmaCapability

    if type(capability) is not RV64GCScalarFmaCapability:
        raise ValueError("explicit supported host CPU packet capability required")
    if any(type(value) is not bool for value in (ordinary_nontrapping, exception_flags_unobserved)):
        raise ValueError("explicit Boolean source effects required")
    return rewrite(
        source,
        emitter=capability.emit,
        width=width,
        ordinary_nontrapping=ordinary_nontrapping,
        exception_flags_unobserved=exception_flags_unobserved,
    )
