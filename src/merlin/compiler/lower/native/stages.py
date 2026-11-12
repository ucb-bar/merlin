"""Existing native stage preludes shared by every generated runner.

Pass algorithms remain in their declared legacy owners during staged migration."""

from __future__ import annotations

from merlin.compiler.lower.model import fusion_guard
from merlin.compiler.lower.model.alloca_scope_lower import RUNNER_PRELUDE as _ALLOCA_SCOPE_LOWER_PRELUDE
from merlin.compiler.lower.model.broadcast_fold import RUNNER_PRELUDE as _BROADCAST_FOLD_PRELUDE
from merlin.compiler.lower.model.concat_dps import RUNNER_PRELUDE as _CONCAT_DPS_PRELUDE
from merlin.compiler.lower.model.contiguous_suffix_copy import MID_STAGE_SRC as _CONTIGUOUS_COPY_MID_SRC
from merlin.compiler.lower.model.contiguous_suffix_copy import RUNNER_PRELUDE as _CONTIGUOUS_COPY_PRELUDE
from merlin.compiler.lower.model.copy_expand import MID_STAGE_SRC as _MID_STAGE_SRC
from merlin.compiler.lower.model.copy_expand import RUNNER_PRELUDE as _COPY_EXPAND_PRELUDE
from merlin.compiler.lower.model.dealloc_lifetime import DEALLOC_CHECK_PRELUDE
from merlin.compiler.lower.model.int_softmax_table import RUNNER_PRELUDE as _INT_SOFTMAX_TABLE_PRELUDE
from merlin.compiler.lower.model.sliced_elementwise_output import PRELUDE as SLICED_OUTPUT_PRELUDE
from merlin.compiler.lower.model.sliced_elementwise_output import RUNNER as SLICED_OUTPUT_RUNNER
from merlin.compiler.lower.model.uniform_fill_copy import MID_STAGE_SRC as _UNIFORM_FILL_COPY_MID_SRC
from merlin.compiler.lower.model.uniform_fill_copy import RUNNER_PRELUDE as _UNIFORM_FILL_COPY_PRELUDE
from merlin.compiler.passes.fragments import DEALLOC_CHECK_RUNNER

# The normal runner already reserves argv[17] for the target data layout.
# Compose the optional copy stages with their new trailing slots rather than
# allowing their original standalone gate positions to reinterpret that layout.
_UNIFORM_FILL_COPY_MID_SRC = _UNIFORM_FILL_COPY_MID_SRC.replace(
    'if len(sys.argv) > 17 and sys.argv[17] == "1":',
    'if len(sys.argv) > 18 and sys.argv[18] == "1":',
)
_CONTIGUOUS_COPY_MID_SRC = _CONTIGUOUS_COPY_MID_SRC.replace(
    'if len(sys.argv) > 18 and sys.argv[18] == "1":',
    'if len(sys.argv) > 19 and sys.argv[19] == "1":',
)
from merlin.compiler.lower.model.adjacent_exp import RUNNER_PRELUDE as _ADJACENT_EXP_PRELUDE
from merlin.compiler.lower.model.automatic_host_features import RUNNER_PRELUDE as _AUTOMATIC_HOST_FEATURES_PRELUDE
from merlin.compiler.lower.model.broadcast_math_hoist import RUNNER_PRELUDE as _BROADCAST_MATH_HOIST_PRELUDE
from merlin.compiler.lower.model.fma_intrinsic import RUNNER_PRELUDE as _FMA_INTRINSIC_PRELUDE
from merlin.compiler.lower.model.guarded_zero_contraction import RUNNER_PRELUDE as _GUARDED_ZERO_CONTRACTION_PRELUDE
from merlin.compiler.lower.model.masked_contraction import RUNNER_PRELUDE as _MASKED_CONTRACTION_PRELUDE
from merlin.compiler.lower.model.masked_contraction import STAGE_RUNNER as _MASKED_CONTRACTION_STAGE_RUNNER
from merlin.compiler.lower.model.named_broadcast_fold import RUNNER_PRELUDE as _NAMED_BROADCAST_FOLD_PRELUDE
from merlin.compiler.lower.model.panel_parallel import MID_STAGE_SRC as _PANEL_PARALLEL_MID_SRC
from merlin.compiler.lower.model.panel_parallel import RUNNER_PRELUDE as _PANEL_PARALLEL_PRELUDE
from merlin.compiler.lower.model.parallel_coarsen import RUNNER_PRELUDE as _PARALLEL_COARSEN_PRELUDE
from merlin.compiler.lower.model.parallel_coarsen import STAGE_SRC as _PARALLEL_COARSEN_STAGE_SRC
from merlin.compiler.lower.model.parallel_grain import LATE_STAGE_SRC as _PARALLEL_GRAIN_LATE_SRC
from merlin.compiler.lower.model.parallel_grain import RUNNER_PRELUDE as _PARALLEL_GRAIN_PRELUDE
from merlin.compiler.lower.model.parallel_team import RUNNER_PRELUDE as _PARALLEL_TEAM_PRELUDE
from merlin.compiler.lower.model.parallel_team import STAGE_SRC as _PARALLEL_TEAM_STAGE_SRC
from merlin.compiler.lower.model.rhs_column_reuse import RUNNER_PRELUDE as _RHS_COLUMN_REUSE_PRELUDE
from merlin.compiler.lower.model.roundeven_intrinsic import RUNNER_PRELUDE as _ROUND_INTRINSIC_PRELUDE
from merlin.compiler.lower.model.scalar_contraction import RUNNER_PRELUDE as _SCALAR_CONTRACTION_PRELUDE
from merlin.compiler.lower.model.scalar_ordered_reduction import RUNNER_PRELUDE as _SCALAR_ORDERED_REDUCTION_PRELUDE
from merlin.compiler.lower.model.scalar_pointwise_packet import RUNNER_PRELUDE as _SCALAR_POINTWISE_PACKET_PRELUDE
from merlin.compiler.lower.model.scalar_pointwise_unroll import RUNNER_PRELUDE as _SCALAR_POINTWISE_UNROLL_PRELUDE
from merlin.compiler.lower.model.scalar_squared_sum import RUNNER_PRELUDE as _SCALAR_SQUARED_SUM_PRELUDE
from merlin.compiler.lower.model.selfcopy import RUNNER_PRELUDE as _SELFCOPY_PRELUDE
from merlin.compiler.lower.model.transpose_fuse import RUNNER_PRELUDE as _TRANSPOSE_FUSE_PRELUDE
from merlin.compiler.lower.model.transpose_maps import RUNNER_PRELUDE as _TRANSPOSE_MAPS_PRELUDE

#: Runner glue: wrap ``_run_stages`` so the placement check runs on the IR the sinking stage
#: produced, in EVERY runner variant, without any of them having to know about it. The wrapper
#: delegates unchanged when the pass list carries no sinking stage.

#: The line the runner prints once the check has run. `lower_to_llvm_ir` REQUIRES it whenever the
#: pipeline carries the sinking stage: a runner variant that drove the PassManager itself and never
#: reached the wrapper would otherwise sink the frees with nothing checking them, and report success.
DEALLOC_CHECK_TOKEN = "OK dealloc_placement"


# The ownership pass can emit one aggregate dealloc for an entire model. Its operands are
# base views produced by extract_strided_metadata, even when each base came from a distinct
# local alloc and has a constant-true ownership condition. The upstream liveness pass cannot
# move a free while it is hidden in that aggregate. This split only handles that narrow case;
# everything else remains in the original aggregate and retains upstream's ownership logic.
SPLIT_FRESH_DEALLOCS_PRELUDE = r"""
def _split_fresh_local_deallocs(ctx, module):
    from collections import defaultdict

    def is_alloc(value):
        owner = value.owner
        return hasattr(owner, 'operation') and owner.operation.name == 'memref.alloc'

    def is_device_output_alias(op, declarations):
        name = str(op.attributes['callee']).lstrip('@')
        decl = declarations.get(name)
        return (
            name.startswith('merlin_dev_') and decl is not None
            and len(op.operands) >= 3 and len(op.results) == 1
            and str(op.results[0].type) == str(op.operands[-1].type)
            and len(decl.arg_attrs) == len(op.operands)
            and all('bufferization.access' in attrs for attrs in decl.arg_attrs)
            and [str(attrs['bufferization.access']) for attrs in decl.arg_attrs]
            == ['"read"'] * (len(op.operands) - 1) + ['"write"']
        )

    def nested_operations(parent):
        for region in parent.regions:
            for block in region.blocks:
                for op in block.operations:
                    yield op
                    yield from nested_operations(op)

    declarations = {
        str(op.attributes['sym_name']).strip('"'): op
        for op in module.body.operations if op.operation.name == 'func.func'
    }
    # Reuse the original selected effects and exact current positional ABI
    # checker. A void call is not an escape barrier only when this actual call
    # operation passed its native forwarding/fresh-result checks. Names and
    # declaration access attributes alone supply no such premise.
    borrowed_checker = globals().get('_obl_borrowed_calls')
    borrowed_uses, _, borrowed_violations = (
        (set(), set(), []) if borrowed_checker is None else borrowed_checker(module))
    if borrowed_violations:
        raise ValueError('fresh dealloc borrowed ABI: ' + '; '.join(borrowed_violations))
    moved = dropped_aliases = 0
    for func in module.body.operations:
        if func.operation.name != 'func.func' or not func.regions[0].blocks:
            continue
        # An unrecognized callee may retain a memref even when it returns void.
        # Search nested regions too: a call in a branch can outlive a top-level
        # aggregate dealloc just as a direct call can.
        unknown_memref_call = any(
            op.operation.name == 'func.call'
            and any(str(value.type).startswith('memref') for value in (*op.operands, *op.results))
            and not is_device_output_alias(op, declarations)
            and op.operation not in borrowed_uses
            for op in nested_operations(func)
        )
        for block in func.regions[0].blocks:
            # Merlin's generated device shim returns its last, writable output
            # descriptor. Earlier operands are read-only, including extra group
            # inputs such as a bias; the shim retains no pointer. Check the exact
            # declaration contract before treating the result as an alias.
            call_alias = {}
            for op in block.operations:
                if op.operation.name != 'func.call' or not any(
                    str(result.type).startswith('memref') for result in op.results
                ):
                    continue
                if is_device_output_alias(op, declarations):
                    call_alias[op.results[0]] = op.operands[-1]
            for old in list(block.operations):
                if old.operation.name != 'bufferization.dealloc' or old.results or len(old.operands) % 2:
                    continue
                n = len(old.operands) // 2
                mems, conds = list(old.operands[:n]), list(old.operands[n:])
                groups = defaultdict(list)
                # An unknown result alias could duplicate a fresh root. Keep the
                # whole aggregate under upstream's ownership semantics in that case.
                if unknown_memref_call:
                    continue
                ambiguous = False
                for i, mem in enumerate(mems):
                    owner = mem.owner
                    if not hasattr(owner, 'operation') or owner.operation.name != 'memref.extract_strided_metadata':
                        ambiguous = True
                        break
                    base = owner.operands[0]
                    if is_alloc(base):
                        groups[base].append((i, 'direct'))
                    elif base in call_alias and is_alloc(call_alias[base]):
                        groups[call_alias[base]].append((i, 'call_alias'))
                    elif not isinstance(base, ir.BlockArgument):
                        ambiguous = True
                        break
                if ambiguous:
                    continue
                chosen = {}
                for root, entries in groups.items():
                    direct = [i for i, kind in entries if kind == 'direct']
                    if len(direct) != 1:
                        continue
                    cond_op = conds[direct[0]].owner
                    if (not hasattr(cond_op, 'operation') or cond_op.operation.name != 'arith.constant'
                            or str(cond_op.attributes['value']) != 'true'):
                        continue
                    chosen[root] = {i for i, _ in entries}
                if not chosen:
                    continue
                removed = set().union(*chosen.values())
                keep = [(mem, cond) for i, (mem, cond) in enumerate(zip(mems, conds)) if i not in removed]
                with ctx, old.location, ir.InsertionPoint(old):
                    for root in chosen:
                        ir.Operation.create('memref.dealloc', operands=[root])
                    if keep:
                        ir.Operation.create(
                            'bufferization.dealloc',
                            operands=[*(mem for mem, _ in keep), *(cond for _, cond in keep)],
                            attributes={'operandSegmentSizes': ir.DenseI32ArrayAttr.get(
                                [len(keep), len(keep), 0])})
                old.operation.erase()
                moved += len(chosen)
                dropped_aliases += len(removed) - len(chosen)
    module.operation.verify()
    print('OK split_fresh_local_deallocs', moved, 'fresh allocs', dropped_aliases, 'device aliases')
"""


SPLIT_FRESH_DEALLOCS_RUNNER = r"""
_SPLIT_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if '__merlin_split_fresh_deallocs__' not in passes:
        return _SPLIT_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                      post_openmp, pre_generalize)
    if passes.count('__merlin_split_fresh_deallocs__') != 1:
        raise RuntimeError('aggregate dealloc split marker must appear exactly once')
    at = passes.index('__merlin_split_fresh_deallocs__')
    head, tail = passes[:at], passes[at + 1:]
    if 'ownership-based-buffer-deallocation' not in head:
        raise RuntimeError('aggregate dealloc split must follow ownership analysis')
    if any('convert-scf-to-openmp' in p for p in head):
        raise RuntimeError('aggregate dealloc split must precede parallel control-flow lowering')
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    _SPLIT_ORIG_RUN_STAGES(ctx, module, ','.join(head),
                           erase if hoist_head else 0, mid if hoist_head else (), (), (),
                           pre_generalize)
    _split_fresh_local_deallocs(ctx, module)
    _SPLIT_ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                           0 if hoist_head else erase, () if hoist_head else mid, late,
                           post_openmp, ())
"""
