"""One ranked invocation over checked batch views and an existing dense primitive.

The original read operands are borrowed across missing batch axes. Packing is
only the existing bounded per-matrix fallback. Whole-batch alias checks run
before the first write, so a later input cannot be overwritten by an earlier
output. A successful bridge does not qualify the external primitive.
"""

from __future__ import annotations

from merlin.compile.scheduling.batched_dense import BatchedDenseProblem, BatchMatrixView

from .device_shim import ShimUnit, _memref_typedef
from .logical_borrowed_abi import (
    _symbol,
    dense_buffers,
    emit_borrowed_dense_translation_unit,
)

_SPAN = r"""
static int batch_span(void *base,intptr_t offset,const intptr_t *sizes,
 const intptr_t *strides,int rank,uintptr_t width,uintptr_t *address,uintptr_t *length){
 if(!base || offset<0)return 0;
 uintptr_t limit=(uintptr_t)INTPTR_MAX/width,last=(uintptr_t)offset;
 if(last>=limit)return 0;
 for(int i=0;i<rank;i++){
  uintptr_t term;
  if(sizes[i]<=0 || strides[i]<=0 ||
     __builtin_mul_overflow((uintptr_t)sizes[i]-1,(uintptr_t)strides[i],&term) ||
     term>=limit-last)return 0;
  last+=term;
 }
 uintptr_t delta=(uintptr_t)offset*width,bytes=(last-(uintptr_t)offset+1)*width;
 if(delta>UINTPTR_MAX-(uintptr_t)base)return 0;
 *address=(uintptr_t)base+delta;
 if(bytes>UINTPTR_MAX-*address)return 0;
 *length=bytes;return 1;
}
/* Sufficient non-overlap proof for padded/permuted positive-stride outputs. */
static int batch_injective(const intptr_t *sizes,const intptr_t *strides,int rank){
 int order[8];for(int i=0;i<rank;i++)order[i]=i;
 for(int i=1;i<rank;i++)for(int j=i;j>0 && strides[order[j]]<strides[order[j-1]];j--){
  int a=order[j];order[j]=order[j-1];order[j-1]=a;
 }
 uintptr_t covered=1;
 for(int i=0;i<rank;i++){
  int a=order[i];if(sizes[a]<=1)continue;
  uintptr_t term;
  if((uintptr_t)strides[a]<covered ||
     __builtin_mul_overflow((uintptr_t)sizes[a]-1,(uintptr_t)strides[a],&term) ||
     __builtin_add_overflow(covered,term,&covered))return 0;
 }
 return 1;
}
"""


def emit_borrowed_batch_translation_unit(
    capability, problem, *, symbol, kernel, borrow_contiguous=True, observer_symbol=None,
):
    """Emit nested runtime loops, not a static list of repeated tensor calls.

    Each inner iteration reuses the existing checked dense bridge. Input views
    may omit logical batch axes or swap the two matrix axes. Outputs preserve
    the complete logical batch shape and require globally injective storage.
    The selected synchronous primitive/effects remain caller obligations.
    """
    if type(problem) is not BatchedDenseProblem:
        raise ValueError("explicit checked batch geometry required")
    problem.__post_init__()
    symbol, kernel = _symbol(symbol), _symbol(kernel)
    helper = symbol + "_matrix"
    if kernel == helper:
        raise ValueError("batch bridge and primitive symbols must be distinct")
    unit = emit_borrowed_dense_translation_unit(
        capability, dense_buffers(problem, ("i8", "i8", "i32")),
        symbol=helper, kernel=kernel, borrow_contiguous=borrow_contiguous,
        observer_symbol=observer_symbol,
    )
    output = BatchMatrixView(problem.output_shape, tuple(range(len(problem.batch_shape))))
    views = (problem.lhs, problem.rhs, output)
    ranks = tuple(len(v.shape) for v in views)
    lines = [unit.text, *[_memref_typedef(r) for r in sorted(set(ranks) - {2})], _SPAN]
    params = [f"{'const ' if i<2 else ''}merlin_lmemref_{rank}d *d{i}"
              for i, rank in enumerate(ranks)]
    lines += [f"void {symbol}({','.join(params)}){{", " uintptr_t address[3],length[3];"]
    for i, (view, rank, width) in enumerate(zip(views, ranks, (1, 1, 4), strict=True)):
        checks = " || ".join(f"d{i}->sizes[{a}]!={n}" for a, n in enumerate(view.shape))
        lines.append(f" if(!d{i} || {checks} || !batch_span(d{i}->aligned,d{i}->offset,"
                     f"d{i}->sizes,d{i}->strides,{rank},{width},&address[{i}],&length[{i}]))__builtin_trap();")
    lines += [f" if(!batch_injective(d2->sizes,d2->strides,{ranks[2]}) || "
              "!disjoint(address[0],length[0],address[2],length[2]) || "
              "!disjoint(address[1],length[1],address[2],length[2]))__builtin_trap();"]
    for axis, extent in enumerate(problem.batch_shape):
        lines.append(f" for(intptr_t b{axis}=0;b{axis}<{extent};b{axis}++){{")
    for i, (view, rank) in enumerate(zip(views, ranks, strict=True)):
        offset = "+".join([f"d{i}->offset", *[
            f"b{axis}*d{i}->strides[{a}]" for a, axis in enumerate(view.batch_axes)
        ]])
        row, col = (rank-1, rank-2) if view.transposed else (rank-2, rank-1)
        m, n = view.matrix_shape
        lines.append(f" merlin_lmemref_2d s{i}={{d{i}->allocated,d{i}->aligned,{offset},"
                     f"{{{m},{n}}},{{d{i}->strides[{row}],d{i}->strides[{col}]}}}};")
    lines += [f" {helper}(&s0,&s1,&s2);", *[" }" for _ in problem.batch_shape], "}"]
    return ShimUnit(text="\n".join(lines)+"\n", symbols=(symbol,), kernel=kernel, kernels=(kernel,))
