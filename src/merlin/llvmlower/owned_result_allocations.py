"""Checked allocation products for execution-sized returned tensor storage.

The generic allocator sees a byte request only after MemRefToLLVM arithmetic.
Guard the actual memref extents before that arithmetic can wrap. This is an
explicit resource profile, never a source-shape or ownership proof.
"""

from dataclasses import dataclass
from pathlib import Path

MARKER = "__merlin_owned_result_allocations__"


@dataclass(frozen=True)
class OwnedAllocationPolicy:
    max_bytes: int

    def verify(self):
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= (1 << 63) - 1:
            raise ValueError("owned allocation policy requires a positive signed-i64 byte budget")


def source_dependency(policy, *, vectorize, transform_schedule, parallel, parallel_harts):
    if type(policy) is not OwnedAllocationPolicy:
        raise ValueError("owned results require a typed allocation resource policy")
    policy.verify()
    if vectorize or transform_schedule is not None or parallel or parallel_harts is not None:
        raise ValueError("owned result allocation checks currently require the ordinary serial host transport")
    return Path(__file__)


def bind_runner(source, policy):
    if type(policy) is not OwnedAllocationPolicy:
        raise ValueError("owned allocations require the original typed resource policy")
    policy.verify()
    anchor = "src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]"
    if source.count(anchor) != 1:
        raise ValueError("owned allocation checks require the ordinary native source seam")
    return source.replace(anchor, f"_OWNED_ALLOCATION_LIMIT = {policy.max_bytes}\n" + RUNNER + "\n" + anchor)


RUNNER = r"""
def _owned_allocation_guards(ctx, module):
    from torch_mlir import ir as _oa_ir
    todo = []
    widths = {'i1':1,'i8':1,'i16':2,'i32':4,'i64':8,'index':8,
              'f16':2,'bf16':2,'f32':4,'f64':8,
              'f8E4M3FN':1,'f8E4M3FNUZ':1,'f8E5M2':1,'f8E5M2FNUZ':1}
    def visit(op):
        if op.name == 'memref.alloca':
            raise ValueError('owned result allocation profile needs a separate stack resource contract')
        if op.name == 'memref.alloc':
            typ = _oa_ir.MemRefType(op.results[0].type)
            element = str(typ.element_type)
            if (element not in widths or typ.rank > 16
                    or typ.layout != _oa_ir.AffineMapAttr.get(_oa_ir.AffineMap.get_identity(typ.rank))
                    or str(typ.memory_space) not in ('None', '0 : i64', '0 : i32')):
                raise ValueError('owned result allocation type/layout/space lacks a checked byte product')
            todo.append((op, tuple(typ.shape), widths[element]))
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner.operation)
    with ctx:
        visit(module.operation)
    with ctx:
        index = _oa_ir.IndexType.get()
        boolean = _oa_ir.IntegerType.get_signless(1)
        i64 = _oa_ir.IntegerType.get_signless(64)
        for old, shape, element_bytes in todo:
            with _oa_ir.InsertionPoint(old):
                def operation(name, operands, typ, attributes=None):
                    return _oa_ir.Operation.create(name,operands=operands,results=[typ],
                        attributes=attributes or {},loc=old.location).results[0]
                def constant(n):
                    return operation('arith.constant',[],index,{'value':_oa_ir.IntegerAttr.get(index,n)})
                def compare(a,b,predicate):
                    return operation('arith.cmpi',[a,b],boolean,
                        {'predicate':_oa_ir.IntegerAttr.get(i64,predicate)})
                def check(condition,message):
                    _oa_ir.Operation.create('cf.assert',operands=[condition],
                        attributes={'msg':_oa_ir.StringAttr.get(message)},loc=old.location)
                zero, one = constant(0), constant(1)
                dimensions, dynamic = [], iter(old.operands)
                empty = compare(zero,one,0)
                for size in shape:
                    dimension = next(dynamic) if size < 0 else constant(size)
                    check(compare(dimension,zero,5),'owned allocation: negative dimension')
                    dimensions.append(dimension)
                    empty = operation('arith.ori',[empty,compare(dimension,zero,0)],boolean)
                product, limit = constant(element_bytes), constant(_OWNED_ALLOCATION_LIMIT)
                for dimension in dimensions:
                    factor = operation('arith.select',[empty,one,dimension],index)
                    maximum = operation('arith.divui',[limit,product],index)
                    check(compare(factor,maximum,7),'owned allocation: byte product exceeds budget')
                    product = operation('arith.muli',[product,factor],index)
    module.operation.verify()
    return len(todo)


_OA_ORIGINAL_RUN_STAGES = _run_stages
def _run_stages(ctx,module,pipeline,erase,mid=(),late=(),post_openmp=(),pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_owned_result_allocations__'
    if marker not in passes:
        raise ValueError('owned allocation policy was not consumed by its selected pipeline')
    if passes.count(marker) != 1 or 'convert-scf-to-cf' not in passes:
        raise ValueError('owned allocation guard seam is missing or duplicated')
    cut = passes.index(marker)
    if cut >= passes.index('convert-scf-to-cf'):
        raise ValueError('owned allocation guards must precede control/index conversion')
    _OA_ORIGINAL_RUN_STAGES(ctx,module,','.join(passes[:cut]),erase,mid,(),(),pre_generalize)
    print('OK owned_allocation_guards',_owned_allocation_guards(ctx,module))
    _OA_ORIGINAL_RUN_STAGES(ctx,module,','.join(passes[cut+1:]),0,(),late,post_openmp,())
"""
