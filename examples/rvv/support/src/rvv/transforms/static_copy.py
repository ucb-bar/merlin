"""Default-off static strided-copy lowering using sequential SCF loops."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

RUNNER = r"""
import json,sys,hashlib
from torch_mlir import ir
source,destination,report_path,max_rank=sys.argv[1:]
max_rank=int(max_rank)
context=ir.Context()
context.allow_unregistered_dialects=True
module=ir.Module.parse(open(source).read(),context)
counts={'rewritten':0,'refusals':{},'copies':[]}
def refuse(reason):
    counts['refusals'][reason]=counts['refusals'].get(reason,0)+1
def walk(op):
    for region in op.regions:
        for block in region.blocks:
            for child in list(block.operations):
                yield from walk(child.operation)
    yield op
def contiguous(shape,strides):
    expected=1
    for size,stride in reversed(list(zip(shape,strides))):
        if size!=1 and stride!=expected:
            return False
        expected*=size
    return True
for op in list(walk(module.operation)):
    if op.name!='memref.copy':
        continue
    if len(op.operands)!=2 or len(op.results):
        refuse('unexpected_copy_signature');continue
    source_value,destination_value=list(op.operands)
    if not all(isinstance(v.type,ir.MemRefType) for v in op.operands):
        refuse('unranked_memref');continue
    src,dst=(ir.MemRefType(v.type)for v in op.operands)
    if not src.has_static_shape or not dst.has_static_shape:
        refuse('dynamic_shape');continue
    shape=list(src.shape)
    if shape!=list(dst.shape) or src.element_type!=dst.element_type:
        refuse('incompatible_copy');continue
    if len(shape)>max_rank:
        refuse('rank_budget');continue
    if any(size<0 or size>2**63-1 for size in shape):
        refuse('invalid_extent');continue
    if not isinstance(src.element_type,ir.IntegerType) or ir.IntegerType(src.element_type).width not in (8,16,32,64):
        refuse('unsupported_element_type');continue
    if any(str(t.memory_space)not in('None','0 : i64','0 : i32')for t in(src,dst)):
        refuse('non_cpu_memory_space');continue
    try:
        src_strides,src_offset=src.get_strides_and_offset()
        dst_strides,dst_offset=dst.get_strides_and_offset()
    except (RuntimeError,ValueError):
        refuse('non_strided_layout');continue
    if contiguous(shape,src_strides)and contiguous(shape,dst_strides):
        refuse('contiguous_upstream_fastpath');continue
    with context,ir.Location.unknown(),ir.InsertionPoint(op):
        index=ir.IndexType.get()
        def const(value):
            return ir.Operation.create('arith.constant',results=[index],attributes={'value':ir.IntegerAttr.get(index,value)}).results[0]
        zero,one=const(0),const(1)
        bounds=[const(size)for size in shape]
        def emit(dimension,indices):
            if dimension==len(shape):
                alignment={'alignment':ir.IntegerAttr.get(ir.IntegerType.get_signless(64),1)}
                value=ir.Operation.create('memref.load',results=[src.element_type],operands=[source_value,*indices],attributes=alignment)
                ir.Operation.create('memref.store',operands=[value.results[0],destination_value,*indices],attributes=alignment)
                return
            loop=ir.Operation.create('scf.for',operands=[zero,bounds[dimension],one],regions=1)
            block=loop.regions[0].blocks.append(index)
            with ir.InsertionPoint(block):
                emit(dimension+1,[*indices,block.arguments[0]])
                ir.Operation.create('scf.yield')
        emit(0,[])
    counts['copies'].append({'shape':shape,'source_type':str(src),'destination_type':str(dst),
       'element_bits':ir.IntegerType(src.element_type).width,
       'iteration_order':'lexicographic_sequential','alias_assumptions_added':False,
       'allocations_added':0,'source_strides':list(src_strides),'destination_strides':list(dst_strides)})
    op.erase()
    counts['rewritten']+=1
assert module.operation.verify()
counts['compiler_binding_sha256']={str(m.__file__):hashlib.sha256(open(m.__file__,'rb').read()).hexdigest()
    for name,m in sys.modules.items() if name.startswith('torch_mlir') and getattr(m,'__file__',None)}
open(destination,'w').write(str(module))
open(report_path,'w').write(json.dumps(counts,indent=2)+'\n')
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_pipeline(pipeline: str) -> tuple[str, str]:
    """Split once while SCF is structured and bufferization copies exist."""
    anchor = ",convert-scf-to-cf,"
    if pipeline.count(anchor) != 1 or pipeline.count("convert-scf-to-cf") != 1:
        raise ValueError("one bare convert-scf-to-cf boundary required")
    before, after = pipeline.split(anchor)
    return before, "convert-scf-to-cf," + after


def lower_file(
    source: str | Path,
    *,
    pipeline: str,
    mlir_opt: str | Path,
    compiler_python: str | Path,
    work: str | Path,
    enabled: bool = False,
    max_rank: int = 4,
    timeout: int = 120,
) -> dict:
    """Explicit opt-in native lowering; default leaves the pipeline unchanged.

    The selected compiler_python provides torch_mlir.ir and imports no Merlin.
    The caller supplies its complete provider pipeline. Work must be new.
    Allocations, descriptor layouts and sequential alias behavior are preserved.
    The returned LLVM-dialect file uses the ordinary mlir-translate/codegen path.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit boolean")
    if type(max_rank) is not int or not 1 <= max_rank <= 8:
        raise ValueError("max_rank must be an integer in 1..8")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("positive timeout required")
    source, work = Path(source).resolve(), Path(work).resolve()
    mlir_opt, compiler_python = (
        Path(mlir_opt).resolve(),
        Path(compiler_python).absolute(),
    )
    before, after = split_pipeline(pipeline) if enabled else ("", "")
    work.mkdir(parents=True, exist_ok=False)
    output = work / "model.llvm.mlir"
    record = {
        "schema": "rvv.static_strided_copy_lowering.v1",
        "enabled": enabled,
        "pass": False,
        "source": str(source),
        "source_sha256": _sha(source),
        "pipeline": pipeline,
        "alias_assumptions_added": False,
        "allocations_added": 0,
        "runtime_dependency_added": False,
        "vlen_hardcoded": False,
        "input_sha256": {
            str(p): _sha(p)
            for p in (source, mlir_opt, compiler_python, Path(__file__).resolve())
        },
        "commands": [],
    }

    def run(argv, name):
        record["commands"].append([str(a) for a in argv])
        result = subprocess.run(
            [str(a) for a in argv], capture_output=True, timeout=timeout, check=False
        )
        (work / (name + ".stdout")).write_bytes(result.stdout)
        (work / (name + ".stderr")).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}")

    try:
        if enabled:
            bufferized, rewritten = (
                work / "model.copy-before.mlir",
                work / "model.copy-after.mlir",
            )
            script = work / "rewrite_copies.py"
            script.write_text(RUNNER)
            run(
                [
                    mlir_opt,
                    source,
                    "--pass-pipeline=builtin.module(" + before + ")",
                    "-o",
                    bufferized,
                ],
                "prepare",
            )
            run(
                [
                    compiler_python,
                    script,
                    bufferized,
                    rewritten,
                    work / "copies.json",
                    max_rank,
                ],
                "rewrite",
            )
            run(
                [
                    mlir_opt,
                    rewritten,
                    "--pass-pipeline=builtin.module(" + after + ")",
                    "-o",
                    output,
                ],
                "finish",
            )
            record["copy_report"] = json.loads((work / "copies.json").read_text())
        else:
            run(
                [
                    mlir_opt,
                    source,
                    "--pass-pipeline=builtin.module(" + pipeline + ")",
                    "-o",
                    output,
                ],
                "lower",
            )
            record["copy_report"] = {"rewritten": 0, "refusals": {}, "copies": []}
        if _sha(source) != record["source_sha256"]:
            raise RuntimeError("source changed during lowering")
        record.update(
            {
                "pass": True,
                "llvm_dialect": str(output),
                "llvm_dialect_sha256": _sha(output),
            }
        )
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        record["output_sha256"] = {
            str(p.relative_to(work)): _sha(p) for p in work.rglob("*") if p.is_file()
        }
        (work / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
