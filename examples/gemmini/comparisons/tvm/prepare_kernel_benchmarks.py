"""Prepare exact kernel programs/fixtures and build complete-call checked ELFs.

This is a workload definition and packaging entry point, not a compiler pass.
Run one case at a time. Full reference preparation belongs on the remote CPU
worker; --list is cheap and prepare/build never launch FireSim.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from kernel_benchmark_data import CASES, array_sha256, case_spec, fixtures
from build_shared_chunk_elf import digest


def build_program(spec):
    import tvm
    from tvm import relax, te, tir
    shapes=dict(spec["roster"])
    if spec["kind"]=="conv":
        shapes["B"]=(shapes["W"][1],)
    params={name:relax.Var(name,relax.TensorStructInfo(shape,"int32" if name=="B" else "int8"))
            for name,shape in shapes.items()}
    bb=relax.BlockBuilder()
    with bb.function("main",list(params.values())):
        with bb.dataflow():
            outputs=[]
            if spec["kind"]=="conv":
                func=convolution(shapes)
                kernel=bb.add_func(func.without_attr("global_symbol"),"convolution")
                buffer=func.buffer_map[func.params[-1]]
                outputs.append(bb.emit(relax.call_tir(kernel,list(params.values()),
                    out_sinfo=relax.TensorStructInfo(buffer.shape,buffer.dtype))))
            else:
                for a,w,name in spec["products"]:
                    m,k=shapes[a]; _,n=shapes[w]
                    x,y=te.placeholder((m,k),"int8","a"),te.placeholder((k,n),"int8","w")
                    axis=te.reduce_axis((0,k),"k")
                    acc=te.compute((m,n),lambda i,j:te.sum(x[i,axis].astype("int32")*
                                   y[axis,j].astype("int32"),axis=axis),name="matmul")
                    out=acc
                    if spec["scale"] is not None:
                        def readout(i,j):
                            value=tir.nearbyint(acc[i,j].astype("float32")*tir.const(spec["scale"],"float32"))
                            return te.min(te.max(value,tir.const(-128,"float32")),tir.const(127,"float32")).astype("int8")
                        out=te.compute((m,n),readout,name="readout")
                    kernel=bb.add_func(te.create_prim_func([x,y,out]).without_attr("global_symbol"),"product")
                    outputs.append(bb.emit(relax.call_tir(kernel,[params[a],params[w]],
                        out_sinfo=relax.TensorStructInfo(out.shape,out.dtype))))
            result=bb.emit_output(outputs[0] if len(outputs)==1 else relax.Tuple(outputs))
        bb.emit_func_output(result)
    return bb.get()


def convolution(shapes):
    """Ordinary biased reduction semantics, with bias explicitly in its init.

    No post-import bias folding: this benchmark declares the biased reduction
    directly. VALID input borders retain supplied bytes; output ABI is flat NHWC.
    """
    from tvm.script import from_source
    b,hi,wi,ci=shapes["IFM"]; co=shapes["W"][1]; ho,wo=hi-2,wi-2
    return from_source(f'''@T.prim_func
def main(IFM: T.Buffer(({b},{hi},{wi},{ci}), "int8"), W: T.Buffer(({9*ci},{co}), "int8"), B: T.Buffer(({co},), "int32"), Y: T.Buffer(({b*ho*wo},{co}), "int8")):
    T.func_attr({{"tir.noalias": True}})
    C = T.alloc_buffer(({b*ho*wo},{co}), "int32")
    for row, oc, ry, rx, ic in T.grid({b*ho*wo},{co},3,3,{ci}):
        with T.block("conv"):
            vm,vo,vr,vs,vc = T.axis.remap("SSRRR", [row,oc,ry,rx,ic])
            with T.init():
                C[vm,vo] = B[vo]
            C[vm,vo] = C[vm,vo] + T.Cast("int32", IFM[vm//{ho*wo},vm//{wo}%{ho}+vr,vm%{wo}+vs,vc]) * T.Cast("int32", W[(vr*3+vs)*{ci}+vc,vo])
    for row, oc in T.grid({b*ho*wo},{co}):
        with T.block("readout"):
            vm,vo = T.axis.remap("SS",[row,oc])
            Y[vm,vo] = T.Cast("int8", T.min(T.max(T.nearbyint(T.Cast("float32", C[vm,vo]) * T.float32(1)), T.float32(-128)), T.float32(127)))
''')


def prepare(case, output):
    import numpy as np
    import tvm
    spec=case_spec(case)
    output.mkdir(parents=True,exist_ok=False)
    program=output/"program.json"
    program.write_text(tvm.ir.save_json(build_program(spec)))
    inputs,outputs=fixtures(spec)
    fixture=output/"fixture.npz"
    arrays={f"input_{i}":v for i,v in enumerate(inputs.values())}
    arrays.update({"expected" if len(outputs)==1 else f"expected_{i}":v for i,v in enumerate(outputs.values())})
    np.savez(fixture,**arrays)
    metadata=dict(case=case,specification=spec,program=dict(path=str(program.resolve()),sha256=digest(program)),
                  fixture_sha256=digest(fixture),comparison=dict(rtol=0.,atol=0.),
                  reference_contract="Exact signed-nibble kernel contract; complete outputs; independent NumPy INT64 accumulation and FP32 RNE",
                  reference_sources=[dict(path=str(Path(__file__).with_name("kernel_benchmark_data.py").resolve()),
                                          sha256=digest(Path(__file__).with_name("kernel_benchmark_data.py")))],
                  inputs=[dict(name=k,shape=list(v.shape),dtype=v.dtype.name,sha256=array_sha256(v)) for k,v in inputs.items()],
                  outputs=[dict(name=k,shape=list(v.shape),dtype=v.dtype.name,sha256=array_sha256(v)) for k,v in outputs.items()],
                  execution="NOT_RUN", compiler_semantics="ordinary mathematical TensorIR; convolution declares explicit bias init",
                  preparation_source_sha256=digest(__file__))
    (output/"reference.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(dict(case=case,outputs=len(outputs),status="prepared_not_executed")))


def build(args):
    import measure
    import tvm
    from chunk_measure import build_chunk
    root=args.prepared.resolve(strict=True); proof=json.loads((root/"reference.json").read_text())
    if digest(root/"program.json")!=proof["program"]["sha256"]:
        raise ValueError("Prepared program changed")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    app=Path(tvm.__file__).resolve().parents[2]/"apps/gemmini"
    command=[sys.executable,str(app/"compile_package.py"),"--program",str(root/"program.json"),
             "--output-dir",str(output/"package"),"--riscv-gcc",str(args.riscv_gcc.resolve()),
             "--adapter-receipt",str(args.adapter_receipt.resolve()),"--graph-fusion-level",str(args.graph_fusion_level),
             "--tensor-memory-bytes",str(256<<20),"--host-math-policy","native"]
    if args.graph_cse:
        command.append("--graph-cse")
    if args.tuning_database:
        command += ["--tuning-database",str(args.tuning_database.resolve())]
    (output/"compile-command.json").write_text(json.dumps(command,indent=2)+"\n")
    with (output/"compile.log").open("w") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    placement=json.loads((output/"package/placement.json").read_text())
    if not placement or any(row['placement']!='gemmini' for row in placement):
        raise ValueError("Kernel accelerator placement refused; inspect package/placement.json")
    class BuildOnlyToolchain(measure.Toolchain):
        def memory_configuration(self):
            return dict(kind="firesim_build_only",deployment_qualified=False)
    tc=BuildOnlyToolchain(gcc=args.riscv_gcc.resolve(),adapter_receipt=args.adapter_receipt.resolve(),
                          spike_receipt=None,hardware_receipt=None,timing_policy="fixed-warm")
    artifact=build_chunk(output/"package",root/"fixture.npz",root/"reference.json",output/"guest",tc,rtol=0.,atol=0.)
    artifact.update(case=proof["case"],execution="NOT_RUN",measurement_scope="complete kernel case, all outputs")
    (output/"guest/artifact.json").write_text(json.dumps(artifact,indent=2)+"\n")
    print(json.dumps(artifact["elfs"],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    actions.add_parser("list")
    prep=actions.add_parser("prepare");prep.add_argument("--case",choices=CASES,required=True)
    prep.add_argument("--output-dir",type=Path,required=True)
    compile_=actions.add_parser("build")
    for key in ("prepared","output-dir","riscv-gcc","adapter-receipt"):
        compile_.add_argument("--"+key,type=Path,required=True)
    compile_.add_argument("--tuning-database",type=Path)
    compile_.add_argument("--graph-fusion-level",type=int,choices=range(4),default=2)
    compile_.add_argument("--graph-cse",action="store_true")
    args=parser.parse_args()
    if args.action=="list":print("\n".join(CASES))
    elif args.action=="prepare":prepare(args.case,args.output_dir)
    else:build(args)


if __name__=="__main__":main()
