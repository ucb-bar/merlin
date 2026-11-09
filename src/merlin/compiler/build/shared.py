"""Build the existing host shared library with explicitly supplied tool selectors.

Retains model/runtime/provider objects, source transactions, helper runtimes
and exact link boundaries. Compiler defaults stay with the selecting caller.
"""

from pathlib import Path

from merlin.common import compile_trace

#: What this module produces from LLVM IR, as compile-trace stages: each object it compiles, and each
#: program it links.
STAGES = compile_trace.declare(
    "codegen",
    ("object", "link"),
    entry="merlin.compiler.build.shared",
    summary="LLVM IR -> object (clang) -> linked program",
)



def build_shared_library(ll_path, out_so, *, run, compiler, object_compiler,
                         runtime_compiler, runtime_source, source_lowering=None):
    """Host .so (ctypes execution on x86), with the MLIR C-runtime shim.

    The runtime shim uses ``__bf16``. Use the same selected clang as the model
    object by default, so an older system ``cc`` cannot break host references;
    an explicit ``$CC`` remains an operator selection.
    """
    out_so = Path(out_so)
    model_o = out_so.with_suffix(".o")
    rt_o = out_so.with_name("mlir_runtime_host.o")
    original_run = run
    if source_lowering is not None:
        from merlin.llvmlower.native_source_model import NativeSourceModel

        if type(source_lowering) is not NativeSourceModel:
            raise ValueError("native source linking requires its original typed transaction")
        source_lowering.begin_codegen(out_so.parent)

        def run(command, **kwargs):
            return source_lowering.run(command, runner=original_run, **kwargs)

    selected_llc = object_compiler()
    if selected_llc is None:
        run(
            [compiler(), "-O2", "-fPIC", "-c", ll_path, "-o", model_o],
            inputs=(ll_path,),
            outputs=(model_o,),
        )
    else:
        run(
            [selected_llc, "-O2", "-filetype=obj", "-relocation-model=pic", ll_path, "-o", model_o],
            inputs=(ll_path,),
            outputs=(model_o,),
        )
    compile_trace.artifact("object", [model_o], pipeline="codegen")
    host_cc = runtime_compiler()
    runtime = runtime_source()
    run(
        [host_cc, "-O2", "-fPIC", "-c", str(runtime), "-o", rt_o],
        inputs=(runtime,),
        outputs=(rt_o,),
        stage="runtime_object",
    )
    provider_objects = ()
    if source_lowering is not None:
        provider_objects = source_lowering.objects(
            Path(ll_path), model_o, rt_o, runtime_source=runtime, compiler=host_cc,
            flags=("-O2", "-fPIC"), runner=original_run,
        )
    from merlin.llvmlower.adjacent_exp import runtime_c_for as exp_runtime_c_for
    from merlin.llvmlower.guarded_zero_contraction import runtime_c_for

    selected_runtime = runtime_c_for(Path(ll_path).read_text()) + exp_runtime_c_for(Path(ll_path).read_text())
    guarded_objects = ()
    if selected_runtime:
        helper_c = out_so.with_name("guarded_zero_contraction_runtime.c")
        helper_o = helper_c.with_suffix(".o")
        helper_c.write_text(selected_runtime)
        run(
            [host_cc, "-O2", "-fPIC", "-c", helper_c, "-o", helper_o],
            inputs=(helper_c,),
            outputs=(helper_o,),
            stage="runtime_object",
        )
        guarded_objects = (helper_o,)
    objects = (model_o, rt_o, *provider_objects, *guarded_objects)
    native_flags = ("-nostartfiles", "-Wl,-z,defs") if source_lowering is not None else ()
    run(
        [host_cc, "-shared", *native_flags, *objects, "-lm", "-o", out_so],
        inputs=objects,
        outputs=(out_so,),
        stage="link",
    )
    compile_trace.artifact("link", [out_so], pipeline="codegen")
    if source_lowering is not None:
        source_lowering.complete(objects, out_so, link_flags=(*native_flags, "-lm"))
    return out_so
