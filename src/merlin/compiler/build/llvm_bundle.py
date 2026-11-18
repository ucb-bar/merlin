"""Link and finalize explicit LLVM region inputs with caller-owned task hooks.

Completion semantics and target task ABI construction are supplied by the OOT
adapter. This owner provides no completion or execution admission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from merlin.compiler.partition.callback_regions import WholeProgramError, WholeProgramResult


def materialize_llvm_bundle(
    result: WholeProgramResult,
    output: str | Path | None = None,
    *,
    symbol: str,
    cflags: Sequence[str],
    host_optimization: str | None = None,
    select_host_optimization: Callable[[str | None], str],
    mlir_install: Path,
    prepare_linked_module: Callable,
    finalize_imported_module: Callable,
) -> Path:
    """Produce one LLVM-dialect module including host, device and descriptor glue.

    The caller explicitly supplies its selected CPU and ABI flags. Only descriptor glue
    is compiled from C; tensor arithmetic remains in the shared host product and
    callback-supplied target modules. All external runtime calls remain visible.
    """
    import subprocess
    from math import prod

    from merlin.compiler.lower.model.toolchain import clang, llvm_install, mlir_translate  # noqa: F401

    selected_install = Path(mlir_install)
    translate = selected_install / "bin" / "mlir-translate"
    if not symbol.isidentifier() or not symbol.isascii():
        raise WholeProgramError("invalid whole-program kernel symbol")
    if not any(f.startswith("-march=") for f in cflags) or not any(f.startswith("-mabi=") for f in cflags):
        raise WholeProgramError("descriptor glue needs explicit selected host CPU and ABI flags")
    abi = result.stats["adapter_abi"]
    entry = abi["entry"]
    if not entry.isidentifier() or not entry.isascii():
        raise WholeProgramError("unsupported host entry symbol")
    work = result.routed_source.parent / "combined"
    work.mkdir(exist_ok=True)

    def run(args):
        completed = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise WholeProgramError("whole-program materialization failed: " + completed.stderr[-6000:])

    # The selected bare-metal C runtime supplies memset. Local standard
    # declarations avoid importing an unrelated host libc/sysroot ABI.
    include = work / "host_runtime_include"
    include.mkdir(exist_ok=True)
    (include / "string.h").write_text(
        "void *memset(void *, int, __SIZE_TYPE__);\nvoid *memcpy(void *, const void *, __SIZE_TYPE__);\n"
    )
    typedefs, declarations, locals_, addresses = [], [], [], []
    ranks = sorted({len(slot["shape"]) for slot in abi["slots"]})
    for rank in ranks:
        arrays = f"long sizes[{rank}]; long strides[{rank}];" if rank else ""
        typedefs.append(
            f"typedef struct {{ void *allocated; void *aligned; long offset; {arrays} }} descriptor_{rank};"
        )
    for i, slot in enumerate(abi["slots"]):
        shape = slot["shape"]
        rank = len(shape)
        stride = [prod(shape[j + 1 :]) for j in range(rank)]
        arrays = ", {" + ", ".join(map(str, shape)) + "}, {" + ", ".join(map(str, stride)) + "}" if rank else ""
        locals_.append(f"  descriptor_{rank} d{i} = {{ p{i}, p{i}, 0{arrays} }};")
        addresses.append(f"&d{i}")
        declarations.append(f"descriptor_{rank} *")
    wrapper = (
        "\n".join(typedefs)
        + '\n_Static_assert(sizeof(long) == 8 && sizeof(void *) == 8, "ranked ABI requires LP64");\n'
    )
    wrapper += f"extern void _mlir_ciface_{entry}(" + ", ".join(declarations) + ");\n"
    wrapper += f"void {symbol}(" + ", ".join(f"void *p{i}" for i in range(len(abi["slots"]))) + ") {\n"
    wrapper += "\n".join(locals_) + f"\n  _mlir_ciface_{entry}(" + ", ".join(addresses) + ");\n}\n"
    wrapper_path = work / "entry.c"
    wrapper_path.write_text(wrapper)
    llvm_inputs = [result.host_llvm_ir.resolve()]
    for i, c_source in enumerate((wrapper_path,) + ((result.shim_c,) if result.shim_c else ())):
        ll = work / f"glue_{i}.ll"
        run([clang(), *cflags, "-O1", "-ffreestanding", "-I", include, "-S", "-emit-llvm", c_source, "-o", ll])
        llvm_inputs.append(ll)
    for i, artifact in enumerate(result.device_artifacts):
        if artifact.format == "llvm-dialect":
            ll = work / f"device_{i}.ll"
            run([translate, "--mlir-to-llvmir", Path(artifact.path).resolve(), "-o", ll])
            llvm_inputs.append(ll)
        elif artifact.format == "llvm-ir":
            llvm_inputs.append(Path(artifact.path).resolve())
        else:
            raise WholeProgramError("single-module materialization needs LLVM device artifacts")
    linked = work / "combined.ll"
    run([selected_install / "bin" / "llvm-link", "-S", *llvm_inputs, "-o", linked])
    target = Path(output) if output is not None else work / "combined.mlir"
    # Whole-program artifacts expose one executable entry. Inline the exact
    # linked helpers, retaining external runtime declarations as such.
    inline_input = work / "inline.ll"
    linked_text = linked.read_text()
    lines = []
    for line in linked_text.splitlines():
        if line.startswith("define ") and ("@" + symbol + "(") not in line:
            if not line.rstrip().endswith("{"):
                raise WholeProgramError("unsupported multiline LLVM function header")
            line = line.rstrip()[:-1] + " alwaysinline {"
        lines.append(line)
    prepared_text, public_symbols = prepare_linked_module("\n".join(lines) + "\n", symbol)
    if (
        type(prepared_text) is not str
        or type(public_symbols) is not tuple
        or symbol not in public_symbols
        or len(public_symbols) != len(set(public_symbols))
        or any(type(value) is not str or not value.isascii() or not value.isidentifier() for value in public_symbols)
    ):
        raise WholeProgramError("target task preparation returned an invalid selected public symbol roster")
    inline_input.write_text(prepared_text)
    inlined = work / "inlined.ll"
    # Keep one choice across source preparation, child lowering and native
    # materialization even if another compilation updates the tuning file.
    if host_optimization is None:
        host_optimization = result.stats.get("adapter_host_source_optimization")
    host_optimization = select_host_optimization(host_optimization)
    if host_optimization in ("bounded_rne", "bounded_rne_fused"):
        if result.stats.get("adapter_host_source_optimization") not in ("bounded_rne", "bounded_rne_fused"):
            raise WholeProgramError("bounded RNE needs its checked source preparation during host lowering")
        if host_optimization == "bounded_rne_fused" and not result.stats.get("adapter_stock_host_pipeline"):
            raise WholeProgramError("bounded RNE fusion needs its actually consumed stock scalar pipeline")
        if "-march=rv64gc" not in cflags or "-mabi=lp64d" not in cflags:
            raise WholeProgramError("bounded RNE requires the admitted scalar RV64GC/lp64d CPU contract")
    native_pipeline = "internalize,always-inline,sroa,instcombine,simplifycfg,globaldce"
    if host_optimization == "scalar_loops":
        # Native LLVM constructs loop/dominance/MemorySSA analyses after the
        # linked descriptor and target functions have been inlined. LICM uses
        # its ordinary alias and speculative-execution checks. Memory-clobbering
        # device asm remains opaque; no FP reassociation or vector pass is added.
        native_pipeline = (
            "internalize,always-inline,function(sroa,instcombine,simplifycfg,"
            "loop-simplify,lcssa,loop-mssa(licm),instcombine,simplifycfg),globaldce"
        )
    optimization_command = [
        selected_install / "bin" / "opt",
        "-S",
        "--passes=" + native_pipeline,
        "--internalize-public-api-list=" + ",".join(public_symbols),
        inline_input,
        "-o",
        inlined,
    ]
    optimization_remarks = work / "host-loop-remarks.yaml"
    optimization_remarks.unlink(missing_ok=True)
    if host_optimization == "scalar_loops":
        optimization_command += ["--pass-remarks=licm", "--pass-remarks-output=" + str(optimization_remarks)]
    run(optimization_command)
    numerical_legalization = None
    if host_optimization in ("bounded_rne", "bounded_rne_fused"):
        from merlin.compiler.lower.quant import late_quant_rne

        original_native = work / "before-bounded-rne.ll"
        original_native.write_bytes(inlined.read_bytes())
        rewritten, numerical_legalization = late_quant_rne.rewrite(
            original_native.read_text(), host_isa="rv64gc", combine_clamp=True
        )
        numerical_legalization["legalizer_sha256"] = hashlib.sha256(
            Path(late_quant_rne.__file__).read_bytes()
        ).hexdigest()
        numerical_legalization["combine_clamp"] = True
        (work / "host-bounded-rne.json").write_text(json.dumps(numerical_legalization, indent=2))
        selected_native = work / "bounded-rne.ll"
        selected_native.write_text(rewritten)
        # Remove the now-dead source DAG through ordinary LLVM DCE. No vector
        # flags, floating reassociation or source input assumptions are added.
        numerical_cleanup_command = [
            selected_install / "bin" / "opt",
            "-S",
            "--passes=function(instcombine,simplifycfg),globaldce",
            selected_native,
            "-o",
            inlined,
        ]
        run(numerical_cleanup_command)
    # simplifycfg may synthesize switches from descriptor/rank guards. The
    # public task CFG represents conditional and unconditional branches, so
    # use LLVM's semantics-preserving lowering after every optimizing pass.
    # No optimizer follows this step: it could reconstruct the switch.
    cfg_input = inlined
    inlined = work / "branch-cfg.ll"
    cfg_command = [
        selected_install / "bin" / "opt",
        "-S",
        "--passes=function(lower-switch),verify",
        cfg_input,
        "-o",
        inlined,
    ]
    run(cfg_command)
    run([translate, "--import-llvm", inlined, "-o", target])
    from xdsl.parser import Parser

    from merlin.compiler.dialects.xdsl_context import text as llvm_text
    from merlin.compiler.ir.llvm_context import make_llvm_context

    generic_import = work / "generic-import.mlir"
    run([selected_install / "bin" / "mlir-opt", target, "--mlir-print-op-generic", "-o", generic_import])
    module = Parser(make_llvm_context(), generic_import.read_text()).parse_module()
    finalize_imported_module(module, symbol, result.stats)
    result.stats["adapter_combined_llvm_dialect"] = str(target.resolve())
    alias_prefix = "\n".join(line for line in generic_import.read_text().splitlines() if line.startswith("#"))
    target.write_text(alias_prefix + "\n" + llvm_text(module, generic=True))
    # Parsing the actual resulting module catches unresolved MLIR dialect or
    # signature issues before handing the artifact to the caller.
    run([selected_install / "bin" / "mlir-opt", target, "-o", work / "verified.mlir"])

    def identity(path):
        payload = path.read_bytes()
        return dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))

    host_recipe = dict(
        schema="submission.scalar_host_optimization.v1",
        selection=host_optimization,
        stage="linked LLVM after exact descriptor and device inlining",
        selected_cpu_flags=list(cflags),
        pipeline=native_pipeline,
        command=list(map(str, optimization_command)),
        tool=identity(selected_install / "bin" / "opt"),
        input=identity(inline_input),
        selected_native_llvm=identity(inlined),
        emitted_llvm_dialect=identity(target),
        final_cfg_lowering=dict(
            command=list(map(str, cfg_command)), input=identity(cfg_input), output=identity(inlined)
        ),
        source_arithmetic="unchanged LLVM semantics; ordinary alias and speculative-execution checks",
        cost_evidence="not_measured",
    )
    if "scalar_window_reductions" in result.stats:
        from merlin.compiler.lower.model import scalar_window_reduction

        host_recipe["scalar_window_reductions"] = dict(
            result.stats["scalar_window_reductions"],
            binding_stage="caller_before_stock_bufferization",
            typed_source=identity(result.host_llvm_ir.parent / "model.scalar-windows.mlir"),
            shared_implementation=identity(Path(scalar_window_reduction.__file__)),
        )
    if optimization_remarks.is_file():
        host_recipe["loop_optimization_remarks"] = identity(optimization_remarks)
        host_recipe["loop_optimization_remark_count"] = sum(
            line.startswith("Pass:") and line.endswith("licm") and line[5:-4].isspace()
            for line in optimization_remarks.read_text().splitlines()
        )
    if numerical_legalization is not None:
        host_recipe["source_numerical_preparation"] = result.stats["adapter_host_numerical_preparation"]
        host_recipe["bounded_rne"] = numerical_legalization
        host_recipe["before_bounded_rne"] = identity(original_native)
        host_recipe["selected_bounded_rne"] = identity(selected_native)
        host_recipe["bounded_rne_cleanup_command"] = list(map(str, numerical_cleanup_command))
        host_recipe["source_arithmetic"] = (
            "Merlin checked bounded ties-even quantization; scalar RV64GC explicit RNE; original unsupported chains retained"
        )
        if "adapter_stock_host_pipeline" in result.stats:
            host_recipe["stock_host_lowering"] = result.stats["adapter_stock_host_pipeline"]
    (work / "host-optimization.json").write_text(json.dumps(host_recipe, indent=2))
    result.stats["adapter_host_optimization"] = host_recipe
    (work / "materialization.json").write_text(
        json.dumps(
            dict(
                schema="submission.whole_program_materialization.v1",
                symbol=symbol,
                host_cpu_flags=list(cflags),
                sources=[str(p) for p in llvm_inputs],
                output=str(target),
                runtime_symbols="external; supplied by selected host runtime",
            ),
            indent=2,
        )
    )
    return target
