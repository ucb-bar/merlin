"""Explicit returned-result selection for the ordinary bare-metal model API.

Model lowering is source-only. The separate invocation blob supplies exact
runtime inputs; result dimensions are observed after the original source call.
"""

import inspect
import shutil
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json
from merlin.common.paths import runtime_dir
from merlin.common.strict_json import loads
from merlin.frontends.execution_result_signature import validate_signature
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.owned_result_driver import generate, load_arguments, verify
from merlin.llvmlower.owned_result_lowering import lower_owned_results
from merlin.runtime.execution_memory import admit_execution_memory
from merlin.runtime.owned_result_readback import PROFILE


def build_if_selected(request):
    """Dispatch only an explicit v6 signature; refuse unimplemented options."""
    from . import spike_model

    signature = loads(Path(request["runtime_signature"]).read_bytes())
    if signature.get("schema") != "merlin.compile-signature.v6":
        return None
    supported = {
        "model_dir", "work", "inputs_npz", "arena_mb", "dram_base", "dram_bytes", "backend", "features",
        "cflags_override", "vlen", "stack_bytes", "code_reserve", "full_readback", "readonly_input_indices",
        "max_readonly_input_bytes", "runtime_signature", "execution_memory_map", "execution_memory_reservations",
    }
    for name, parameter in inspect.signature(spike_model.build).parameters.items():
        if name in supported:
            continue
        value, default = request[name], parameter.default
        if not (value is None if default is None else type(value) is type(default) and value == default):
            raise ValueError(f"execution-owned target build does not implement selected option {name}")
    if not request["full_readback"] or request["inputs_npz"] is None:
        raise ValueError("execution-owned target build requires complete readback and positional runtime inputs")
    count = len(signature["inputs"])
    if request["readonly_input_indices"] != tuple(range(count)):
        raise ValueError("execution-owned target build requires complete original input observation")
    return _build(request, signature)


def _build(request, signature):
    from . import spike, spike_model
    from .host_compiler_policy import selected_model_compiler_plan

    source_path = Path(request["model_dir"]) / "model.mlir"
    source = source_path.read_text()
    signature = validate_signature(parse_mlir_text(source), signature)
    arrays = load_arguments(source, signature, request["inputs_npz"],
                            max_input_bytes=request["max_readonly_input_bytes"])
    for key in ("arena_mb", "stack_bytes"):
        if type(request[key]) is not int or request[key] <= 0:
            raise ValueError("execution-owned target build needs positive arena/stack extents")
    if request["stack_bytes"] % 16:
        raise ValueError("execution-owned target stack must retain 16-byte alignment")
    if (type(request["dram_bytes"]) is not int or request["dram_bytes"] <= 0
            or type(request["dram_base"]) is not int or request["dram_base"] < 0
            or request["dram_base"] + request["dram_bytes"] > 1 << 64):
        raise ValueError("execution-owned target build requires an explicit bounded DRAM span")
    if request["code_reserve"] is not None and (
            type(request["code_reserve"]) is not int or request["code_reserve"] <= 0):
        raise ValueError("execution-owned target code reserve must be positive")
    work = Path(request["work"]).resolve()
    work.mkdir(parents=True, exist_ok=False)
    runtime, abi, harness = runtime_dir() / "c", runtime_dir() / "abi", spike.harness_dir()
    features = frozenset(request["features"] or ())
    plan = selected_model_compiler_plan(backend=request["backend"], cflags_override=request["cflags_override"],
                                        vlen=request["vlen"], features=features)
    observed, flags, gcc = plan["observation"], plan["gcc_cflags"], spike.gcc_path()
    compilation = CompilationRecipe(work, producer=Path(__file__))
    with compilation.phase("source-only owned-result lowering"):
        lowered = lower_owned_results(source, workdir=work / "lower", data_layout=observed["data_layout"],
            index_bits=observed["index_bits"], entry=signature["entry"], features=features,
            max_bytes=signature["allocation_policy"]["max_bytes"], timeout=30)
    record = generate(source, signature, arrays, work / "cgen", max_input_bytes=request["max_readonly_input_bytes"])
    for name, path in (("source", source_path), ("signature", Path(request["runtime_signature"])),
                       ("invocation", Path(request["inputs_npz"])), ("driver", work / "cgen/driver.json")):
        compilation.bind_preparation(name, path)
    arena_bytes = request["arena_mb"] << 20
    layout = spike_model._layout(arena_bytes, max(1, record["input_bytes"]), dram_base=request["dram_base"],
        dram_bytes=request["dram_bytes"], code_reserve=request["code_reserve"] or (64 << 20))
    blob = work / "weights.bin"
    shutil.copyfile(work / "cgen/invocation.bin", blob)
    if not blob.stat().st_size:
        blob.write_bytes(b"\x00")  # Physical sentinel only; the typed invocation still has zero input bytes.
    ld = gcc.with_name(gcc.name.replace("gcc", "ld"))
    compilation.run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", blob.name],
        runner=spike_model._run, inputs=[blob], output=work / "weights_blob.o", cwd=work)
    objects = []

    def compile_unit(path, name, compiler, options, headers=()):
        output = work / name
        compilation.run([*compiler, *options, "-c", path, "-o", output], runner=spike_model._run,
                        inputs=[path, *headers], output=output)
        objects.append(output)

    compile_unit(lowered.llvm, "model.o", [observed["compiler_resolved"]], observed["cross_flags"])
    includes = ["-I", abi, "-I", runtime, "-I", harness, "-std=c11"]
    common_headers = [abi / "owned_result_views.h", runtime / "owned_result_driver.h",
                      runtime / "owned_result_readback.h", harness / "htif.h"]
    definitions = [f'-DMERLIN_OWNED_INPUT_BASE_ADDR={layout["weights_base"]}ULL',
                   f'-DMERLIN_OWNED_INPUT_BYTES={record["input_bytes"]}ULL']
    units = [
        (work / "cgen/model_owned.c", "driver.o", []),
        (runtime / "owned_result_driver.c", "control.o", []),
        (runtime / "owned_result_readback.c", "readback.o", []),
        (abi / "owned_result_views.c", "views.o", []),
        (abi / "owned_results.c", "owners.o", ["-DMERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1",
                                               "-DMERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1"]),
        (harness / "owned_result_main.c", "main.o", definitions),
        (harness / "crt.S", "crt.o", []), (harness / "htif.c", "console.o", []),
        (harness / "libc_min.c", "libc.o", []),
        (harness / "merlin_malloc.c", "malloc.o", [f'-DMERLIN_ARENA_BASE_ADDR={layout["arena_base"]}ULL',
                                                  f'-DMERLIN_ARENA_SIZE_BYTES={arena_bytes}ULL']),
    ]
    for path, name, definitions in units:
        compile_unit(path, name, [gcc], [*flags, *includes, *definitions], common_headers)
    runtime_compiler = spike_model._mlir_runtime_compiler(Path(observed["compiler_resolved"]), gcc, flags)
    compile_unit(abi / "mlir_runtime.c", "mlir.o", runtime_compiler, [], [abi / "half_extend.h"])
    libm, libm_hash, gcc_hash = spike_model._selected_libm_archive(gcc, flags, ())
    image = work / "model.elf"
    compilation.run([gcc, *flags, "-nostdlib", "-nostartfiles",
        f'-Wl,--defsym,MERLIN_WEIGHTS_BASE={layout["weights_base"]}',
        f'-Wl,--defsym,MERLIN_STACK_BYTES={request["stack_bytes"]}',
        f'-Wl,--defsym,{spike_model.DRAM_BASE_SYMBOL}={request["dram_base"]}',
        f'-Wl,--defsym,{spike_model.DRAM_SPAN_SYMBOL}={layout["mem_bytes"]}',
        f'-Wl,--defsym,{spike_model.HART_COUNT_SYMBOL}=1',
        "-T", harness / "model_link.ld", *objects, work / "weights_blob.o", libm, "-lgcc", "-o", image],
        runner=spike_model._run, inputs=[harness / "model_link.ld", *objects, work / "weights_blob.o", libm],
        output=image)
    if (sha256_file(observed["compiler_resolved"]) != observed["compiler_sha256"]
            or sha256_file(gcc) != gcc_hash or sha256_file(libm) != libm_hash):
        raise ValueError("selected owned-result compiler or math archive changed during build")
    lowered.verify()
    verify(source, signature, arrays, work / "cgen", max_input_bytes=request["max_readonly_input_bytes"])
    reservations = spike_model._model_memory_reservations(image, arena_base=layout["arena_base"],
        arena_bytes=arena_bytes, stack_bytes=request["stack_bytes"])
    admission = None
    if request["execution_memory_map"] is not None:
        admission = admit_execution_memory(image, request["execution_memory_map"],
            reservations + request["execution_memory_reservations"])
    elif request["execution_memory_reservations"]:
        raise ValueError("additional owned-result memory reservations require a selected decoded map")
    # Even a functional simulator image must keep loaded code/stack/blob and the
    # allocator disjoint inside the caller's declared span. This is not a board map.
    from merlin.runtime.elf_audit import read_elf

    entry, segments, _sections = read_elf(image, strict=True)
    end = request["dram_base"] + request["dram_bytes"]
    intervals = []
    for segment in segments:
        if segment.paddr != segment.vaddr or not request["dram_base"] <= segment.vaddr < segment.end <= end:
            raise ValueError("owned-result loaded image exceeds its declared memory span")
        intervals.append((segment.vaddr, segment.end))
    if not any(segment.vaddr <= entry < segment.end and "X" in segment.flags for segment in segments):
        raise ValueError("owned-result entry is outside executable loaded memory")
    for reservation in reservations:
        if not request["dram_base"] <= reservation.begin < reservation.begin + reservation.bytes <= end:
            raise ValueError("owned-result runtime reservation exceeds its declared memory span")
        intervals.append((reservation.begin, reservation.begin + reservation.bytes))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("owned-result loaded image, stack and allocator must be disjoint")
    compilation.completed(image)
    report = {"elf": str(image), "mem_bytes": layout["mem_bytes"], "vlen": request["vlen"],
              "schema": "merlin.execution-owned-image.v1", "result_mode": "execution-owned",
              "readback_profile": PROFILE, "driver": record, "index_lowering": observed,
              "arena_base": layout["arena_base"], "arena_bytes": arena_bytes, "weights_base": layout["weights_base"],
              "allocation_context": "serial-nonconcurrent", "harts": 1, "console": "htif",
              "source_execution": "once; result extents observed after original computation",
              "execution_memory_admission": admission,
              "scope": "functional compiler/runtime image; no accelerator or timing qualification"}
    write_pretty_json(work / "owned-result-image.json", report)
    return report
