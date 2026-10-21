"""Link existing session stages into one freestanding image, without running it.

The original caller supplies the complete output observer and memory interval.
Device stages retain ordinary preparation/object builders and one exact live
final-image admission service. No target effects or numerical laws are inferred.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir
from merlin.llvmlower import device_preparation as D
from merlin.llvmlower import session_bundle, session_freestanding
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.llvmlower.device_build import (
    admit_linked_device_elf,
    freeze_device_elf_custody,
    revalidate_device_elf_custody,
)

from . import session_build_guard as G
from . import spike_model as M
from . import spike_session_stage as S


def _pins(paths):
    return {Path(path).absolute(): sha256_file(path) for path in paths}


def _check(pins):
    if any(not path.is_file() or path.is_symlink() or sha256_file(path) != digest for path, digest in pins.items()):
        raise ValueError("session original source, tool or compiled artifact changed")


def _main():
    return """/* Existing single-hart runtime ABI; observation is outside every interval. */
#include <stdint.h>
#include "htif.h"
#include "merlin_session_freestanding.h"
extern int merlin_session_observe(int,long,int,const void *,size_t,void *);
uint64_t merlin_session_clock(void) {
  uint64_t value;
  __asm__ volatile("fence rw,rw" ::: "memory");
  __asm__ volatile("csrr %0, mcycle" : "=r"(value) :: "memory");
  __asm__ volatile("fence rw,rw" ::: "memory");
  return value;
}
void merlin_session_record_cycles(int kind,int program,long step,uint64_t ticks) {
  htif_puts("SESSION_CYCLES "); htif_putd(kind); htif_putc(' ');
  htif_putd(program); htif_putc(' '); htif_putd(step); htif_putc(' ');
  htif_putd((long)ticks); htif_putc('\\n');
}
int main(unsigned long hart) {
  if(hart) { for(;;) __asm__ volatile("wfi"); }
  console_init();
  int status=merlin_session_run_freestanding(merlin_session_observe,0);
  htif_puts(status ? "SESSION_REFUSED\\n" : "SESSION_DONE\\n");
  htif_exit(status ? 1 : 0);
}
"""


def build_session(
    root,
    work,
    *,
    limits,
    arena_bytes,
    stack_bytes,
    dram_base,
    dram_bytes,
    devices,
    observer_sources,
    observer_inputs=(),
    cflags_override=None,
    features=None,
    int8_compute=False,
    execution_memory_map=None,
):
    """Build, never execute, a serial LP64 session through the existing tool owners.

    ``observer_sources`` implement ``merlin_session_observe`` for every result;
    they are compiled only AFTER all device compiler invocations. Independent
    references belong to these original caller inputs, never the target compiler.
    Reset/setup/step counter intervals exclude observers and do not measure a
    continuous session or establish physical completion or hardware cycles.
    """
    if type(limits) is not session_freestanding.SessionMemoryLimits:
        raise ValueError("session build requires original typed memory limits")
    limits.__post_init__()
    for name, value in dict(
        arena_bytes=arena_bytes, stack_bytes=stack_bytes, dram_base=dram_base, dram_bytes=dram_bytes
    ).items():
        if type(value) is not int or not 0 < value < 1 << 63:
            raise ValueError(f"session {name} requires a positive signed-64-bit extent")
    if type(int8_compute) is not bool or (features is not None and type(features) is not frozenset):
        raise ValueError("session preparation requires original boolean/frozen feature selection")
    if type(observer_sources) is not tuple or not observer_sources or type(observer_inputs) is not tuple:
        raise ValueError("session needs original observer sources and an immutable input roster")
    root, work = Path(root).resolve(), Path(work).absolute()
    if work.exists() or work.resolve() != work or work == root or root in work.parents:
        raise ValueError("session build needs a fresh direct work directory outside original source")
    source_roster = frozenset(root.rglob("*"))
    if any(p.is_symlink() for p in source_roster):
        raise ValueError("session original source must not contain symlinks")
    source_pins = _pins(p for p in source_roster if p.is_file())
    observer_sources = tuple(Path(p).absolute() for p in observer_sources)
    observer_inputs = tuple(Path(p).absolute() for p in observer_inputs)
    source_pins.update(_pins((*observer_sources, *observer_inputs, Path(__file__), Path(S.__file__))))
    _check(source_pins)
    session = session_bundle.load(root)
    if type(devices) is not tuple or len(devices) != len(session.programs):
        raise ValueError("session requires one original routing or None per stage")
    guard_check, guard_code = G.check, G.check.__code__
    guard = G.snapshot((M, S, G, sys.modules[__name__], S.Z), devices)
    settings = (limits, cflags_override, features, int8_compute, execution_memory_map)
    settings_state = guard["reader"][0](settings)
    source_pins.update(_pins(guard["paths"]))
    plan = M.selected_model_compiler_plan(
        backend="scalar", cflags_override=cflags_override, vlen=None, features=features
    )
    gcc = M._spike.gcc_path()
    clang = Path(plan["observation"]["compiler_resolved"])
    tool_pins = _pins((gcc, clang))
    tool_pins.update(_pins((M.toolchain.nm(), G.elf_audit._tool("readelf"))))
    tool_pins.update(
        _pins((gcc.with_name("riscv64-unknown-elf-ld"), clang.parent / "llvm-nm", clang.parent / "llvm-objcopy"))
    )
    source_pins.update(
        _pins(
            p
            for folder in (M._harness_dir(), runtime_dir() / "c", runtime_dir() / "abi")
            for p in folder.iterdir()
            if p.is_file()
        )
    )
    archive, archive_sha, driver_sha = M._selected_libm_archive(gcc, plan["gcc_cflags"], ())
    tool_pins.update(_pins((archive,)))
    if execution_memory_map is not None:
        if type(execution_memory_map) is not G.execution_memory.MemoryMapBinding:
            raise ValueError("session needs its original typed memory-map binding")
        memory_record = execution_memory_map.validate()
        source_pins.update(_pins((execution_memory_map.path, *(item["path"] for item in memory_record["file_pins"]))))
    controls = dict(os.environ)
    stages, stage_pins, generated_pins = [], {}, {}

    def closed():
        if G.check is not guard_check or guard_check.__code__ is not guard_code:
            raise ValueError("session original implementation checker changed")
        guard_check(guard, devices)
        if frozenset(root.rglob("*")) != source_roster or dict(os.environ) != controls:
            raise ValueError("session original source roster or environment controls changed")
        _check(tool_pins)
        if guard["reader"][0](settings) != settings_state:
            raise ValueError("session original build settings changed")
        _check(source_pins)
        _check(stage_pins)
        _check(generated_pins)

    def current():
        closed()
        for device, selection in zip(devices, frozen, strict=True):
            if selection["admission"] is not None:
                from merlin.llvmlower.device_build import require_device_elf_admission

                require_device_elf_admission(device, selection["admission"])
            D.require_device_offload_selection(device, selection["offload"])
            closed()
        closed()

    frozen = S.freeze_routes(devices, len(session.programs))
    current()
    lay = M._layout(
        arena_bytes,
        limits.max_weights_bytes,
        dram_base=dram_base,
        dram_bytes=dram_bytes,
        code_reserve=limits.weights_base - dram_base,
    )
    if lay["weights_base"] != limits.weights_base or limits.address_limit > dram_base + dram_bytes:
        raise ValueError("session original weights interval does not match the existing packed runtime layout")
    if limits.max_static_io_bytes + stack_bytes > limits.weights_base - dram_base:
        raise ValueError("session static I/O and stack exceed the declared code/static reservation")
    work.mkdir(parents=True)
    compilation = CompilationRecipe(work, producer=Path(__file__))
    for index, path in enumerate(source_pins):
        compilation.bind_preparation(f"original_{index}", path)
    generated = work / "session_gen"
    generated_record = session_freestanding.generate(root, generated, limits=limits, timing=True)
    generated_pins = _pins(p for p in generated.rglob("*") if p.is_file())
    _check(source_pins)
    for index, (program, row, device, selection) in enumerate(
        zip(session.programs, generated_record["session"]["programs"], devices, frozen, strict=True)
    ):
        current()
        _check(generated_pins)
        _check(tool_pins)
        stage = S.build_stage(
            program,
            row,
            work / f"stage_{index}",
            device=device,
            frozen=selection,
            plan=plan,
            features=features,
            int8_compute=int8_compute,
            gcc=gcc,
            check_current=current,
        )
        stages.append(stage)
        stage_pins.update(_pins(p for p in (work / f"stage_{index}").rglob("*") if p.is_file()))
        current()
        _check(stage_pins)
        _check(tool_pins)
    # No target compiler receives any observer/reference source or object below this point.
    rt, harness = runtime_dir() / "c", M._harness_dir()
    for index, path in enumerate(stage_pins):
        compilation.bind_preparation(f"stage_product_{index}", path)
    for index, path in enumerate(generated_pins):
        compilation.bind_preparation(f"generated_{index}", path)
    runtime_cc = M._mlir_runtime_compiler(clang, gcc, plan["gcc_cflags"])
    units = [
        (rt / "merlin_model.c", ()),
        (runtime_dir() / "abi/mlir_runtime.c", ()),
        (generated / "merlin_session.c", ()),
        (generated / "merlin_session_freestanding.c", ()),
    ]
    for row in generated_record["session"]["programs"]:
        folder = Path(row["generated"])
        units.extend((folder / name, ("-I", folder)) for name in ("model_call.c", "stage_adapter.c"))
    main = work / "session_main.c"
    main.write_text(_main())
    units += [(main, ("-I", harness)), *((p, ()) for p in observer_sources)]
    objects = [stage["object"] for stage in stages]
    for index, (source, extra) in enumerate(units):
        obj = work / f"session_{index}.o"
        compilation.run(
            [*runtime_cc, "-I", rt, "-I", harness, "-I", generated, *extra, "-c", source, "-o", obj],
            runner=M._run,
            inputs=[source, *observer_inputs],
            output=obj,
        )
        objects.append(obj)
    address_defines = [
        f"-DMERLIN_ARENA_BASE_ADDR={lay['arena_base']}ULL",
        f"-DMERLIN_ARENA_SIZE_BYTES={arena_bytes}ULL",
    ]
    for name in ("crt.S", "htif.c", "libc_min.c", "merlin_malloc.c"):
        obj = work / f"runtime_{Path(name).stem}.o"
        compilation.run(
            [gcc, *plan["gcc_cflags"], *address_defines, "-c", harness / name, "-o", obj],
            runner=M._run,
            inputs=[harness / name],
            output=obj,
        )
        objects.append(obj)
    blob = work / "weights_blob.o"
    compilation.run(
        [gcc.with_name("riscv64-unknown-elf-ld"), "-r", "-b", "binary", "-o", blob, "weights.bin"],
        runner=M._run,
        inputs=[generated / "weights.bin"],
        output=blob,
        cwd=generated,
    )
    objects.append(blob)
    elf = work / "session.elf"
    compilation.run(
        [
            gcc,
            *plan["gcc_cflags"],
            "-nostdlib",
            "-nostartfiles",
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={limits.weights_base}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={stack_bytes}",
            f"-Wl,--defsym,{M.DRAM_BASE_SYMBOL}={dram_base}",
            f"-Wl,--defsym,{M.DRAM_SPAN_SYMBOL}={dram_bytes}",
            "-T",
            harness / "model_link.ld",
            *objects,
            archive,
            "-o",
            elf,
        ],
        runner=M._run,
        inputs=[harness / "model_link.ld", *objects, archive],
        output=elf,
    )
    product_pins = _pins(p for p in work.rglob("*") if p.is_file())
    _check(source_pins)
    _check(tool_pins)
    if sha256_file(archive) != archive_sha or sha256_file(gcc) != driver_sha:
        raise ValueError("session original math archive or compiler changed")
    active = [i for i, stage in enumerate(stages) if stage["arguments"]["signatures"]]
    admission = None
    custody = None
    if active:
        first = active[0]
        admission = admit_linked_device_elf(
            devices[first], frozen[first]["admission"], elf=elf, linked_sha256=product_pins[elf], directory=work
        )
        custody = freeze_device_elf_custody(
            devices[first], frozen[first]["admission"], elf=elf, linked_sha256=product_pins[elf], result=admission
        )
    for index, stage in enumerate(stages):
        D.finish_device_offload_selection(
            devices[index],
            frozen[index]["offload"],
            admission=frozen[index]["admission"],
            result=admission,
            elf=elf,
            linked_sha256=product_pins[elf],
            target_configuration=frozen[index]["configuration"],
            artifacts=stage["artifacts"],
        )
    if custody is not None:
        revalidate_device_elf_custody(devices[active[0]], frozen[active[0]]["admission"], custody)
    current()
    if custody is not None:
        revalidate_device_elf_custody(devices[active[0]], frozen[active[0]]["admission"], custody)
    _check(source_pins)
    _check(tool_pins)
    _check(product_pins)
    _check(stage_pins)
    _check(generated_pins)
    memory = _check_memory(elf, lay, arena_bytes, stack_bytes, dram_base, dram_bytes, execution_memory_map)
    _check(source_pins)
    _check(product_pins)
    current()
    _check(product_pins)
    compilation.completed(elf)
    recipe_sha = sha256_file(compilation.path)
    product_pins[compilation.path] = recipe_sha
    verify_completed_recipe(compilation.path, executable=elf, expected_recipe_sha256=recipe_sha)
    invocations = sorted(work.rglob("invocation.json"))
    for path in invocations:
        G.invocation_record.verify(path)
    _check(source_pins)
    _check(tool_pins)
    _check(product_pins)
    result = dict(
        elf=str(elf),
        sha256=sha256_file(elf),
        session=generated_record,
        memory=memory,
        stages=[
            {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in stage.items()
                if k in ("source", "llvm", "object", "recipe", "compiler_plan")
            }
            for stage in stages
        ],
        invocation_records=[dict(path=str(path), sha256=sha256_file(path)) for path in invocations],
        linked_elf_admission=admission,
        timing_scope=generated_record["timing"],
    )
    with (work / "session_build.json").open("x") as report:
        report.write(json.dumps(result, indent=2) + "\n")
    _check(source_pins)
    _check(product_pins)
    return result


def _check_memory(elf, layout, arena_bytes, stack_bytes, dram_base, dram_bytes, binding):
    from merlin.runtime.elf_audit import read_elf
    from merlin.runtime.execution_memory import admit_execution_memory

    _, segments, _ = read_elf(elf, strict=True)
    loaded = [segment for segment in segments if segment.kind == "LOAD"]
    if not loaded:
        raise ValueError("session actual ELF has no loaded segments")
    for segment in loaded:
        if segment.kind != "LOAD":
            continue
        if not dram_base <= segment.vaddr < segment.end <= dram_base + dram_bytes:
            raise ValueError("session actual ELF load lies outside declared memory interval")
        if max(segment.vaddr, layout["arena_base"]) < min(segment.end, layout["arena_base"] + arena_bytes):
            raise ValueError("session actual ELF load overlaps the declared allocator")
    reservations = M._model_memory_reservations(
        elf, arena_base=layout["arena_base"], arena_bytes=arena_bytes, stack_bytes=stack_bytes
    )
    if any(not dram_base <= item.begin < item.begin + item.bytes <= dram_base + dram_bytes for item in reservations):
        raise ValueError("session runtime stack or allocator exceeds declared memory")
    spans = [(item.begin, item.begin + item.bytes) for item in reservations]
    if any(max(a, c) < min(b, d) for i, (a, b) in enumerate(spans) for c, d in spans[i + 1 :]):
        raise ValueError("session runtime reservations overlap")
    if any(max(a, seg.vaddr) < min(b, seg.end) for a, b in spans for seg in loaded):
        raise ValueError("session runtime reservation overlaps actual ELF loads")
    if binding is not None:
        return admit_execution_memory(elf, binding, reservations)
    return dict(
        scope="declared interval arithmetic only; no decoded physical memory qualification",
        layout=layout,
        stack_bytes=stack_bytes,
        arena_bytes=arena_bytes,
    )
