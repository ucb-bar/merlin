"""An OPEN model -- host regions between its groups -- as one program: the model's own host code, and
one dispatch call per device group answered by the package, the target's library, or the host.

    record = build(package_dir, model_capsule, target=t, machine=m, header=h, extra=npz, out=o)

:mod:`merlin.analysis.whole_model.build.stages` builds a CLOSED model: every compute group is a device call and
the only host work is the input's quantization, so the program is a list of library/kernel calls. A
transformer is not closed. Its norms, softmax, rotary embedding and per-token dynamic quantization
compute between the contractions, and on an int8 unit they are host work by the target's own
declaration (a float datapath the unit does not have). Refusing such a model -- ``model_closure`` --
is right for a program that only strings group calls together; it is wrong for a program that can
run the host part too.

THE HOST PART IS THE MODEL'S OWN IR. Nothing here writes a second spelling of a layer norm. The
capture's module is lowered for the target's host ISA by the stock whole-model lowering
(:mod:`merlin.compiler.lower.model`), with each device group cut out and replaced by a call to
``merlin_dispatch_g<index>`` -- an external function with the C interface the lowering already emits.
The cut is EXACTLY the group's committed value (the accumulator an int8 contraction commits, before
the widening cast the capture follows it with), so the host code on either side is unchanged.

EACH DISPATCH HAS THREE ANSWERS, attributed per group, never summed:

* ``package`` -- the package's own kernel for that group, asked through the same capsule route the
  closed build asks (:func:`merlin.compiler.lower.device.whole_program.whole_program_buffer` with
  ``open_model=True``) and bound by the same ABI resolution;
* ``vendor`` -- the target's library call, on the loop-free path the target driver names when an
  instruction role is prohibited;
* ``host`` -- the group's own IR, lowered as a function of its own (``merlin_host_g<index>``), which is
  also the answer when the machine cannot read the group's result out (a full-width accumulator on a
  narrow-readout machine) -- with that cause.

Which one a group takes is recorded with the reason. The C that makes the call -- the library call,
the kernel's argument order, the timing brackets -- is the TARGET's (its whole-model driver's
``render_dispatch``); this module writes none of it and names no target.

THE ORACLE IS NUMPY. :mod:`merlin.evaluation.model.linalg_semantics` evaluates the capture's module with no
compiler in it. Every device group is then graded LOCALLY on the device: its exact integer result is
checked on the core against the inputs it actually read (a projection check, see the driver). The
host code is graded where it hands data to the device (a sample of each dispatch's input against the
oracle, within a declared bound) and at the model's output against the capsule's golden under the
capsule's own numeric policy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.analysis.whole_model.build.builder import (
    _CHEAP_PRODUCERS,
    _chunk_bounds,
    chunk_forward,
    forward_body_size,
    resolve_chunk_ops,
)
from merlin.analysis.whole_model.build.open_execution import (
    BENCH_SCHEMA,
    PRUNABLE,
    build_kernel_bench,
    grade_kernel_bench,
    is_open_model,
    main,
    run_functional,
    service_build,
)
from merlin.analysis.whole_model.build.open_program import (
    BUFFER_ACCESS,
    DISPATCH_PREFIX,
    HOST_PREFIX,
    Dispatch,
    OpenModelError,
    _committed_members,
    _is_zero,
    _shape_dtype,
    externalize_dispatches,
)

# ---- whole_model_open --------------------------------------------------------------------------


def build(
    package_dir: str | Path | None,
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    extra: str | Path | None = None,
    out: str | Path,
    verify: str = "local",
    timeout: int = 600,
    jobs: int | None = None,
    decline: Sequence[Any] = (),
    prohibited_roles: Sequence[str] = (),
    dram_base: int = 0x80000000,  # derived-ok: the RISC-V platform DRAM base the bare-metal harness links at
    dram_bytes: int | None = None,
    hostin_bound: int = 1,
    compile_timeout: int = 7200,
    profile: bool = False,
    require_device_readout: bool = True,
    prune: bool = True,
    host_hart: int | None = None,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    chunk_ops: int | str | None = None,
) -> dict[str, Any]:
    """Build ``model_capsule`` as one program: its host code and one dispatch per device group.

    ``package_dir`` answers the dispatches it lowers (``None``: the target's library, on the loop-free
    path when ``prohibited_roles`` names a role). ``extra`` supplies the leaf arguments the capsule's
    files do not (see :func:`forward_arguments`). ``verify='local'`` adds each dispatch's projection
    check and the sampled activation check (outside every bracket, inside the whole window -- a timing
    build passes ``'none'``). ``dram_bytes`` packs the image into a board's DRAM; ``None`` uses the
    functional simulator's map. ``hostin_bound`` is the DECLARED bound, in the activation's integer
    steps, of the chained activation check. Returns the build record, also written to
    ``<out>/whole_model_open_build.json``; the expectations a run is graded against are in
    ``<out>/expectations.json``. ``host_hart`` builds a TWO-HART program for a machine whose registry
    entry declares its harts: the host code, compiled for that hart's own ISA (its vector unit, when it
    has one), runs there, and every device dispatch is handed to the unit's hart (see :func:`two_harts`).
    ``phase0_recipe`` / ``descriptor`` name the corpus binding every dispatch put to the package is
    stated under (:func:`merlin.analysis.whole_model.build.stages.corpus_binder`); required with a package.

    ``chunk_ops`` splits ``forward``'s one flat body into that many-op-bounded, sequentially-called
    functions before it is lowered (see :func:`chunk_forward`): LLVM's compile cost on the single giant
    function a large open model emits is superlinear in its size. ``None`` (the default) keeps the
    unchunked program byte for byte -- an opt-in path until a real build's gate validates it.
    ``"auto"`` derives the size from the forward itself (:func:`resolve_chunk_ops`); the record's
    ``forward_chunks`` carries the size actually used (``None`` when the forward fit in one chunk).
    """
    import os
    import shutil

    import numpy as np

    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.analysis.whole_model.build.open_program import (
        HARNESS_SOURCES,
        HOST_CODE_OPTIONS,
        HOST_LOWERING_FEATURES,
        MACHINE_CANNOT_READ_OUT,
        SCHEMA,
        _c_include_dirs,
        _cross_flags,
        _device_part,
        _lifted,
        _run,
        _sha256,
        check_unit_hart_code,
        compile_host_units,
        forward_arguments,
        functional_arena_bytes,
        host_code_units,
        module_text,
        run_concurrently,
        two_harts,
    )
    from merlin.common import provenance as PROV
    from merlin.compiler.lower.c import runtime as c_runtime
    from merlin.compiler.lower.model import toolchain
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.backends.baremetal import image_link as SM
    from merlin.runtime.backends.baremetal import model_build as _model_elf_metadata
    from merlin.runtime.backends.baremetal import model_build as _model_memory_layout

    if verify not in ("local", "none"):
        raise OpenModelError(f"verify is 'local' or 'none', not {verify!r}")
    try:
        resolve_chunk_ops(chunk_ops)  # a misspelled size is refused before anything is built
    except ValueError as exc:
        raise OpenModelError(str(exc)) from exc
    stages = WMB._StageClock()
    capsule = WMB.load_model_capsule(model_capsule)
    abi_header = WMB.machine_header(machine, header, header_sha256)
    harts = two_harts(machine, host_hart) if host_hart is not None else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or min(16, os.cpu_count() or 1)
    driver = backends.whole_model_driver(target)

    # THE MACHINE FIRST. Every dispatch of an open model commits its accumulator, so on a machine whose
    # header states no full-width readout none of them can run on the unit, and the build below would
    # refuse after lowering every group (about an hour for a transformer). Refused here instead, before
    # the package is asked anything, unless the caller asked for exactly that all-host program.
    recipe = WMB._with_header(backends.harness_build_recipe(target), Path(header), out / "harness")
    includes = [f"-I{root}" for root in recipe.include_roots]
    parameter_header = driver.program._parameter_header(None, includes)
    full_width = driver.program.full_width_readout(parameter_header)
    if full_width is not True and require_device_readout:
        raise OpenModelError(
            f"{MACHINE_CANNOT_READ_OUT}: every device group of an open model commits the accumulator, and "
            f"{machine!r}'s parameter header states no full-width accumulator readout, so none of them can "
            f"run on its unit; build for a machine whose header states one"
        )

    # THE ORACLE depends on the model and the target, never on the package: it starts now, in a process
    # of its own (or is read back from its cache), and is joined where the routes first need it.
    from merlin.analysis.whole_model.build.open_execution import OpenOracleJob

    oracle_job = OpenOracleJob(capsule, target=target, extra=extra)
    try:
        arguments, argument_sources = forward_arguments(capsule, extra=extra)

        # 1-3. THE DEVICE PART, stated and put to the package; the cut over the normalized module.
        # Every dispatch put to the package is stated as the capsule the corpus writes for it.
        binding = WMB.corpus_binder(target, phase0_recipe=phase0_recipe, descriptor=descriptor) if package_dir else None
        with stages("device_part"):
            part = _device_part(
                capsule,
                target=target,
                package_dir=package_dir,
                out=out,
                timeout=timeout,
                jobs=jobs,
                decline=decline,
                binder=binding.binder if binding is not None else None,
            )
        buffer, rows, by_group, stated = part["buffer"], part["rows"], part["by_group"], part["stated"]
        main, host, dispatches, normalized = part["main"], part["host"], part["dispatches"], part["normalized"]
        # 5. THE HOST CODE AND THE GROUPS' OWN BODIES, lowered by the stock whole-model lowering.
        program = out / "program"
        program.mkdir(parents=True, exist_ok=True)
        chunks_made = 0
        chunk_size = resolve_chunk_ops(
            chunk_ops, forward_ops=forward_body_size(main) if chunk_ops is not None else None
        )
        if chunk_size is not None:
            with stages("chunk_forward"):
                chunks_made = chunk_forward(main, chunk_ops=chunk_size)
        main_text = module_text(main)
        if profile:
            # WHERE THE HOST CODE'S CYCLES GO: a mark before every top-level op of the host code (the
            # profiler the whole-model bare-metal path already has), with the id -> op table beside it.
            # It changes the emitted code, so a profiled image is for the split, never for the count.
            from merlin.analysis.whole_model.build.builder import chunk_symbols
            from merlin.compiler.lower.model import profile as op_profile

            # A chunked forward's own top-level ops are its chunk calls: mark inside them too.
            main_text, table = op_profile.instrument(main_text, ["forward", *chunk_symbols(chunks_made)])
            op_profile.write_table(table, out / "op_profile_table.json")
        (program / "main.mlir").write_text(main_text, encoding="utf-8")
        (program / "host.mlir").write_text(str(host), encoding="utf-8")
        cross = [SM.CLANG_TARGET, *_cross_flags(recipe.cflags), "-O2", "-ffreestanding", "-fno-builtin"]
        if harts is not None:
            # The host code, and only the host code, is compiled for the host hart's own ISA: everything the
            # unit's hart executes (the dispatches, the harness, the kernels) keeps the recipe's.
            cross = [f"-march={harts['host_isa']}" if f.startswith("-march=") else f for f in cross]
        with stages("host_code"):
            objects = compile_host_units(
                host_code_units(program),
                program,
                cross=cross,
                dispatches=dispatches,
                compile_timeout=compile_timeout,
                jobs=jobs,
            )

        # THE JOINS, where the routes first need what ran behind the cut and the host code.
        with stages("objects_join"):
            part["object_dedup"] = part["objects"].result()
        with stages("oracle_join"):
            expected = oracle_job.result([d.group for d in dispatches])
    except BaseException:
        oracle_job.cancel()
        raise

    # 4. ROUTES. The package's kernel where it answered and its operands are the dispatch's own bytes;
    # else the library, on the loop-free path when an instruction role is prohibited; else the host.
    loop_free = None
    library = None
    if prohibited_roles:
        loop_free = driver.program._trap_prohibited_in_library(
            includes, None, sorted(WMB._prohibited(target, prohibited_roles)), out
        )
        library = driver.program.library_path_without_loops(parameter_header.read_text(encoding="utf-8"))
    host_library = getattr(driver.program, "LIBRARY_HOST", None)
    routes: dict[int, dict[str, Any]] = {}
    attribution: list[dict[str, Any]] = []
    for dispatch in dispatches:
        row, statement = by_group[dispatch.group], stated[dispatch.group]
        seen = expected["dispatch"][dispatch.group]
        op = str(statement.get("op"))
        entry = {"group": dispatch.group, "op": op, "region": dispatch.region, "zero_init": dispatch.zero_init}
        route: dict[str, Any]
        if row["on"] == WMB.ON_PACKAGE:
            operands = {str(v): str(k) for k, v in (statement.get("operands") or {}).items()}
            roles = [operands.get(str(a.get("program"))) for a in row["args"]]
            weight = next((a for a in row["args"] if operands.get(str(a.get("program"))) == "rhs"), None)
            if None in roles or any(a.get("gather") for a in row["args"]):
                route = {"on": WMB.ON_HOST, "cause": "kernel_argument_not_a_dispatch_operand", "why": f"roles {roles}"}
            elif weight is not None and weight.get("sha256") not in (None, seen["rhs_sha256"]):
                route = {
                    "on": WMB.ON_HOST,
                    "cause": "device_weight_is_not_the_operand",
                    "why": "the package reads its weight in a layout whose bytes differ from the operand the "
                    "host code hands the dispatch; linking it would need the laid-out copy embedded",
                }
            else:
                route = {"on": WMB.ON_PACKAGE, "symbol": row["symbol"], "object": row["object"], "args": roles}
        else:
            path = (library or {}).get(op, {}) if library else {}
            route = {
                "on": WMB.ON_VENDOR,
                "path": path.get("path") or getattr(driver.program, "LIBRARY_DEFAULT", "WS"),
                "declined_as": row.get("cause"),
                "why": row.get("why"),
            }
            if path.get("host"):
                route.update({"library_path_on_host": True, "library_path_why": path.get("why")})
        accelerated = route["on"] == WMB.ON_PACKAGE or (
            route["on"] == WMB.ON_VENDOR and route.get("path") != host_library
        )
        if accelerated and full_width is not True:
            route = {
                "on": WMB.ON_HOST,
                "cause": driver.program.ACCUMULATOR_READOUT_UNAVAILABLE,
                "why": "the machine's parameter header states no full-width accumulator readout, and this "
                "dispatch commits the accumulator; its own IR computes it on the core",
                "declined": route["on"],
            }
        route["zero_init"] = dispatch.zero_init
        routes[dispatch.group] = route
        entry.update(route)
        if route["on"] == WMB.ON_VENDOR and route.get("path") == host_library:
            # THE LIBRARY'S OWN HOST CODE: the call is the library's, the work is the core's.
            entry.update({"on": WMB.ON_HOST, "cause": "library_loop_free_path", "library_call": True})
        attribution.append(entry)
    stages.mark("routes")

    # 6. THE RUNTIME: the capture's arguments as the generic C runtime binds them.
    bundle = out / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    (bundle / "model.mlir").write_text(str(main), encoding="utf-8")
    os.link(capsule.weights, bundle / "weights.safetensors")
    shutil.copyfile(capsule.weights_manifest, bundle / "weights.safetensors.manifest.json")
    manifest = json.loads(capsule.weights_manifest.read_text(encoding="utf-8"))
    signature_inputs = [
        (str((manifest.get(str(i)) or {}).get("name") or ""), i)
        for i in range(len(arguments))
        if (manifest.get(str(i)) or {}).get("kind") == "input" and not _lifted(manifest.get(str(i)) or {})
    ]
    np.savez(bundle / "inputs.npz", **{f"in{n}": arguments[i] for n, (_name, i) in enumerate(signature_inputs)})
    (bundle / "input_order.json").write_text(json.dumps({name: n for n, (name, _i) in enumerate(signature_inputs)}))
    if argument_sources["extra"] is not None:
        # ONLY THE LEAVES THE MODEL READS, under the capture's own keys: the runtime generator matches
        # lifted constants by sorted position, so an unrelated key in the file would shift them.
        source = np.load(argument_sources["extra"]["path"])
        np.savez(bundle / "extra.npz", **{key: source[key] for key in argument_sources["extra"]["keys"]})
    cgen = program / "cgen"
    info = c_runtime.generate(bundle, cgen, bundle / "inputs.npz", prepared_dir=program)
    # THE CODE RESERVE GROWS WITH WHAT IS LINKED. A package's kernels are its own code, and a
    # fully-scheduled kernel per group is large (measured: 0.4-1.7 MB each, ~150 MB for 302 groups),
    # so the region ahead of the weights is sized by the objects' own bytes -- an upper bound on what
    # they load -- rather than a constant that the first large package overruns at link time.
    linked_code = sum(Path(o).stat().st_size for o in objects) + sum(
        Path(str(r["object"])).stat().st_size for r in routes.values() if r["on"] == WMB.ON_PACKAGE
    )
    code_reserve = _model_memory_layout._CODE_RESERVE_FIXED + int(info.get("static_io_bytes", 0)) + linked_code
    arena = (
        {
            # Everything the board has past the code reserve and the weights, less the alignment the
            # map places each region on (a megabyte per region, two regions).
            "bytes": (int(dram_bytes) - code_reserve - int(info["weights_bytes"])) - (4 << 20) & ~((1 << 20) - 1),
            "basis": "dram_bytes",
        }
        if dram_bytes
        else functional_arena_bytes(
            program,
            [name for name, _text in host_code_units(program)],
            dispatches=dispatches,
            row_multiple=WMB.pointee_row_padding(target)["multiple"],
        )
    )
    arena_bytes = int(arena["bytes"])
    layout = _model_memory_layout._layout(
        arena_bytes,
        int(info["weights_bytes"]),
        dram_base=int(dram_base),
        dram_bytes=int(dram_bytes) if dram_bytes else None,
        code_reserve=code_reserve,
    )

    stages.mark("runtime")

    # 7. THE TARGET'S C: every dispatch's call, and the program's main with its protocol lines.
    samples = {
        d.group: {
            "indices": expected["dispatch"][d.group]["sample_indices"],
            "values": expected["dispatch"][d.group]["sample_values"],
            "bound": int(hostin_bound),
        }
        for d in dispatches
    }
    rendered = driver.dispatch.render_dispatch(
        [{**d.to_dict(), "op": stated[d.group].get("op")} for d in dispatches],
        routes,
        uart=driver.program.UART,
        verify=verify,
        samples=samples,
        # A digest of every dispatch's result bytes, taken after the window in EVERY build: it is what
        # shows a timing build (no checks) wrote the same bytes as the locally graded build.
        words_helper=getattr(driver.program, "_WORDS_HELPER", ""),
        # The pitch the kernel ABI pads every pointee row to: a package kernel reads and writes its
        # operands at it, and a dense row that is not a whole number of tile edges is not that layout.
        row_padding=WMB.pointee_row_padding(target)["multiple"],
        **({"unit_rpc": True} if harts is not None else {}),
    )
    # THE TARGET'S WORD IS THE LAST: a route its driver cannot write (a library path with no
    # full-width result) is the host's, with the driver's cause, and the attribution says so.
    final = {int(row["group"]): row for row in rendered["census"]}
    for entry in attribution:
        written = final[int(entry["group"])]
        if written["on"] != routes[int(entry["group"])]["on"]:
            entry.update(
                {
                    "on": written["on"],
                    "cause": written.get("cause"),
                    "why": written.get("why"),
                    "declined": written.get("declined"),
                }
            )
            entry.pop("library_call", None)
    (program / "dispatch.c").write_text(rendered["source"], encoding="utf-8")
    policy = CC.load_capsule(capsule.directory).get("numeric_policy") or {}
    (program / "main.c").write_text(
        driver.dispatch.render_main(
            driver.program.UART,
            rendered["census"],
            profile=profile,
            # THE END RESULT IS THE ORACLE'S OUTPUT UNDER THE CAPSULE'S OWN POLICY, stated by the program.
            reference=np.asarray(expected["outputs"][0], np.float32).reshape(-1).tolist(),
            atol=float(policy.get("atol", 0.0)),
            rtol=float(policy.get("rtol", 0.0)),
            words_helper=getattr(driver.program, "_WORDS_HELPER", ""),
            **({"host_hart": harts["host"], "unit_hart": harts["unit"]} if harts is not None else {}),
        ),
        encoding="utf-8",
    )

    # 8. COMPILE AND LINK, with the target's own compiler for C and the harness's layout.
    harness = SM._harness_dir()
    runtime = SM._c_runtime_dir()
    from merlin.common.paths import runtime_dir

    gcc_flags = [*_cross_flags(recipe.cflags), "-O2", "-ffreestanding", "-fno-builtin"]
    address = [
        f"-DMERLIN_ARENA_BASE_ADDR={hex(layout['arena_base'])}ULL",
        f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL",
        f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(layout['weights_base'])}ULL",
    ]
    base_includes = ["-I", runtime, "-I", cgen, "-I", harness]
    units = {
        "model_call.o": (cgen / "model_call.c", base_includes),
        "merlin_model.o": (runtime / "merlin_model.c", base_includes),
        "main.o": (program / "main.c", [*base_includes, *address]),
        "crt.o": (harness / "crt.S", []),
        "console.o": (harness / "htif.c", []),
        # Word-wide memcpy/memset: the host code's bufferized copies are its own, and byte loops made
        # them ~4 instructions a byte (measured on a one-group SmolVLA regime).
        "libc_min.o": (harness / "libc_min.c", ["-DMERLIN_WORD_MEMOPS", "-fno-tree-loop-distribute-patterns"]),
        "printf_min.o": (harness / "printf_min.c", ["-I", harness]),
        "malloc.o": (harness / "merlin_malloc.c", [*address, "-I", harness]),
        "dispatch.o": (program / "dispatch.c", [*includes, "-DBAREMETAL=1", "-Wno-incompatible-pointer-types"]),
    }
    if profile:
        table_size = len(json.loads((out / "op_profile_table.json").read_text(encoding="utf-8"))) + 2
        units["op_prof.o"] = (
            runtime / "merlin_op_prof.c",
            ["-DMERLIN_PROF_BAREMETAL", f"-DMERLIN_PROF_MAX_OPS={table_size}", "-I", harness],
        )
    # THE MLIR RUNTIME IS BUILT BY THE COMPILER THAT BUILT THE MODEL. Its bf16 conversion helpers are
    # called by the lowered model, and the register a bf16 travels in is the CALLER's convention: the
    # clang-lowered model passes and returns it in a float register, a GCC build of the helper returns
    # it in an integer one (src/merlin/runtime/native/abi/mlir_runtime.c says so), and every bf16 value the host
    # code narrows then comes back as garbage. Measured: the text model's activations off by ~100 steps.
    runtime_object = program / "mlir_rt.o"
    system_includes = [flag for root in _c_include_dirs(recipe.compiler) for flag in ("-isystem", root)]
    # Every unit is its own compile and none reads another's object: they run at once (a transformer's
    # dispatch.c and the MLIR runtime each take most of a minute), and link in the order listed.
    compiles = [
        [recipe.compiler, *gcc_flags, *extra_flags, "-c", source, "-o", program / name]
        for name, (source, extra_flags) in units.items()
    ]
    compiles.append(
        [toolchain.clang(), *cross, *system_includes, "-c", runtime_dir() / "abi/mlir_runtime.c", "-o", runtime_object]
    )
    stages.mark("render")
    run_concurrently(compiles, jobs=jobs, timeout=compile_timeout)
    stages.mark("c_units")
    objects.extend(program / name for name in units)
    objects.append(runtime_object)
    compiler = Path(recipe.compiler)
    linker = compiler.with_name(compiler.name[: -len("gcc")] + "ld") if compiler.name.endswith("gcc") else None
    if linker is None or not linker.is_file():
        raise OpenModelError(f"no linker beside the target's compiler {compiler} to package the weights blob")
    _run([linker, "-r", "-b", "binary", "-o", program / "weights_blob.o", "weights.bin"], cwd=cgen)
    objects.append(program / "weights_blob.o")
    kernels = [Path(o) for o in rendered["objects"]]
    elf = program / "model.elf"
    _run(
        [
            recipe.compiler,
            *gcc_flags,
            "-nostdlib",
            "-nostartfiles",
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(layout['weights_base'])}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={hex(1 << 22)}",
            # The DRAM span the image was laid out for, stated IN the image: a simulator that runs it
            # reads this rather than defaulting to a span the arena lies past (_model_elf_metadata.declared_memory).
            f"-Wl,--defsym,{_model_elf_metadata.DRAM_BASE_SYMBOL}={hex(int(dram_base))}",
            f"-Wl,--defsym,{_model_elf_metadata.DRAM_SPAN_SYMBOL}={hex(int(layout['mem_bytes']))}",
            # How many harts the image runs on, stated IN it, as its memory span is.
            *([f"-Wl,--defsym,{_model_elf_metadata.HART_COUNT_SYMBOL}={harts['count']}"] if harts is not None else []),
            "-T",
            harness / "model_link.ld",
            *objects,
            *kernels,
            "-lm",
            "-lgcc",
            "-o",
            elf,
        ],
        timeout=compile_timeout,
    )
    stages.mark("link")

    if harts is not None:
        harts["unit_code_checked"] = len(
            check_unit_hart_code(
                [*objects, *kernels],
                host_objects=[program / "model_main.o", program / "model_host.o", runtime_object],
                unit_isa=harts["unit_isa"],
            )
        )
    isa = None
    if prohibited_roles:
        from merlin.analysis.whole_model.grading.isa_prohibition import check_program

        isa = check_program(
            elf,
            target=target,
            roles=prohibited_roles,
            compiler=recipe.compiler,
            group_objects={str(r["group"]): Path(r["object"]) for r in rows if r.get("object")},
            library_groups=[str(a["group"]) for a in attribution if a["on"] != WMB.ON_PACKAGE],
        )
        if not isa["clean"]:
            raise OpenModelError(f"the linked program issues a prohibited instruction: {isa['summary']}")
    stages.mark("isa_scan")

    golden = np.asarray(next(iter(capsule.outputs.values())), np.float32).reshape(-1)
    declared = CC.load_capsule(capsule.directory)
    expectations = {
        "schema": "whole_model_open_expectations_v1",
        "groups": [d.group for d in dispatches],
        "golden": golden.tolist(),
        "oracle_output": np.asarray(expected["outputs"][0], np.float32).reshape(-1).tolist(),
        "numeric_policy": declared.get("numeric_policy") or {},
        "hostin_bound": int(hostin_bound),
        "dispatch": {str(k): v for k, v in expected["dispatch"].items()},
    }
    (out / "expectations.json").write_text(json.dumps(expectations) + "\n", encoding="utf-8")
    # WHAT THIS PROGRAM'S OUTPUT DEPENDS ON APART FROM WHO ANSWERED ITS GROUPS -- the key its reference
    # arm is cached under (whole_model_reference). The layout's addresses are not in it: they move with
    # the size of the linked kernels and change no number the host code computes.
    from merlin.analysis.whole_model.build import oracle as REF

    host_files = {
        **{f"program/{n}": program / n for n in ("main.mlir", "host.mlir", "main.c", "mlir_rt.o")},
        **{f"program/{p.name}": p for p in (program / "model_main.o", program / "model_host.o")},
        **{f"cgen/{p.name}": p for p in sorted(cgen.iterdir()) if p.is_file() and p.suffix in (".c", ".h")},
        **{f"harness/{n}": harness / n for n in HARNESS_SOURCES},
        "runtime/merlin_model.c": runtime / "merlin_model.c",
        "runtime/abi/mlir_runtime.c": runtime_dir() / "abi/mlir_runtime.c",
    }
    leaves = (argument_sources.get("extra") or {}).get("path")
    reference_identity = {
        "capsule": REF.files_identity(
            {"interface": capsule.interface, "weights": capsule.weights, **({"leaves": leaves} if leaves else {})}
        ),
        "host_code": REF.files_identity(
            host_files,
            [
                *cross,
                *gcc_flags,
                *(["profile"] if profile else []),
                # How the host code was lowered changes its objects as surely as its flags do.
                *(f"lowering:{feature}" for feature in HOST_LOWERING_FEATURES),
                *(f"host_code:{option}" for option in HOST_CODE_OPTIONS),
                # A unit's own flags change its object as surely as its source does.
                *(f"{name}:{flag}" for name, (_source, flags) in sorted(units.items()) for flag in flags),
            ],
            root=out,
        ),
        "toolchain": REF.toolchain_identity(recipe.compiler, toolchain.clang(), gcc_flags),
        "machine": {"machine": machine, "header_sha256": (abi_header or {}).get("sha256") or REF.UNKNOWN},
    }
    counts: dict[str, int] = {}
    for row in attribution:
        counts[row["on"]] = counts.get(row["on"], 0) + 1
    stages.mark("expectations")
    record = {
        "schema": SCHEMA,
        "target": target,
        "machine": machine,
        # Wall seconds per build stage: a slow build is fixed at the stage that is slow.
        "stage_seconds": stages.record(),
        "corpus_binding": binding.record if binding is not None else None,
        "object_dedup": part["object_dedup"],
        "forward_chunks": {
            "chunk_ops": chunk_size,
            "chunks": chunks_made,
            **({"requested": chunk_ops} if chunk_ops != chunk_size else {}),
        },
        "oracle_cache": oracle_job.state,
        "capsule": {
            "name": capsule.name,
            "directory": str(capsule.directory),
            "interface_sha256": _sha256(capsule.interface),
        },
        "package": None
        if package_dir is None
        else {
            "directory": str(Path(package_dir).resolve()),
            "replies": buffer["whole_program"].get("package_replies"),
        },
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        "verify": verify,
        "profile": profile,
        "arguments": argument_sources,
        "normalization": normalized,
        "layout": {k: (hex(v) if isinstance(v, int) else v) for k, v in layout.items()}
        | {"arena_bytes": hex(arena_bytes), "dram_base": hex(int(dram_base))},
        # Where the arena's size came from: the board's DRAM, or the program's own heap demand.
        "arena": arena,
        "program": {
            "abi_header": abi_header,
            "compiler": str(recipe.compiler),
            "flags": gcc_flags,
            "cross_flags": cross,
            "includes": includes,
            "full_width_readout": full_width,
            "harts": harts,
            "library_paths": library,
            "loop_free_header": loop_free,
            "linked_kernels": [{"path": str(k), "sha256": _sha256(k)} for k in kernels],
        },
        # The kernel objects linked, by content, in the closed build's spelling (isa_prohibition.check_build).
        "linked_objects": [{"path": str(k), "sha256": _sha256(k)} for k in kernels],
        "dispatches": [d.to_dict() for d in dispatches],
        "host_regions": buffer["whole_program"].get("host_regions"),
        "attribution": {"counts": counts, "per_group": attribution},
        "isa_prohibition": isa,
        "expectations": str(out / "expectations.json"),
        "reference_identity": reference_identity,
        "provenance": PROV.record(
            pins=WMB._pins_for(target),
            sources=[capsule.interface],
            artifacts={"elf": elf, "weights": capsule.weights},
        ),
    }
    (out / "whole_model_open_build.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    if prune:
        record["pruned_bytes"] = prune_intermediates(out)
    return record


def prune_intermediates(out: str | Path) -> int:
    """Remove what a finished open build kept only to link; returns the bytes freed.

    A package build asks 302 groups and compiles each, and its work trees reached ~14 GB per build.
    What stays is what a reader or a later step needs: the ELF and its records, the expectations, and
    each linked kernel object (the prohibition check reads their symbols)."""
    import shutil

    out = Path(out)
    targets = [
        out / "lower",
        out / "bundle",
        out / "harness",
        out / "program" / "lower_main",
        out / "program" / "lower_host",
    ]
    objects = out / "objects"
    if objects.is_dir():
        targets += [d for d in objects.iterdir() if d.is_dir()]
    freed = 0
    for target in targets:
        if target.is_dir():
            freed += sum(f.lstat().st_size for f in target.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(target, ignore_errors=True)
    return freed


if __name__ == "__main__":
    raise SystemExit(main())
