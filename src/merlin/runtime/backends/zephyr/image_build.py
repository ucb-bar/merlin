"""Builds a whole-model Zephyr application image."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from merlin.common.paths import runtime_dir
from merlin.compiler.backend.source_proposals import MatrixRouting
from merlin.compiler.build.model_preparation import (
    allocation_bytes,
    load_matrix_signatures,
    parallel_arms,
    prepare_for_lowering,
)
from merlin.compiler.build.zephyr import (
    TreeConfiguration,
    memory_plan,
    project_config,
    ram_for_weights,
    require_external_arena,
)
from merlin.compiler.build.zephyr import ZephyrBuildError as ZephyrModelError
from merlin.compiler.build.zephyr import cpu_disable_overlay as _cpu_disable_overlay
from merlin.compiler.ir.mlir_query import parse_scope
from merlin.compiler.lower.c import runtime as c_runtime
from merlin.compiler.lower.model import toolchain
from merlin.compiler.lower.model.lower import lower_model_file
from merlin.compiler.lower.options.masked_contraction import MaskEffectContract
from merlin.compiler.lower.source.carrier import host_admitted
from merlin.compiler.lower.source.interval import IntervalEffectContract
from merlin.runtime.backends.baremetal import spike_launcher as _spike
from merlin.runtime.backends.zephyr.application import (
    _cflags,
    _cmakelists,
    _completion_metric_line,
    _main_c,
    _run,
    _tool_env,
    _zephyr_base,
    build_available,
    build_tool,
    march_with_vlen,
)

# ---- zephyr_build ------------------------------------------------------------------------------

# ---- build / run -------------------------------------------------------------------


@parse_scope()
@host_admitted
def build_app(
    model_dir: str | Path,
    work: str | Path,
    *,
    board: str | None = None,
    backend: str = "rvv",
    rvv_hart: int = 0,
    cpus: int = 2,
    inputs_npz: str | Path | None = None,
    ram_bytes_override: int | None = None,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    rvv_schedule: str | None = None,
    cflags_override: list[str] | None = None,
    features: frozenset[str] | None = None,
    n_harts: int = 1,
    iters: int = 1,
    warmup: int = 0,
    omp_threads: int | None = None,
    vlen: int | None = None,
    sdk_dir: str | Path | None = None,
    debug: bool = False,
    matrix: MatrixRouting | None = None,
    matrix_scalar_tile: bool = False,
    completion_metric_prefix: str | None = None,
    device: Any | None = None,
    prepared_model_transform: Callable[[Path, Path], Path] | None = None,
    masked_contraction_effects: MaskEffectContract | None = None,
    source_observation_effects: IntervalEffectContract | None = None,
    source_scalar_carrier=None,
) -> dict:
    """Build the selected image; see docs/design/zephyr_model_memory.md for sizing rationale."""
    _completion_metric_line(completion_metric_prefix)
    if board is None:
        raise ZephyrModelError("a board from MERLIN_BOARD_CATALOG must be explicitly selected")
    from merlin.runtime.boards.catalog import board as _board_desc

    brd = _board_desc(board, **({"vlen": vlen} if vlen is not None else {}))
    if brd.flow != "zephyr" or brd.zephyr_default_ram_bytes is None or brd.zephyr_link_limit_bytes is None:
        raise ZephyrModelError(f"{brd.name}: selected board lacks a Zephyr port and its memory facts")
    if matrix is not None:
        matrix.provider()  # Refuse unselected/incomplete support before build output or native tools.
    else:
        load_matrix_signatures(Path(work), None)
    if not build_available():
        raise ZephyrModelError("Zephyr build toolchain unavailable (see env in module doc)")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    from merlin.compiler.lower.options.weight_prepack import prepare_build_bundle

    model_dir = prepare_build_bundle(model_dir, work, features)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    # NOTE scalar multicore IS supported (see the lowering call below): the scalar path cannot use the
    # forall-under-the-RVV-schedule route, so it parallelizes at the linalg loop level instead. This is
    # the ONLY way to use a hart that has no vector unit -- a heterogeneous SoC may bring up more cores
    # than it attaches vector units to, and those extra cores are otherwise unreachable.
    if iters < 1:
        raise ZephyrModelError(f"iters must be >= 1, got {iters}")
    # A VLEN the build assumes must match the one it will run on -- see march_with_vlen.
    cflags = (
        march_with_vlen(cflags_override or _cflags(backend), vlen)
        if backend == "rvv"
        else list(cflags_override or _cflags(backend))
    )

    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    ar = gcc.with_name("riscv64-unknown-elf-ar")
    objcopy = gcc.with_name("riscv64-unknown-elf-objcopy")
    clang = toolchain.clang()

    # 1. Normalize with the dispatch_runtime preparation passes, then lower to LLVM/object.
    # Board facts as DATA (runtime.boards): the console options, the vector-state width, the DT RAM
    # label and the DRAM ceiling all come from the descriptor instead of being assumed. `vlen` given
    # explicitly wins over the board's, so a caller can sweep it.
    if vlen is None and brd.vlen is not None:
        vlen = brd.vlen  # build for the board's real vector length by default
    # A vector model may only fan out over harts that HAVE a vector unit. A heterogeneous SoC is normal
    # -- three cores brought up, a vector unit on two of them -- and the difference is invisible in the
    # device tree, which lists identical cpu@N nodes. Fanning out too far does not fail cleanly: the
    # worker on the scalar hart takes an illegal instruction, never reaches the barrier its peers wait
    # on, and the image hangs until the operator times out. Measured on a 3-core tapeout with 2
    # vector cores: the 1-hart images passed and every 3-hart image timed out at 10 minutes.
    # The harts this image may actually run on. A vector image is confined to the vector-capable ones;
    # a scalar image may use every hart, which is the only way to reach a core with no vector unit.
    # Emitted into the OpenMP shim only when it differs from the 0..n-1 default, so an ordinary
    # homogeneous board's image is byte-identical to before.
    _usable = brd.hart_ids_for(backend)
    hart_ids = _usable if _usable != tuple(range(n_harts)) else None
    if backend == "rvv" and n_harts > brd.n_vector_harts:
        raise ZephyrModelError(
            f"{brd.name}: refusing an RVV image over {n_harts} harts -- only {brd.n_vector_harts} of "
            f"its {brd.harts} harts can execute vector code. The extra worker would trap on its first "
            f"vector instruction and deadlock the barrier (a timeout, with no fault printed). Build "
            f"{brd.n_vector_harts} harts or fewer, or use backend='scalar' to use every hart."
        )
    # IR_LOCK prevents concurrent xDSL parsers from producing bogus ParseError on valid IR.
    from merlin.compiler.ir.xdsl_io import IR_LOCK

    prof_table: list[dict] | None = None
    with IR_LOCK:
        prepared, features = prepare_for_lowering(
            model_dir / "model.mlir",
            work,
            int8_compute=int8_compute,
            quant_passes=quant_passes,
            features=features,
            blocking=True,
            harts=n_harts,
            vlen=vlen,
            matrix=matrix,
            device=device,
            prepared_model_transform=prepared_model_transform,
        )
        from merlin.compiler.lower.device.offload import build_arguments, load_sidecar

        if build_arguments(load_sidecar(work), expected_granularity=getattr(device, "granularity", None))["signatures"]:
            raise ZephyrModelError("active device route has no device-object/final-link admission consumer")
        if load_matrix_signatures(work, matrix):
            raise ZephyrModelError(
                "active matrix route has no independently selected final linked-ELF admission consumer"
            )
        # A DEBUG image interleaves a mark between the top-level ops of @forward. Two things come out
        # of it: a per-op cost table at the end of a successful run, and -- the reason it is here at
        # all -- a continuously published "which op are we in", which the heartbeat prints while the
        # model is still running. Without it a stalled image can only say that it stalled; with it, it
        # says where. The table is written next to the image so a returned PROF line resolves to a name.
        if debug:
            from merlin.compiler.lower.model import profile as _op_profile

            try:
                text, prof_table = _op_profile.instrument(Path(prepared).read_text(), structural=True)
                prepared = Path(work) / "model_prof.mlir"
                Path(prepared).write_text(text)
                _op_profile.write_table(prof_table, Path(work) / "op_profile_table.json")
            except _op_profile.OpProfileError as exc:
                # Losing per-op detail must not cost us the rest of the diagnostics.
                print(
                    f"[zephyr_model] op profiling unavailable ({exc}); "
                    "debug image keeps STAGE/ALIVE/MEM but ALIVE cannot name an op"
                )
                prof_table = None
        # For the rvv backend, bake native RVV (fixed-width vector ops on the matmuls) into the
        # IR rather than leaving it to clang's auto-vectorizer — see llvmlower.pipeline.
        # Multicore reaches the two backends by DIFFERENT routes, because they parallelize at
        # different levels. The rvv path layers an explicit outer `scf.forall` UNDER the package's
        # transform schedule (`parallel_harts`), so the inner tiling and the emitted vfmacc/vwmacc are
        # untouched. The scalar path has no such schedule to sit under; it parallelizes at the linalg
        # loop level instead (`parallel=True` -> convert-linalg-to-parallel-loops + convert-scf-to-
        # openmp), which is the same pipeline the K1 big models use. Both emit `__kmpc_*` satisfied by
        # the Zephyr OpenMP shim, so the worker side is identical.
        res = lower_model_file(
            prepared,
            work / "lower",
            targets=(),
            textual=True,
            vectorize=(backend == "rvv"),
            transform_schedule=rvv_schedule,
            features=features,
            parallel=(backend != "rvv" and n_harts > 1),
            parallel_harts=(n_harts if n_harts > 1 and backend == "rvv" else None),
            # The block-preserving split `prepare_for_lowering` above derived and
            # tagged this IR with; None for a package with no per-op block table,
            # which keeps the legacy class-wide split.
            parallel_chunks=parallel_arms(work),
            masked_contraction_effects=masked_contraction_effects,
            source_observation_effects=source_observation_effects,
            source_scalar_carrier=source_scalar_carrier,
        )
    # What this lowering will ask the heap for, read off the IR that is about to be compiled. Measured
    # here rather than estimated later: the file exists for exactly this build, and the number decides the
    # region size below.
    alloc_total, alloc_dynamic = allocation_bytes(res.ll_path)

    # BACKEND-level feature flags, applied to the MODEL OBJECT ONLY (never to the C glue below):
    # some codegen decisions are not expressible in the IR -- the width of the vector register group
    # the auto-vectorizer asks for is a backend query, so no tile size reaches it. `features` here is
    # the set `prepare_for_lowering` resolved, so a derived feature is honoured. Empty (the baseline
    # and every IR-only feature set) leaves the flags untouched, so model.o stays byte-identical.
    from merlin.compiler.optimizations.registry import apply_cflags as _apply_cflags

    model_cflags = _apply_cflags(cflags, frozenset(features or frozenset()))
    _run([clang, "--target=riscv64-unknown-elf", *model_cflags, "-c", res.ll_path, "-o", work / "model.o"])

    # 2. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    # Preparation can append build-time constants to @forward (for example invariant weight
    # quantization). Point C generation at the same authority so its descriptor table cannot drift
    # from the object ABI merely because generated C lives in the `cgen` subdirectory.
    info = c_runtime.generate(model_dir, cgen, inputs_npz, prepared_dir=work)

    # 3. weights.bin -> binary blob object; archive it with model.o.
    _run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"], cwd=cgen)
    weights_size = (cgen / "weights.bin").stat().st_size

    # Build identity, part one: the lowered model object plus the weights blob -- what computes the
    # answer. The app configuration is folded in further down, once it exists, because the compute is
    # only half of what makes two images behave differently. Hashing the model alone was a real defect:
    # the Kodiak vector-state fix changed the Kconfig and the DT overlay and nothing else, so the fixed
    # and the hanging image reported the SAME build_hash, and the one thing the protocol exists to
    # answer -- "which binary produced this log?" -- could not be answered.
    import hashlib as _hashlib

    _h = _hashlib.sha256()
    _h.update((work / "model.o").read_bytes())
    _h.update((cgen / "weights.bin").read_bytes())
    quantization_policy = None
    if int8_compute:
        quantization_policy = json.loads((work / "quantization-policy.json").read_text())
        _h.update((work / "quantization-policy.json").read_bytes())

    # External-weights mode for blobs that would overflow the selected port's link
    # window: keep the low code/arena region compact and place the blob at the
    # board-declared contiguous DRAM address, referenced by an absolute pointer.
    # DECIDE ON THE LINKED FOOTPRINT, NOT THE BLOB SIZE. medany constrains the whole ram0 span
    # (code + weights + activation arena), so a blob that fits the limit on its own can still push
    # the region past the window once the arena is added -- and then linked mode is chosen for an
    # image that cannot work. MEASURED on gemma2_2b_int8_section12: 1462 MiB of weights sat under the
    # 1900 MiB threshold, linked mode was selected, the arena took ram0 to ~3.7 GiB, and the image
    # never reached its first op in 30 min (silent -- no fault, which is what made it expensive to
    # find). The SAME bundle forced external, with a compact low RAM region and a separate weights blob,
    # reaches op 0 in 8 minutes. Comparing the computed region is strictly safer than comparing the
    # blob: it can only move builds from linked to external, and external is the mode that works at
    # every size tested (1034 MiB and 2485 MiB regions both execute).
    from merlin.compiler.ir.mlir_query import activation_peak_bytes
    from merlin.compiler.ir.xdsl_io import IR_LOCK

    peak = None  # measured only on the path that sizes the region from it
    linked_region = None
    if ram_bytes_override is None:
        # Same lock rationale as the sizing parse below: three concurrent build_app calls produced a
        # bogus ParseError on valid IR, which read as a broken build rather than a race.
        with IR_LOCK:
            peak = int(activation_peak_bytes(model_dir / "model.mlir") or 0) or None
        linked_region = ram_for_weights(weights_size, peak, alloc_total, default_ram_bytes=brd.zephyr_default_ram_bytes)
    layout = memory_plan(brd, weights_size, linked_region or 0, ram_bytes_override)
    external = layout.external
    weights_base = layout.weights_base
    if external:
        _run([objcopy, "--rename-section", ".data=.merlin_weights", work / "weights_blob.o"])
    # The region and simulator span differ in external mode: the linked arena only
    # occupies the low region, while the simulator must map the weights as well.
    ram_bytes = layout.spike_span_bytes
    ram_region_bytes = layout.ram_region_bytes

    archive = work / "libmerlinmodel.a"
    archive.unlink(missing_ok=True)
    # The matrix-unit shim, when the rewrite routed anything to one, goes into the SAME archive as the
    # model object -- so it is covered by the --whole-archive the generated CMakeLists already applies and
    # no CMake change is needed. Built from the SIDECAR the rewrite wrote: the symbols the module actually
    # calls are the ones that must be defined, and a set reconstructed here could drift into a link error.
    matrix_objs: list[Path] = []
    matrix_build = None
    matrix_sigs = load_matrix_signatures(work, matrix)
    if matrix_sigs:
        if matrix is None:
            raise ZephyrModelError(
                f"{len(matrix_sigs)} matrix-unit signature(s) were routed but no `matrix=` routing is "
                "available to build them against; the image would not link"
            )
        # A routed contraction is an opaque `call` by the time the parallel transform schedule runs, so
        # that schedule -- which matches `linalg.matmul`/`linalg.batch_matmul` -- cannot split it. On a
        # chip with a matrix unit per core, the ONLY place the other units can be reached from is inside
        # the kernel's own tile loop, so a multi-hart image compiles that loop with OpenMP. A serial
        # object on such a chip is not an error and not a wrong answer: it uses one unit, silently.
        from merlin.compiler.backend.matrix_kernels import build_kwargs

        matrix_build = matrix.provider().build_object(
            matrix_sigs,
            work / "matrix",
            unit=matrix.unit,
            config=matrix.config,
            cc=clang,
            cflags=["--target=riscv64-unknown-elf", *cflags],
            scalar_tile=matrix_scalar_tile,
            parallel_tiles=n_harts > 1,
            **build_kwargs(matrix, matrix_sigs),
        )
        matrix_objs = [matrix_build.object_path]
        print(
            f"[matrix] linked {len(matrix_sigs)} entry point(s) for {matrix.unit} "
            f"({matrix.config}, tile edge {matrix_build.tile_edge}, "
            f"{'SCALAR STAND-IN' if matrix_build.scalar_tile else 'device instructions'}, "
            f"{matrix_build.scratch_bytes} B pack scratch, "
            f"tile loop {
                'parallel over ' + str(n_harts) + ' hart(s)'
                if matrix_build.parallel_tiles
                else 'serial — ONE unit will be used'
            })"
        )
    _run([ar, "rcs", archive, work / "model.o", work / "weights_blob.o", *matrix_objs])

    # 4. emit the Zephyr application tree.
    app = work / "app"
    (app / "src").mkdir(parents=True, exist_ok=True)
    (app / "boards").mkdir(parents=True, exist_ok=True)
    # A multicore image needs every hart it fans out to, plus the master's.
    cpus = max(cpus, n_harts, rvv_hart + 1)
    # Console facts, when the board has a console of its own to bring up. Derived here (not in
    # _prj_conf) because the DT overlay below needs the same values, and deriving twice invites the
    # config and the device tree to disagree about which UART this is.
    from merlin.runtime.boards.catalog import CONSOLE_UART
    from merlin.runtime.boards.sdk_facts import DEFAULT_BAUD as _DEFAULT_BAUD

    console_facts = None
    if brd.console == CONSOLE_UART:
        from merlin.runtime.boards.sdk_facts import derive_uart_console

        if not sdk_dir or not brd.sdk_chip:
            raise ZephyrModelError(
                f"{brd.name} declares a UART console, so it needs sdk_dir + the descriptor's "
                "sdk_chip: the UART address and its clock rates are derived from the target SDK's "
                "own headers, never hardcoded or defaulted"
            )
        console_facts = derive_uart_console(sdk_dir, brd.sdk_chip)
    prj_conf_text = project_config(
        cpus, backend, brd, console_facts, debug=debug, tree=TreeConfiguration(_zephyr_base())
    )
    (app / "prj.conf").write_text(prj_conf_text)
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    weights_section_ld = None
    if external:
        # snippet: divert the renamed blob section into the WEIGHTS DT memory-region.
        weights_section_ld = app / "weights_sections.ld"
        weights_section_ld.write_text(
            ".merlin_weights : {\n\tKEEP(*(.merlin_weights))\n\tKEEP(*(.merlin_weights.*))\n} GROUP_LINK_IN(WEIGHTS)\n"
        )

    def _cmake(bh: str) -> str:
        return _cmakelists(
            archive,
            rt,
            abi,
            cgen,
            weights_section_ld,
            omp=(n_harts > 1),
            n_harts=n_harts,
            hart_ids=hart_ids,
            backend=backend,
            debug=debug,
            build_hash=bh,
            console_facts=console_facts,
            chip_freq_hz=brd.chip_freq_hz,
            op_profile=prof_table is not None,
        )

    # Written below, once the identity it embeds is known.
    cmakelists_text = _cmake("")
    # Board overlay. External mode has a low code/arena region and a separate
    # WEIGHTS region; their sizes and origin derive from the selected catalog.
    overlay = ""
    if console_facts is not None:
        # Point `chosen` at the chip's UART and state its address from the DERIVED fact rather than
        # trusting the board DT to already agree: the config above selects the driver, but `chosen`
        # is what the console actually binds to, and a board whose DT says HTIF would keep using it.
        overlay += (
            f"/ {{\n\tchosen {{\n"
            f"\t\tzephyr,console = &{brd.uart_label};\n"
            f"\t\tzephyr,shell-uart = &{brd.uart_label};\n\t}};\n}};\n\n"
            f'&{brd.uart_label} {{\n\tstatus = "okay";\n'
            f"\treg = <{hex(console_facts.uart_base)} 0x1000>;\n"
            f"\tcurrent-speed = <{_DEFAULT_BAUD}>;\n}};\n\n"
        )
    if external:
        wsz = layout.weights_region_bytes
        # The WEIGHTS region lives under a 2-cell (#address/size-cells=2) container so its
        # reg can express a 64-bit base+size — letting the blob exceed 4 GB (tiny, pi05).
        # The chipyard root is 1-cell, so we add a child bus with 2/2 cells. base & size are
        # emitted as <hi lo> pairs.
        b_hi, b_lo = (weights_base >> 32) & 0xFFFFFFFF, weights_base & 0xFFFFFFFF
        s_hi, s_lo = (wsz >> 32) & 0xFFFFFFFF, wsz & 0xFFFFFFFF
        overlay += (
            f"&{brd.ram_label} {{\n\treg = <{hex(brd.dram_base)} {hex(ram_region_bytes)}>;\n}};\n\n"
            f"/ {{\n\tweights_bus {{\n"
            f"\t\t#address-cells = <2>;\n\t\t#size-cells = <2>;\n\t\tranges;\n"
            f"\t\tweights0: memory@{weights_base:x} {{\n"
            f'\t\t\tcompatible = "zephyr,memory-region", "mmio-sram";\n'
            f"\t\t\treg = <{hex(b_hi)} {hex(b_lo)} {hex(s_hi)} {hex(s_lo)}>;\n"
            f'\t\t\tzephyr,memory-region = "WEIGHTS";\n\t\t}};\n\t}};\n}};\n'
        )

    def _region_overlay(size: int) -> str:
        """The ram0 override for a region of ``size``, or "" at the DTS default.

        A function rather than a literal because the region may have to be CORRECTED after the link: the
        arena is the leftover after the image, and the image's size is not known until it exists.
        """
        if size <= brd.zephyr_default_ram_bytes:
            return ""
        # Never past what the chip HAS: a region larger than physical DRAM is a boot that dies before
        # main() with no console output at all.
        if size > brd.dram_bytes:
            raise ZephyrModelError(
                f"{brd.name}: this model needs a {size / 2**20:.0f} MB region but the board has "
                f"{brd.dram_bytes / 2**20:.0f} MB of DRAM. Shrink the model (a smaller capture), or "
                f"correct the board descriptor if the real DRAM is larger."
            )
        return f"&{brd.ram_label} {{\n\treg = <{hex(brd.dram_base)} {hex(size)}>;\n}};\n"

    # The port's CPU-node count is target-owned catalog data, never inferred from
    # its name. Unknown ports do not receive an unrelated overlay.
    cpu_overlay = _cpu_disable_overlay(cpus, brd.dt_cpu_nodes) if brd.dt_cpu_nodes is not None else ""
    overlay_base = overlay

    def _overlay_for(size: int) -> str:
        """The whole overlay for a region of ``size``. Assembled from parts so the region can change."""
        text = overlay_base + ("" if external else _region_overlay(size))
        return text + (("\n" if text else "") + cpu_overlay if cpu_overlay else "")

    overlay = _overlay_for(ram_bytes)

    # Build identity, part two: the app configuration. Two images with identical compute but a
    # different Kconfig, device tree or link are different images -- they take different code paths on
    # the silicon, and one can hang where the other passes. Folding these in is what makes
    # `METRIC build_hash` answer "which binary produced this log?" rather than "which model did it
    # compute?". `brd.build_board` is in there too, since the same app text against a different board
    # port is a different image again.
    def _harness(bh: str) -> str:
        return _main_c(
            rvv_hart,
            weights_base=weights_base,
            n_harts=n_harts,
            iters=iters,
            warmup=warmup,
            omp_threads=omp_threads,
            build_hash=bh,
            console=brd.console,
            debug=debug,
            dram_base=brd.dram_base,
            region_bytes=ram_region_bytes,
            completion_metric_prefix=completion_metric_prefix,
        )

    _h.update(prj_conf_text.encode())
    _h.update(cmakelists_text.encode())
    _h.update(brd.build_board.encode())
    # The OVERLAY and the harness are folded in per build pass rather than here, because a corrected
    # region changes both -- and an image linked for a different region is a different image, which is
    # exactly what this hash exists to distinguish.
    # 5. configure + build. Possibly TWICE, because the region and the image are circular: the arena is
    #    whatever is left of the region after the image, and the image's size is not known until it has
    #    been linked. The first pass sizes the region from the weights plus the measured allocation
    #    demand; if the linked image turns out to be larger than the weights by more than the slack --
    #    on a 1.5 GB model the code, the .bss and the per-hart stacks came to 150 MB, and the arena
    #    landed 17 MB short of what the model asks for -- the region is corrected to the MEASURED image
    #    plus that demand and the app is rebuilt. Bounded at one correction: the image does not change
    #    size when the region does (only the linker script's region length does), so one pass settles it.
    build_dir = work / "build"
    env = _tool_env()
    elf = build_dir / "zephyr" / "zephyr.elf"

    def _emit_and_build() -> str:
        h = _hashlib.sha256(_h.digest())
        h.update(overlay.encode())
        # The harness with an EMPTY identity, so the hash is not an input to itself. This is what makes
        # iters/warmup/rvv_hart/console and the linked region part of the identity too.
        h.update(_harness("").encode())
        bh = h.hexdigest()[:16]
        (app / "prj.conf").write_text(prj_conf_text)
        (app / "src" / "main.c").write_text(_harness(bh))
        (app / "CMakeLists.txt").write_text(_cmake(bh))
        # Pass the overlay EXPLICITLY. Zephyr's automatic discovery matches `boards/<BOARD>.overlay`
        # against the board being built, so writing it under this descriptor's name meant that for any
        # board built through a generic port (build_board != name) the overlay was silently IGNORED --
        # taking the DRAM-size override and the CPU-disable overlay with it, with no warning and a
        # successful build. Naming one file and pointing cmake at it cannot miss that way.
        extra = []
        if overlay:
            overlay_file = app / "merlin.overlay"
            overlay_file.write_text(overlay)
            extra.append(f"-DEXTRA_DTC_OVERLAY_FILE={overlay_file}")
        _run(
            [
                build_tool("cmake") or "cmake",
                "-B",
                build_dir,
                "-G",
                "Ninja",
                f"-DBOARD={brd.build_board}",
                *extra,
                "-S",
                app,
            ],
            env=env,
        )
        _run([build_tool("ninja") or "ninja", "-C", build_dir], env=env)
        if not elf.is_file():
            raise ZephyrModelError(f"build produced no elf at {elf}")
        return bh

    build_hash = _emit_and_build()
    demand = alloc_total + 128 * 1024 * 1024
    if external:
        from merlin.runtime.boards.image_audit import read_elf as _read_elf

        low_demand = (alloc_total or peak or 0) + 128 * 1024 * 1024
        require_external_arena(brd, ram_region_bytes, _read_elf(elf)[1], low_demand)
    if not external and alloc_total:
        from merlin.runtime.boards.image_audit import read_elf as _read_elf

        image = sum(sg.memsz for sg in _read_elf(elf)[1])
        if ram_bytes - image < demand:
            align = 16 * 1024 * 1024
            grown = ((image + demand + align - 1) // align) * align
            print(
                f"[zephyr_model] region {ram_bytes / 2**20:.0f} MB leaves "
                f"{(ram_bytes - image) / 2**20:.0f} MB of arena for a model that asks for "
                f"{alloc_total / 2**20:.0f} MB; the linked image measures {image / 2**20:.0f} MB. "
                f"Relinking for {grown / 2**20:.0f} MB.",
                flush=True,
            )
            ram_bytes = grown
            ram_region_bytes = grown
            overlay = _overlay_for(ram_bytes)
            build_hash = _emit_and_build()
    out = {
        "elf": elf,
        "app_dir": app,
        "build_dir": build_dir,
        "board": brd.name,
        "backend": backend,
        "ram_bytes": ram_bytes,
        "ram_region_bytes": ram_region_bytes,
        "weights_base": weights_base,
        "build_hash": build_hash,
        "quantization_policy": quantization_policy,
        **info,
    }
    # What the matrix-unit shim in this image actually is, for a caller that has to state it: the tile
    # edge and alignment it was built for, whether the tile loop can reach more than one unit, and WHICH
    # REVISION of the unit's sources the encodings were derived from. The last one is the whole point --
    # an image carrying instructions derived from the wrong revision links and runs.
    if matrix_build is not None:
        out["matrix_build"] = matrix_build.to_dict()
        out["matrix_routing"] = matrix.identity()
    # The memory arithmetic, so a caller can verify the region it got rather than trusting the formula
    # that produced it. `allocation_bytes_total` is the requirement under an allocator that never
    # reclaims; `allocation_dynamic_calls` says how much of the answer is unknown at build time.
    out["allocation_bytes_total"] = alloc_total
    out["allocation_dynamic_calls"] = alloc_dynamic
    out["activation_peak_bytes"] = peak
    # The per-op table has to travel WITH the image. A debug run emits a thousand `PROF <id> ...` lines
    # and an `ALIVE ... op=<id>` naming where it stopped, and every one of those ids is meaningless
    # without the mapping from id to op name, family and shape. Shipping the trace without the table is
    # shipping a log nobody can read.
    if prof_table is not None:
        out["op_profile_table"] = Path(work) / "op_profile_table.json"
        out["op_profile_ops"] = len(prof_table)
    return out
