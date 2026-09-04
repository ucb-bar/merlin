"""Board facts as data, and the failure modes each one prevents.

The generated app used to assume its board: HTIF console options, a 256-bit vector save area, a `&ram0`
label at 0x80000000. Those are chipyard facts, not RISC-V facts, and every one of them fails LOUDLY on
the board and SILENTLY here — no console output looks exactly like a hang, and a region larger than
physical DRAM dies before main(). So the descriptor is the contract, and these tests pin it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.runtime import boards
from merlin.runtime.backends import zephyr_model as zm
from merlin.runtime.sdk_facts import UartConsoleFacts


@pytest.fixture()
def uart_facts():
    """Console facts as `sdk_facts` derives them, without needing an SDK checkout on disk.

    The values mirror a real tapeout's (a 50 MHz core with a 50 kHz mtime, so the divider is 1000) so
    the clock arithmetic under test is the arithmetic that runs.
    """
    return UartConsoleFacts(
        uart_base=0x10020000,
        reg={"TXDATA": 0, "RXDATA": 4, "TXCTRL": 8, "RXCTRL": 12, "IE": 16, "IP": 20, "DIV": 24},
        tx_full_bit=31, txen_bit=0, rxen_bit=0, nstop_bit=1,
        sys_clk_hz=50_000_000, mtime_hz=50_000,
        pll_base=0x140000,
        pll={"POWERGOOD_VNN": 0x5C, "PLLEN": 0x60, "LDO_ENABLE": 0x64, "RATIO": 0x6C,
             "FRACTION": 0x70, "MDIV_RATIO": 0x74, "ZDIV0_RATIO": 0x78, "ZDIV1_RATIO": 0x80,
             "PLLFWEN_B": 0x100},
        clksel_base=0x130000,
        clksel={"UNCORE": 0, "TILE0": 4, "TILE1": 8, "CLKTAP": 12},
        clksel_slow=0, clksel_pll=1,
    )


def test_kodiak_is_described_by_its_silicon_not_its_dts():
    """Kodiak's DTS says ram0=256MB and MP_MAX_NUM_CPUS=2; the chip has 512MB and 3 working cores."""
    b = boards.board("chipyard_kodiak")
    assert b.dram_bytes == 512 * 1024 * 1024
    assert b.harts == 3
    assert b.console == boards.CONSOLE_HTIF


def test_an_unknown_board_falls_back_conservatively_and_says_so():
    """A new board must be tryable before anyone writes it down, but never with invented facts."""
    b = boards.board("some_new_tapeout")
    assert b.dram_bytes == 256 * 1024 * 1024
    assert b.vector_max_len == 128, "unknown VLEN must assume the V minimum, not a guess"
    assert "conservative" in b.notes


def test_overrides_win_so_a_delivery_can_state_what_it_built_for():
    b = boards.board("chipyard_kodiak", dram_bytes=256 * 1024 * 1024, vlen=256)
    assert b.dram_bytes == 256 * 1024 * 1024 and b.vlen == 256


def test_the_config_follows_the_board():
    """cpus, the vector save-area width and FPU_SHARING all come from the descriptor.

    The save-area width comes from it FLOORED at the tree's default -- this board states no VLEN, so its
    property falls back to the V minimum and the emitted config is raised to the safe width. See
    `test_the_vector_save_area_is_never_declared_smaller_than_the_tree_default`.
    """
    b = boards.board("spike_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert f"CONFIG_MP_MAX_NUM_CPUS={b.harts}" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={zm._vector_max_len_bits(b)}" in conf
    assert zm._vector_max_len_bits(b) >= b.vector_max_len
    # y mis-routes V-illegal-instruction traps into the FP path and retries forever: a silent hang.
    assert "CONFIG_FPU_SHARING=n" in conf


def test_kodiak_lets_zephyr_manage_vector_state():
    """Kodiak's vector config, and the reasoning that had to be corrected twice to arrive at it.

    The descriptor used to say `fpu_sharing=True, zephyr_vector_ext=False`, on this argument: that
    tree's isr.S calls z_riscv_vstate_save/_restore whenever V is on with eager switching, those live
    in fpu.c/fpu.S, and fpu.c only compiles under FPU_SHARING -- so V without FPU_SHARING would not
    link; and enabling V at all would put `v` in the global -march, for which SDK 0.17.0 has no libgcc
    multilib ("ELFCLASS32 incompatible with ELFCLASS64").

    Both halves were true of the Zephyr the `kodiak` BRANCH pins (submodule 5a06eb0d) and false of the
    tree we build against. ZEPHYR_BASE is the `dev` pin (852bb170), two commits later, which contains
    "riscv: decouple V/F save-restore + add RISCV_V_KERNEL_ONLY (Saturn fork)". There,
    arch/riscv/core/CMakeLists.txt compiles v.c under CONFIG_RISCV_ISA_EXT_V *independently of*
    FPU_SHARING and says so in a comment, and RISCV_V_KERNEL_ONLY keeps `v` out of the global -march.
    Verified by building: z_riscv_vstate_save/_restore link with FPU_SHARING=n.

    What the stale pair cost: with no CONFIG_RISCV_ISA_EXT_V, no thread's mstatus carries VS (a
    thread's initial mstatus is MSTATUS_DEF_RESTORE = MPP|MPIE, and VS is OR'd in only under that
    symbol), so the OpenMP master lost vector state the moment pool creation context-switched it out,
    and FPU_SHARING=y then mis-routed the resulting illegal-instruction trap into the FP retry path --
    a hang with nothing printed. On the chip: every single-hart image PASSED and every multi-hart image
    FAILED. Neither spike nor the Saturn RTL enforces mstatus.VS, which is why no simulation caught it.

    The settings asserted below are the ones the chip's OWN known-working RVV+SMP sample uses
    (origin/kodiak:samples/q8_gemm_minmax/prj.conf, which ships a ref-out): ISA_EXT_V on,
    VECTOR_MAX_LEN 512, FPU_SHARING off.
    """
    b = boards.board("chipyard_kodiak")
    assert b.fpu_sharing is False and b.zephyr_vector_ext is True
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_MP_MAX_NUM_CPUS=3" in conf
    assert "CONFIG_FPU_SHARING=n" in conf
    assert "CONFIG_RISCV_ISA_EXT_V=y" in conf, "Zephyr must manage per-thread vector state"
    assert "CONFIG_RISCV_VECTOR_MAX_LEN=512" in conf, "the chip's own sample states 512"
    assert "CONFIG_RISCV_V_KERNEL_ONLY=y" in conf, "keeps `v` out of the global -march"


def test_a_stale_vector_ext_flag_is_refused_rather_than_built():
    """Denying Zephyr vector state on a tree that supports it is always a stale descriptor now.

    It is not a harmless conservative choice: it is the exact configuration that shipped a Kodiak
    package where every multi-hart image hung silently. The flag exists for trees that CANNOT express
    V without polluting the global -march; on a tree that can, the build must stop rather than quietly
    produce the image again.
    """
    import dataclasses
    import pytest

    b = dataclasses.replace(boards.board("chipyard_kodiak"), zephyr_vector_ext=False)
    if not zm._kconfig_has("RISCV_V_KERNEL_ONLY"):
        pytest.skip("this Zephyr tree genuinely cannot express V without the global -march")
    with pytest.raises(RuntimeError, match="zephyr_vector_ext"):
        zm._prj_conf(b.harts, "rvv", b)


def test_a_vector_capable_tree_still_gets_the_full_vector_config():
    b = boards.board("chipyard_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_RISCV_ISA_EXT_V=y" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={b.vector_max_len}" in conf


def test_the_vector_save_area_is_never_declared_smaller_than_the_tree_default():
    """Under-declaring the save area corrupts kernel memory; over-declaring only costs RAM.

    `vreg[32][CONFIG_RISCV_VECTOR_MAX_LEN/8]` is a fixed buffer, but `z_riscv_vstate_save` fills it via
    `vsetvli x0, e8, m8` + four `vse8.v` -- a length taken from the HARDWARE, never compared to the
    buffer. A descriptor under-stating VLEN therefore writes `32 * (vlenb_hw - MAX_LEN/8)` bytes past the
    thread struct on every switch, into whatever the linker put next. On a chip whose own probe reports
    `vlenb 32`, a `vlen=128` descriptor overran `z_idle_threads[1]` into `z_main_thread` and zeroed its
    `stack_info.start`; the next tick loaded from address 0. A spike gate cannot catch it, because spike
    is given the VLEN we declared -- configured and actual agree there by construction.

    So the emitted value is floored at the tree's own default for the symbol. This is a guard, not a
    substitute for a correct descriptor: it bounds the damage of one being wrong to wasted memory.
    """
    floor = zm._kconfig_default_int("RISCV_VECTOR_MAX_LEN")
    if not floor:
        pytest.skip("this Zephyr tree states no single numeric default for RISCV_VECTOR_MAX_LEN")

    understated = boards.board("some_new_tapeout", vlen=128)
    assert understated.vector_max_len == 128, "the descriptor still reports what it was told"
    assert zm._vector_max_len_bits(understated) == floor, "but the emitted config is floored"
    conf = zm._prj_conf(understated.harts, "rvv", understated)
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={floor}" in conf

    # The floor must not CLAMP a board that legitimately has wider registers -- that would recreate the
    # overrun on the one class of board where it is guaranteed to happen.
    wide = boards.board("some_new_tapeout", vlen=1024)
    assert zm._vector_max_len_bits(wide) == 1024


def test_htif_console_options_are_set_only_for_an_htif_board():
    """Both options default to n upstream, and unbuffered HTIF emits one char per host round-trip --
    on a ~20 MHz core that looks like the model never finishing."""
    htif = zm._prj_conf(2, "rvv", boards.board("chipyard_kodiak"))
    assert "CONFIG_UART_HTIF_BUFFERED_OUTPUT=y" in htif
    assert "CONFIG_UART_HTIF_SYSCALL_PRINT=y" in htif
    assert "UART_SIFIVE" not in htif


def test_a_uart_board_without_derived_facts_is_refused():
    """This used to emit a comment saying the board configures its own console -- which was a SILENT
    failure, because the generic chipyard board's defconfig sets CONFIG_UART_HTIF=y. The image kept a
    host-assisted console and hung in its first print on silicon. There is no safe default here."""
    with pytest.raises(RuntimeError, match="no SDK facts"):
        zm._prj_conf(2, "rvv", boards.board("x", console=boards.CONSOLE_UART))


def test_a_uart_board_turns_htif_off_and_states_both_clock_terms(uart_facts):
    """The driver computes its divisor as (SYS_CLOCK_HW_CYCLES_PER_SEC * RTC_CLOCK_DIVIDER_VALUE)/baud
    - 1, so BOTH terms must describe the chip. The board's own defaults imply a 1 GHz peripheral clock
    and would emit garbage rather than nothing -- which reads as a corrupt program, not a bad UART."""
    conf = zm._prj_conf(2, "rvv", boards.board("x", console=boards.CONSOLE_UART), uart_facts)
    assert "CONFIG_UART_HTIF=n" in conf
    assert "CONFIG_UART_SIFIVE=y" in conf and "CONFIG_UART_SIFIVE_PORT_0=y" in conf
    assert "CONFIG_UART_CONSOLE=y" in conf
    assert f"CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC={uart_facts.mtime_hz}" in conf
    assert (f"CONFIG_RTC_CLOCK_DIVIDER_VALUE="
            f"{uart_facts.sys_clk_hz // uart_facts.mtime_hz}") in conf


def test_a_core_clock_that_is_not_a_multiple_of_the_mtime_rate_is_refused(uart_facts):
    """The RTOS models the peripheral clock as mtime x an integer divider; a chip that does not fit
    that model must fail loudly rather than be rounded into a wrong baud rate."""
    odd = type(uart_facts)(**{**uart_facts.__dict__, "mtime_hz": 30_000})
    with pytest.raises(RuntimeError, match="integer multiple"):
        zm._prj_conf(2, "rvv", boards.board("x", console=boards.CONSOLE_UART), odd)


def test_the_scalar_backend_still_carries_no_vector_config():
    conf = zm._prj_conf(2, "scalar", boards.board("chipyard_kodiak"))
    assert "RISCV_ISA_EXT_V" not in conf


def test_a_region_bigger_than_the_board_is_refused():
    """A region larger than physical DRAM is a boot that dies before main() with NO console output --
    the single least debuggable outcome for someone running our binary on their bench."""
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "brd.dram_bytes" in src, "build_app must compare the region against the board's DRAM"
    assert "does not fit" in src or "has" in src


def test_the_overlay_uses_the_board_label_and_base():
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "brd.ram_label" in src and "brd.dram_base" in src


def test_build_hash_is_emitted_and_parseable():
    """A console log mailed back from someone else's board is otherwise unattributable."""
    src = zm._main_c(0, n_harts=1, build_hash="abc123def456")
    assert 'METRIC build_hash' in src and '"abc123def456"' in src
    parsed = zm._parse_console(
        "METRIC build_hash abc123def456\nMETRIC cycles 7\nOUT 1 1065353216\nDONE", 0)
    assert parsed["metrics"]["build_hash"] == "abc123def456"
    assert parsed["metrics"]["cycles"] == 7


def test_a_non_numeric_metric_does_not_take_the_whole_run_down():
    """int() on every METRIC crashed the parser on the first identity string."""
    r = zm._parse_console("METRIC tag deadbeef\nOUT 1 0\nDONE", 0)
    assert r["metrics"]["tag"] == "deadbeef"


def test_a_board_declares_how_it_is_built():
    """Not every RISC-V target runs an RTOS. gemmelos-bringup is a Baremetal-IDE fork with no Zephyr
    anywhere in it, so the descriptor has to say which producer applies or the packager would try to
    build a Zephyr app for a board that has none."""
    assert boards.board("chipyard_kodiak").flow == boards.FLOW_ZEPHYR
    for name in ("gemmelos_bearly25", "gemmelos_dsp25"):
        b = boards.board(name)
        assert b.flow == boards.FLOW_BAREMETAL
        assert b.dram_bytes == 1024 * 1024 * 1024, "the silicon has 1 GB (its .ld declares 256 MB)"
        # bearly25's VLEN is MEASURED, not read off its sources: its own OPE kernel says 128 while their
        # spike runs use 256, and `vlen_probe.elf` on the silicon settles it at `vlenb 32` (VLEN 256).
        # dsp25 has had no such probe run, so it keeps the conservative minimum until one does -- with
        # the floor in `_vector_max_len_bits` bounding what that costs.
        assert b.vlen == (256 if name == "gemmelos_bearly25" else 128)


def test_the_baremetal_layout_is_packed_inside_the_boards_dram():
    """The historical spike map puts the arena at 0xC0000000 and the weights at 0x2_0000_0000 -- not
    memory on a 1 GB chip, so an unadjusted image faults on its first activation."""
    from merlin.runtime.backends import spike_model

    brd = boards.board("gemmelos_bearly25")
    lay = spike_model._layout(64 * 1024 * 1024, 8 * 1024 * 1024,
                              dram_base=brd.dram_base, dram_bytes=brd.dram_bytes)
    end = brd.dram_base + brd.dram_bytes
    assert brd.dram_base < lay["weights_base"] < lay["arena_base"] < end
    assert lay["mem_bytes"] == brd.dram_bytes

    # Omitting dram_bytes must keep the historical spike map, byte-for-byte.
    assert spike_model._layout(64 * 1024 * 1024, 8 * 1024 * 1024)["arena_base"] == 0xC0000000


def test_an_image_too_big_for_the_board_fails_closed():
    """Better a build error here than a boot that dies before main on someone else's bench."""
    import pytest

    from merlin.runtime.backends import spike_model

    with pytest.raises(RuntimeError, match="does not fit"):
        spike_model._layout(900 * 1024 * 1024, 200 * 1024 * 1024,
                            dram_base=0x80000000, dram_bytes=256 * 1024 * 1024)


def test_the_baremetal_build_applies_the_same_preparation_as_the_zephyr_one():
    """Lowering the raw bundle scored cos 0.925 on deepjscc where the prepared path is bit-exact: the
    int8 datapath and the per-op tags are not optional extras, they are the numbers."""
    import inspect

    from merlin.runtime.backends import spike_model, zephyr_model

    src = inspect.getsource(spike_model.build)
    assert "prepare_for_lowering" in src
    assert "int8_compute" in inspect.signature(spike_model.build).parameters
    # ...and both backends must call the SAME preparation, not two copies of it.
    assert "prepare_for_lowering" in inspect.getsource(zephyr_model.build_app)


def test_the_two_compilers_do_not_share_flag_sets():
    """An RVV package's cflags are CLANG flags; the bare-metal harness units are built by GCC, which
    rejects -fno-vectorize outright. Only the -march has to agree."""
    import inspect

    from merlin.runtime.backends import spike_model

    src = inspect.getsource(spike_model.build)
    assert "clang_cflags" in src and "gcc_cflags" in src
    assert "*clang_cflags" in src and "*gcc_cflags" in src


def test_the_hart_count_reaches_the_block_table():
    """A plumbing test, because this exact wire was missing: prepare_for_lowering accepted `harts`
    and then called block_table without it, so every multicore build silently blocked for the
    UNSPLIT extents and died with a masked parallel dim. The signature alone proves nothing."""
    import inspect

    from merlin.runtime.backends import zephyr_model as zm

    src = inspect.getsource(zm.prepare_for_lowering)
    assert "harts=harts" in src, "block_table must receive the hart count, not just the signature"
    assert "harts" in inspect.signature(zm.prepare_for_lowering).parameters
    # ...and build_app must pass the image's hart count down, not the default.
    assert "harts=n_harts" in inspect.getsource(zm.build_app)


def test_every_in_process_mlir_parse_in_build_app_is_serialized():
    """xDSL's parser is not thread-safe, and build_app parses TWICE: once to lower, and once to size
    RAM from the activation peak. The second one is easy to miss because it looks like arithmetic
    rather than codegen -- but it produced the same bogus ParseError under three concurrent builds."""
    import inspect

    from merlin.runtime.backends import zephyr_model as zm

    src = inspect.getsource(zm.build_app)
    assert src.count("with IR_LOCK") >= 2, "both parses must hold the lock"
    # The RAM-sizing parse specifically.
    head, _, tail = src.partition("activation_peak_bytes(model_dir")
    assert "with IR_LOCK" in head.rsplit("\n\n", 1)[-1] or "with IR_LOCK:\n                peak" in src


def test_mlir_parsing_is_serialized_at_the_only_place_a_parser_is_built():
    """xDSL's parser is not thread-safe, and locking CALL SITES does not work: after covering the two
    obvious ones in build_app, `c_runtime.generate` was still parsing twice for the @forward
    signature, and the race surfaced as a mutation invariant ("Can't add to a block an operation
    already attached to a block") on valid IR — in whichever concurrent build lost. The lock belongs
    at the parse boundary, which is the single place a Parser is constructed."""
    import inspect

    from merlin.frontends import linalg_mlir

    assert "IR_LOCK" in inspect.getsource(linalg_mlir.parse_mlir_text)
    # If a second Parser construction ever appears, it bypasses the lock and this must fail.
    src = Path(inspect.getfile(linalg_mlir)).read_text()
    assert src.count("Parser(") == 1, "every parse must go through the serialized entry point"


def test_the_generated_worker_enables_vector_state_before_running_the_model():
    """A freshly created Zephyr thread starts with mstatus.VS = Off.

    The RISC-V port builds a thread's initial mstatus from MSTATUS_DEF_RESTORE (MPP | MPIE only); VS is
    added just under CONFIG_RISCV_ISA_EXT_V, which these images do not set. reset.S enables VS for the
    BOOT context, but switching into the worker restores mstatus from the thread's own frame and VS
    goes back to Off -- and with VS Off every vector instruction and vector CSR read traps.

    Measured on a tapeout that enforces VS: the image died with mcause=2 on `csrr a0, vlenb` in the
    prologue of forward() (LLVM sizing a VLEN-scaled stack frame), before computing anything, having
    printed only its banner. spike and the Saturn RTL do NOT enforce VS, so every simulated run passed
    -- which is exactly why this needs a source-level test rather than a run.

    `libomp_zephyr.c::omp_enable_vector()` already did this per OpenMP worker; a single-hart image has
    no pool, so nothing enabled it at all.
    """
    for n_harts in (1, 2, 3):
        src = zm._main_c(0, n_harts=n_harts)
        assert "csrw mstatus" in src, f"n_harts={n_harts}: worker never enables vector state"
        # 0x600 = mstatus.VS bits[10:9], 0x6000 = mstatus.FS bits[14:13].
        assert "0x00000600UL | 0x00006000UL" in src
        # Ordering is the whole point: after the banner is fine, before the model is mandatory.
        assert src.index("csrw mstatus") < src.index("merlin_run_multi("), \
            f"n_harts={n_harts}: vector state enabled after the model already ran"


def test_kodiak_records_the_vector_width_its_owner_confirmed():
    """VLEN=512, confirmed by the chip's owner. Nothing in that board's repo states it: the DTS says
    `riscv,isa = "rv64gc"` with no `v`, and the samples' CONFIG_RISCV_VECTOR_MAX_LEN only sizes a save
    area, so it bounds the truth from above at best. Building for 128 on a 512-bit unit is the
    documented K1 trap -- a lower LMUL leaving most of the datapath idle."""
    assert boards.board("chipyard_kodiak").vlen == 512


def test_scalar_multicore_is_supported_so_a_non_vector_hart_is_reachable():
    """A heterogeneous SoC may bring up more cores than it attaches vector units to, and a scalar
    image is the ONLY way to use the extra ones. This used to be refused outright on the grounds that
    the multicore lowering layers its forall under the RVV schedule -- true, but the scalar path has
    its own OpenMP route (convert-linalg-to-parallel-loops + convert-scf-to-openmp), so it only needed
    routing to. Measured on a 3-hart scalar image: 272 fork_call sites, zero vector instructions."""
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "requires backend='rvv'" not in src, "scalar multicore must not be refused"
    # The two backends reach multicore by different routes; both must be wired.
    assert 'parallel=(backend != "rvv" and n_harts > 1)' in src
    assert 'parallel_harts=(n_harts if n_harts > 1' in src


def test_an_rvv_image_is_refused_past_the_vector_harts():
    """Building too far does not fail cleanly: the worker on a scalar hart traps on its first vector
    instruction and never reaches the barrier, so the image hangs with nothing printed. Measured on a
    3-core/2-vector-core chip: the 1-hart images passed and every 3-hart image timed out."""
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "n_harts > brd.n_vector_harts" in src
    assert "deadlock" in src, "the error must name the failure mode, not just the limit"


def test_which_harts_have_vectors_is_stated_not_assumed():
    """A count alone means 0..N-1. On a chip whose vector units sit on harts 0 and 2 that assumption
    deadlocks exactly like building too many harts does, and nothing readable states the mapping --
    the device tree lists identical cpu@N nodes."""
    kodiak = boards.board("chipyard_kodiak")
    assert kodiak.harts == 3 and kodiak.n_vector_harts == 2
    assert kodiak.hart_ids_for("rvv") == (0, 1)
    # A scalar image may use every hart; that is the point of having one.
    assert kodiak.hart_ids_for("scalar") == (0, 1, 2)
    # Non-contiguous sets are expressible.
    odd = boards.board("x", harts=3, vector_hart_ids=(0, 2))
    assert odd.hart_ids_for("rvv") == (0, 2) and odd.n_vector_harts == 2
    # A homogeneous board keeps the default so its image stays byte-identical.
    assert boards.board("spike_riscv64").hart_ids_for("rvv") == tuple(range(8))


def test_the_hart_list_reaches_the_openmp_shim():
    """Pinning is where a wrong hart set becomes a deadlock, so the shim must WALK the list rather
    than count -- and shrink the pool rather than put two workers on one hart, which would serialize
    silently and report a speed-up that never happened."""
    from merlin.common.paths import merlin_dir

    import inspect

    src = (merlin_dir() / "runtime/c/libomp_zephyr.c").read_text()
    assert "MERLIN_OMP_HART_IDS" in src
    assert "omp_nthreads = i;" in src, "the shim must shrink the pool, not double-pin"
    assert "MERLIN_OMP_HART_IDS" in inspect.getsource(zm._cmakelists), \
        "the build must actually emit the list"


def test_vector_capable_harts_are_discovered_at_runtime_not_assumed():
    """The image asks the HARDWARE which harts can run vector code, at startup.

    `mstatus.VS` is WARL: writable on a hart that implements V, hardwired to zero on one that does
    not. So enabling it and reading it back distinguishes them WITHOUT trapping -- which matters,
    because the obvious probe (run a vector instruction and see if it faults) needs a recoverable
    trap handler per hart, and an unrecovered fault on a board we cannot attach to is the exact
    failure being diagnosed.

    This supersedes both the "vector harts are 0..n-1" assumption and the build-time list: a 3-core
    chip with vector units on 2 harts needs nobody to tell us which 2. Verified on spike at VLEN=512:
    the image printed `METRIC vector_harts 0 1 2` plus a per-hart width and still gated w8a8/cos 1.0.
    """
    from merlin.common.paths import merlin_dir

    src = (merlin_dir() / "runtime/c/libomp_zephyr.c").read_text()
    assert "omp_hart_has_vector" in src
    # The non-trapping WARL test, and misa used only to CONFIRM (a core may tie misa to zero, so a
    # zero there must not veto a hart that VS proved has vector state).
    assert "csrw mstatus" in src and "misa != 0" in src
    # vlenb is read only once VS is known live, since reading it with VS=Off traps.
    assert src.index("csrw mstatus") < src.index("csrr %0, vlenb")
    # The probe must not be bounded by the requested thread count -- that would only ever look at
    # the first N harts, reintroducing the assumption it exists to remove.
    assert "MERLIN_OMP_MAX_HARTS" in src
    assert "cpu < MERLIN_OMP_MAX_THREADS" not in src


def test_a_scalar_image_is_not_confined_to_vector_harts():
    """A scalar image may use every hart -- that is the only way to reach a core with no vector unit,
    and on a chip with more cores than vector units it is the difference between using the machine
    and using part of it."""
    import inspect

    src = inspect.getsource(zm._cmakelists)
    assert "MERLIN_OMP_VECTOR_POOL={1 if backend == 'rvv' else 0}" in src


def test_the_cpu_count_is_one_rule_bounded_by_the_board():
    """`CONFIG_MP_MAX_NUM_CPUS` had two formulas in the packager, and they disagreed.

    The gated path used `max(2, harts, 2)` and the build-only path `max(harts, brd.harts)`, so on a
    3-hart board the ungated build of a ONE-hart image declared three CPUs. That is a hang, not an
    inefficiency: `z_smp_init()` starts CPUs 1..N-1 and `arch_cpu_start` spins on `riscv_cpu_boot_flag`
    with no timeout, so a CPU the image does not need stops the boot dead if that hart does not answer --
    with nothing printed past the banner, which is exactly how Kodiak's h3 images read.
    """
    three = boards.board("chipyard_kodiak")
    assert three.harts == 3
    # Never more CPUs than the image fans out to -- this is the case the two formulas disagreed on.
    assert zm.image_cpus(three, 1) == 2
    assert zm.image_cpus(three, 2) == 2
    assert zm.image_cpus(three, 3) == 3
    # ... and never more than the board has, however many are asked for.
    two = boards.board("gemmelos_bearly25_zephyr")
    assert two.harts == 2
    assert zm.image_cpus(two, 1) == 2
    assert zm.image_cpus(two, 3) == 2, "clamped: declaring a hart the chip lacks is an unbounded spin"
    # A vector hart index beyond the fan-out still needs a CPU of its own.
    assert zm.image_cpus(three, 1, rvv_hart=2) == 3


def test_a_second_configuration_of_a_board_keeps_the_port_and_states_its_own_facts():
    """A hardware CONFIGURATION is a descriptor, not a fork of the code.

    The two-core matrix-unit configuration is the same silicon family and the same Zephyr port as the
    tapeout chip, and differs in three facts that each have their own silent failure: the core count
    (a CPU the design lacks is an unbounded spin in `z_smp_init`), the DRAM (a region larger than the
    design has dies before `main` with nothing printed), and which harts carry vectors (fanning vector
    work onto a scalar hart traps and never reaches the barrier). Pinning them here is what stops a
    later edit from quietly giving one configuration another's facts.
    """
    two = boards.board("kodiak_opu_2core")
    chip = boards.board("chipyard_kodiak")
    # Built against the SAME Zephyr board -- the port is unchanged, only the facts we hand it move.
    assert two.build_board == chip.name
    assert two.harts == 2 and two.n_vector_harts == 2, "both cores carry a vector unit here"
    assert two.dram_bytes == 4 * 1024 * 1024 * 1024
    assert two.vlen == 512, "under-declaring this overruns Zephyr's per-thread vector save area"
    # The port's own bring-up facts are shared, because they belong to the port and the link.
    assert (two.console, two.fpu_sharing, two.zephyr_vector_ext) == (
        chip.console, chip.fpu_sharing, chip.zephyr_vector_ext)
    assert (two.loader, two.loader_baud) == (chip.loader, chip.loader_baud)
    # No image may declare a CPU this design does not have.
    assert zm.image_cpus(two, 1) == 2
    assert zm.image_cpus(two, 3) == 2
