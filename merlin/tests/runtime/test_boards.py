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
    """cpus, the vector save-area width and FPU_SHARING all come from the descriptor."""
    b = boards.board("spike_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert f"CONFIG_MP_MAX_NUM_CPUS={b.harts}" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={b.vector_max_len}" in conf
    # y mis-routes V-illegal-instruction traps into the FP path and retries forever: a silent hang.
    assert "CONFIG_FPU_SHARING=n" in conf


def test_kodiaks_config_reflects_what_its_zephyr_can_actually_build():
    """Two settings Kodiak forces, both learned by failing to link, both recorded in the descriptor.

    FPU_SHARING must be y: its Zephyr's isr.S calls z_riscv_vstate_save/_restore whenever V is on with
    eager switching, and those live in fpu.c/fpu.S which only compile under FPU_SHARING -- with n the
    image does not link. And Zephyr-level V must be OFF: that tree has no RISCV_V_KERNEL_ONLY, so
    enabling V puts `v` in the GLOBAL march and SDK 0.17.0 has no matching libgcc multilib (the link
    falls back to a 32-bit one: "ELFCLASS32 incompatible with ELFCLASS64"). Vectors still work because
    reset.S enables mstatus.VS under CONFIG_FPU, and our model.o carries `v` itself.
    """
    b = boards.board("chipyard_kodiak")
    assert b.fpu_sharing is True and b.zephyr_vector_ext is False
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_MP_MAX_NUM_CPUS=3" in conf
    assert "CONFIG_FPU_SHARING=y" in conf
    assert "CONFIG_RISCV_ISA_EXT_V=y" not in conf
    assert "CONFIG_FPU=y" in conf, "mstatus.VS comes from CONFIG_FPU on this tree"


def test_a_vector_capable_tree_still_gets_the_full_vector_config():
    b = boards.board("chipyard_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_RISCV_ISA_EXT_V=y" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={b.vector_max_len}" in conf


def test_htif_console_options_are_set_only_for_an_htif_board():
    """Both options default to n upstream, and unbuffered HTIF emits one char per host round-trip --
    on a ~20 MHz core that looks like the model never finishing. A non-HTIF board must not get options
    its driver does not have."""
    htif = zm._prj_conf(2, "rvv", boards.board("chipyard_kodiak"))
    assert "CONFIG_UART_HTIF_BUFFERED_OUTPUT=y" in htif
    assert "CONFIG_UART_HTIF_SYSCALL_PRINT=y" in htif
    uart = zm._prj_conf(2, "rvv", boards.board("x", console=boards.CONSOLE_UART))
    assert "UART_HTIF" not in uart
    assert "board-provided" in uart


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
        assert b.vlen == 128, "stated by its own OPE kernel; their spike runs use 256"


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
