"""Deriving a chip's console facts from its SDK headers, instead of hardcoding them.

The bug these guard against is specific and was shipped: bare-metal images printed over HTIF, a
host-assisted channel, so on real silicon (no host clearing ``tohost``) they hung inside the first
print before any model work — indistinguishable from a core that never booted. Speaking the chip's own
UART instead needs its MMIO addresses, register offsets, bit positions and clock rates, and every one
of those is a fact about one tapeout that must be *extracted* rather than written down.

The SDK is synthesised here rather than pointed at a checkout, so these tests are portable and so the
parser is exercised on the shapes that actually broke it: a ``/** ... */`` description after the ``;``
(which silently dropped every second register), casts in enum initialisers, an address spelled as an
expression over other defines, and a struct whose own offset comments can be used as a check.
"""
from __future__ import annotations

import pytest

from merlin.runtime import sdk_facts
from merlin.runtime.sdk_facts import (SdkFactError, eval_define, parse_defines, parse_enum,
                                      parse_struct_offsets, strip_comments)

# A UART register map in the vendor style: CMSIS-ish volatile qualifiers and `/** ... */` docs placed
# AFTER the semicolon, which is what made a naive split read them as part of the next declaration.
UART_H = """
#ifndef __UART_H
#define __UART_H
#define UART_TXDATA_FULL_POS                    (31U)
#define UART_TXDATA_FULL_MSK                    (0x1U << UART_TXDATA_FULL_POS)
#define UART_TXCTRL_TXEN_POS                    (0U)
#define UART_TXCTRL_NSTOP_POS                   (1U)
#define UART_RXCTRL_RXEN_POS                    (0U)

typedef struct {
  __IO uint32_t TXDATA;                         /** Transmit data register */
  __I  uint32_t RXDATA;                         /** Receive data register */
  __IO uint32_t TXCTRL;                         /** Transmit control register */
  __IO uint32_t RXCTRL;                         /** Receive control register */
  __IO uint32_t IE;                             /** UART interrupt enable */
  __I  uint32_t IP;                             /** UART interrupt pending */
  __IO uint32_t DIV;                            /** Baud rate divisor */
} UART_Type;
#endif
"""

# The PLL map carries explicit offset comments, so the computed offsets have an independent check.
PLL_H = """
typedef struct {
  __IO uint32_t POWERGOOD_VNN;                          // 0x00
  __IO uint32_t PLLEN;                                  // 0x04
  __IO uint32_t LDO_ENABLE;                             // 0x08
  __IO uint32_t RATIO;                                  // 0x0C
  __IO uint32_t FRACTION;                               // 0x10
  __IO uint32_t MDIV_RATIO;                             // 0x14
  __IO uint32_t ZDIV0_RATIO;                            // 0x18
  __IO uint32_t ZDIV1_RATIO;                            // 0x1c
  __IO uint32_t PLLFWEN_B;                              // 0x20
} PLL_Type;
"""

HAL_RCC_H = """
typedef enum {
  CLKSEL_SLOW = (uint32_t)0,
  CLKSEL_PLL0 = (uint32_t)1,
  CLKSEL_PLL1 = (uint32_t)2
} ClockSel_Opts;

typedef struct {
  __IO uint32_t UNCORE;                                 // 0x00
  __IO uint32_t TILE0;                                  // 0x04
  __IO uint32_t TILE1;                                  // 0x08
  __IO uint32_t CLKTAP;                                 // 0x0C
} ClockSel_Type;
"""

# Note the indirections a real header uses: UART0_BASE via UART_BASE, and a clock selector that is an
# OFFSET inside the RCC block expressed through a pointer cast. Both must resolve.
CHIP_CONFIG_H = """
#include "uart.h"
#define SYS_CLK_FREQ   50000000
#define MTIME_FREQ     50000
#define RCC_BASE                 0x00100000U
#define UART_BASE                0x10020000U
#define PLL_BASE                 0x00140000U
#define UART0_BASE               (UART_BASE)
#define RCC_CLOCK_SELECTOR       ((ClockSel_Type*)(RCC_BASE + 0x30000))
#define PLL                      ((PLL_Type *)PLL_BASE)
"""


@pytest.fixture()
def sdk(tmp_path):
    """A Baremetal-IDE-shaped SDK tree with two chips, so chip selection is actually exercised."""
    for chip in ("chipa", "chipb"):
        d = tmp_path / "platform" / chip
        (d / "include").mkdir(parents=True)
        (d / "chip_config.h").write_text(CHIP_CONFIG_H)
        (d / "include" / "hal_rcc.h").write_text(HAL_RCC_H)
    uart = tmp_path / "driver" / "rocket-chip-blocks" / "uart"
    uart.mkdir(parents=True)
    (uart / "uart.h").write_text(UART_H)
    # A DECOY uart.h from another vendor driver: picking by filename alone would be ambiguous, which is
    # why the finder disambiguates by content.
    decoy = tmp_path / "driver" / "national-semiconductor" / "ns16550a"
    decoy.mkdir(parents=True)
    (decoy / "uart.h").write_text("#define UART_ADDRESS 0x10000000\nvoid uart_putc(char c);\n")
    pll = tmp_path / "driver" / "intel" / "pll"
    pll.mkdir(parents=True)
    (pll / "pll.h").write_text(PLL_H)
    return tmp_path


# ------------------------------------------------------------------ expression evaluation ---------
@pytest.mark.parametrize("body,want", [
    ("0x10020000U", 0x10020000),
    ("500000000ULL", 500_000_000),
    ("(31U)", 31),
    ("(0x1U << 31)", 1 << 31),
    ("(uint32_t)0", 0),                        # a cast, not a value
    ("((ClockSel_Type*)(0x100000 + 0x30000))", 0x130000),
    ("1 | 2 | 4", 7),
    ("(1 << 3) + 1", 9),
])
def test_evaluates_the_constant_forms_vendor_headers_use(body, want):
    assert eval_define("X", {"X": body}) == want


def test_resolves_references_to_other_defines():
    defs = parse_defines(CHIP_CONFIG_H)
    # UART0_BASE is spelled `(UART_BASE)`: a parenthesised identifier that is a VALUE, not a cast.
    assert eval_define("UART0_BASE", defs) == 0x10020000
    # And the clock selector is an offset inside the RCC block, only written down in this cast form.
    assert eval_define("RCC_CLOCK_SELECTOR", defs) == 0x130000


def test_a_missing_or_uncomputable_define_raises_rather_than_defaulting():
    # Fail closed: a defaulted console address produces no output, the one failure a remote user
    # cannot debug.
    with pytest.raises(SdkFactError):
        eval_define("NOPE", {})
    with pytest.raises(SdkFactError):
        eval_define("X", {"X": "SOME_UNDECLARED_THING"})


def test_cyclic_defines_raise_instead_of_recursing_forever():
    with pytest.raises(SdkFactError):
        eval_define("A", {"A": "B", "B": "A"})


def test_function_like_macros_are_not_read_as_constants():
    assert "F" not in parse_defines("#define F(x) ((x) + 1)\n")


def test_strip_comments_keeps_string_literals_intact():
    assert "http://x" in strip_comments('const char *s = "http://x";')


# ------------------------------------------------------------------------ register maps -----------
def test_offsets_survive_a_doc_comment_placed_after_the_semicolon():
    # The regression: `/** ... */` trailing one declaration lands at the head of the next chunk when
    # splitting on ';', and dropping it silently lost every second register -- so TXCTRL/RXCTRL/DIV
    # vanished and the console wrote to the wrong addresses.
    off = parse_struct_offsets(UART_H, "UART_Type")
    assert off == {"TXDATA": 0, "RXDATA": 4, "TXCTRL": 8, "RXCTRL": 12, "IE": 16, "IP": 20, "DIV": 24}


def test_computed_offsets_are_checked_against_the_headers_own_comments():
    assert parse_struct_offsets(PLL_H, "PLL_Type")["PLLFWEN_B"] == 0x20
    # A header whose stated offset disagrees with the layout means the map was misread; that must be
    # loud, because a misread register map writes to a neighbouring register.
    lying = PLL_H.replace("__IO uint32_t LDO_ENABLE;                             // 0x08",
                          "__IO uint32_t LDO_ENABLE;                             // 0x40")
    with pytest.raises(SdkFactError, match="misread"):
        parse_struct_offsets(lying, "PLL_Type")


def test_unknown_field_types_and_arrays_raise():
    with pytest.raises(SdkFactError):
        parse_struct_offsets("typedef struct {\n  weird_t X;\n} S;\n", "S")
    with pytest.raises(SdkFactError):
        parse_struct_offsets("typedef struct {\n  uint32_t X[4];\n} S;\n", "S")


def test_mixed_widths_get_natural_alignment():
    src = "typedef struct {\n uint8_t A;\n uint32_t B;\n uint64_t C;\n} S;\n"
    assert parse_struct_offsets(src, "S") == {"A": 0, "B": 4, "C": 8}


def test_enum_values_parse_through_casts_and_implicit_increments():
    got = parse_enum(HAL_RCC_H, "ClockSel_Opts")
    assert got == {"CLKSEL_SLOW": 0, "CLKSEL_PLL0": 1, "CLKSEL_PLL1": 2}
    assert parse_enum("typedef enum { A = 4, B, C } E;\n", "E") == {"A": 4, "B": 5, "C": 6}


# --------------------------------------------------------------------------- the facts ------------
def test_derives_every_console_fact_from_the_sdk(sdk):
    f = sdk_facts.derive_uart_console(sdk, "chipa")
    assert f.uart_base == 0x10020000
    assert f.reg["DIV"] == 24                 # the register the baud divisor is written to
    assert f.tx_full_bit == 31 and f.txen_bit == 0 and f.nstop_bit == 1
    assert f.sys_clk_hz == 50_000_000 and f.mtime_hz == 50_000
    assert f.pll_base == 0x140000 and f.clksel_base == 0x130000
    assert (f.clksel_slow, f.clksel_pll) == (0, 1)
    assert f.pll["RATIO"] == 0x0C
    # Provenance names the files the numbers came from, so a wrong value is traceable to a header.
    assert f.provenance["chip_config.h"] == "platform/chipa/chip_config.h"
    assert f.provenance["uart.h"].endswith("rocket-chip-blocks/uart/uart.h")   # not the decoy


def test_unknown_chip_and_missing_sdk_raise(sdk, tmp_path):
    with pytest.raises(SdkFactError):
        sdk_facts.derive_uart_console(sdk, "nosuchchip")
    with pytest.raises(SdkFactError):
        sdk_facts.derive_uart_console(tmp_path / "nope", "chipa")


def test_macros_carry_the_facts_and_omit_the_pll_when_not_raising_it(sdk):
    f = sdk_facts.derive_uart_console(sdk, "chipa")
    plain = f.macros()
    assert "-DMERLIN_UART_BASE=0x10020000ULL" in plain
    assert "-DMERLIN_UART_DIV_OFF=24" in plain
    assert f"-DMERLIN_UART_BAUD={sdk_facts.DEFAULT_BAUD}" in plain
    # Unescaped quotes: these flags are passed to subprocess as a list, with no shell to strip a
    # backslash, so an escaped quote would land inside the string literal and fail to compile.
    assert '-DMERLIN_CONSOLE_NAME="uart"' in plain
    # No PLL macros unless a target frequency was asked for -- console_uart.c keys the whole PLL
    # sequence off MERLIN_CHIP_FREQ_HZ, so their absence IS "leave the chip on its reset clock".
    assert not [m for m in plain if "PLL" in m or "CHIP_FREQ" in m]

    raised = f.macros(chip_freq_hz=500_000_000)
    assert "-DMERLIN_CHIP_FREQ_HZ=500000000ULL" in raised
    assert "-DMERLIN_PLL_RATIO_OFF=12" in raised
    assert "-DMERLIN_CLKSEL_N=4" in raised


def test_a_non_contiguous_clock_selector_is_refused(sdk):
    # The harness walks the clock domains as one 32-bit array (that is what the SDK's set_all_clocks
    # does); a gap would send one of those writes to a neighbouring register.
    f = sdk_facts.derive_uart_console(sdk, "chipa")
    holed = type(f)(**{**f.__dict__, "clksel": {"UNCORE": 0, "TILE0": 4, "CLKTAP": 64}})
    with pytest.raises(SdkFactError, match="contiguous"):
        holed.macros(chip_freq_hz=500_000_000)


def test_an_incomplete_register_map_is_refused(sdk):
    f = sdk_facts.derive_uart_console(sdk, "chipa")
    short = type(f)(**{**f.__dict__, "reg": {"TXDATA": 0}})
    with pytest.raises(SdkFactError, match="register map lacks"):
        short.macros()
