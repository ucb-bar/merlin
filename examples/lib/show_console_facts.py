"""Print the console + clock facts Merlin derives from a chip's own SDK headers.

Its own file rather than an inline `python -c` so the example can echo a command a reader can retype,
and so the derivation is greppable from the repo.

Why this stage exists at all: a chip with no Zephyr port has no board files to read a UART address out
of, and hardcoding one is how you ship an image that prints nothing — indistinguishable from a hang. The
numbers below are parsed from the vendor's headers at build time, and the same values feed both the
Kconfig (`CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC`, the baud divisor) and the device-tree overlay, so the two
cannot disagree about which UART this is.
"""
from __future__ import annotations

import argparse
import json

from merlin.runtime.sdk_facts import derive_uart_console


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sdk_dir", help="the chip's SDK checkout")
    ap.add_argument("--chip", required=True, help="platform key inside that SDK (e.g. bearly25)")
    a = ap.parse_args(argv)

    f = derive_uart_console(a.sdk_dir, a.chip)
    print(json.dumps({
        "uart_base": hex(f.uart_base),
        "uart_regs": {k: hex(v) for k, v in sorted(f.reg.items())},
        "sys_clk_hz": f.sys_clk_hz,
        "mtime_hz": f.mtime_hz,
        "pll_base": hex(f.pll_base) if f.pll_base else None,
        "clksel_base": hex(f.clksel_base) if f.clksel_base else None,
    }, indent=2))
    print("\n# sys_clk_hz is the RESET clock. pll_base/clksel_base are what the 500 MHz variant")
    print("# reprograms, in the vendor's own order, before the console divisor is applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
