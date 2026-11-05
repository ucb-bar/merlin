The Kodiak rig loads and runs images over Ethernet (UDP-TSI) through its FPGA, from `~/Desktop/kodiak` on
the rig host. **The rig is shared: first check that nobody is running** (`who`, running
`udp_tsi_host`/`execute_program` processes, fresh logs). Never reset the FPGA or the chip under someone
else's run.

```bash
# health check: the FPGA answers, then the chip boots a known image
python scripts/py/udp_tsi_host.py ping
python scripts/py/udp_tsi_host.py run sw/benchmarks/hello.riscv
#   must print: 0x1000 readback 0x80000000 — confirmed chip BootAddrReg

# run one image and keep its console
python scripts/py/udp_tsi_host.py --credited 8 --auto run <THIS.elf> | tee <THIS>.console.txt
# or every ELF in a directory, with retries and a timeout
scripts/bash/execute_program.sh <dir with the ELFs>
```

Without `--auto` the loader stops at "Press Enter to kick hart 0 via MSIP", which looks like a hang.
`readback None`, followed by silence after `Proxy FESVR started.`, means the FPGA and/or the chip need a
reset (`python scripts/py/udp_tsi_host.py fpga-reset`, `scripts/bash/arduino_all.sh t`), and only when
the rig is idle.

### Keep the loader attached for the WHOLE run

This board's console is HTIF, not a self-contained UART: the image writes a word and waits for the host
(the loader's FESVR) to acknowledge it. Stay attached until `DONE`. A detached image looks hung in its very
first print while it is in fact computing and waiting to be read.

Run `vlen_probe.elf` first (expect `vlenb 64`), then each `_debug` image before its plain twin.
