# Physical Kodiak BootAddr reset-path read-only audit

Goal: Read-only audit of physical Kodiak SSH access, occupancy, and the historical
BootAddrReg failure. No FPGA/chip/UDP command was issued.

## 2026-11-07 UTC observation

- `ssh -o BatchMode=yes -o ConnectTimeout=15 kodiak-rig hostname` returned
  `bwrckodiak`. `<Kodiak rig workspace>` is readable. The SSH path is
  the established laptop-independent alias; this audit did not alter it.
- `tmux ls` showed `eth` and `huh` **attached** (plus session `1`). `ps` showed no
  `udp_tsi_host.py`, `execute_program.sh`, `arduino_all.sh`, OpenOCD, or Vivado
  process. Attached sessions remain untouched. This does not establish current
  exclusive ownership or chip health.
- The rig's current default bitstream file is
  `<Kodiak rig workspace>/bitstreams/nexysvideo/ethernet/chisel-3-startup/NexysVideoDualChipBringUpHarness.bit`,
  SHA256 `02083a671e63de5437cb1d12492cfb516c40019d2a456e749f2c27b0a9169d6c`.
  The current script SHAs are `udp_tsi_host.py` =
  `857c09058feb653ed71db559c588c63ee602161507731c6a2b39cac5077c5ad4`
  and `execute_program.sh` =
  `ff629e587779b88a4ac8f153557a0e59a0651dac2b1e71c5608387708dc840c5`.
  Historical logs identify the same bitstream **path**; they do not prove the
  historical bytes match the current SHA.

## Reset-path finding

The direct `python scripts/py/udp_tsi_host.py run FILE` CLI defaults to
`--reset-mask 0x3` (`scripts/py/udp_tsi_host.py:1561-1564,1689`) and holds both
chips in reset during loading (`:1454-1463`), then releases them immediately
before BootAddr communication (`:1485-1491,1103-1112`).
`set_chip_reset()` only sends a UDP control packet (`:583-589`); it does not
wait for an acknowledgement or a settle interval. By contrast,
`scripts/bash/execute_program.sh` defaults to `--no-reset` (`:20,160-165`);
the explicit wrapper `--chip-reset` option removes that flag (`:45,161-163`).

The existing 2026-08-18 wrapper logs show a perfect *observational* split:

| BootAddr readback | Reset during load | Rig log and SHA256 |
| --- | --- | --- |
| `None` | yes | `scripts/log/run-20260818-164805.log` `03ee1be06719d95922cc22cef89cd48db82c16edb6bf4e10b1b44e282255f5a1` |
| `None` | yes | `scripts/log/run-20260818-165358.log` `0140277763ec5da14ebfcaaa62b5d76ffe4a93efaa7a95cf33062e3f03c350ee` |
| `None` | yes | `scripts/log/run-20260818-173949.log` `d5376891b60788bf74c0c1554c591c775b1b9da0c2f126d640a7266107aa42c8` |
| `None` | yes | `scripts/log/run-20260818-181619.log` `115d2450af941b9a5c72cffa9cab850f55d8b411ff43e20f19485eebb2d1661e` |
| `0x80000000` | no | `scripts/log/run-20260818-220529.log` `ce7f89bb0e49cc78c151f2965d299b7390aa8ffc3909f70bd43d93930ae185c6` |
| `0x80000000` | no | `scripts/log/run-20260818-220950.log` `171b3ba290b6c216429e4f1d8130eb0dd9000b35b8b0354b8a1a67669399d402` |
| `0x80000000` | no | `scripts/log/run-20260818-223142.log` `4325b58f818dfdb49bf3fd617e5c8f2fa662f732f0bd267aeb0791b96f441f0e` |
| `0x80000000` | no | `scripts/log/run-20260818-230000.log` `abe20f0100e708534e00d999b79108f62ffea99306e182605cdfe93c36acb615` |
| `0x80000000` | no | `scripts/log/run-20260818-230039.log` `20328ce02dc72f9550a98b890cf486789134bbd931a20f54356b647010c270bb` |
| `0x80000000` | no | `scripts/log/run-20260818-230151.log` `0bf48b9acbf672b9a1c63639fb470eb2d45c786e3bbe02379e9954b4b33c6f91` |
| `0x80000000` | no | `scripts/log/run-20260818-230441.log` `650ebfb47af0d96e3e4dd841f49a796b37d735a0bcd8d3348986da13ed8aadec` |

Example failed hello: `run-20260818-181619.log:62,75-79,87,106-109`
records successful FPGA programming and ping, release from chip reset, a
BootAddr write ACK, then readback `None`. Example positive readback:
`run-20260818-220950.log:62,75-79,106-107,136` records the same bitstream
path, ping, BootAddr `0x80000000`, and program PASS. The earlier no-reset
`run-20260818-220529.log` had a positive readback but later timed out;
BootAddr is a necessary gate, not full execution proof.

**Causal status:** reset sequencing or an unacknowledged cross-port control
write is a plausible software/tool-path cause of BootAddr `None`, but the logs
are not a controlled A/B test. The present chip state is unmeasured. Keep
physical programming, reset, ping, and loader tests blocked until ownership is
rechecked and a safe, agreed recovery/test sequence is available. Do not infer
that the compiler or OPU caused this BootAddr failure.
