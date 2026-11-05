# Running images on the Kodiak chip

The Kodiak test rig is **shared**. A load or a reset kills whoever is already running, so the first
step is always to check that the rig is idle.

## 0. Access

The rig is a BWRC desktop that reaches the chip through a NexysVideo FPGA over Ethernet (UDP-TSI).
Ask the rig owner for its host and login. Its SSH port is not open to the wider network, so go through
a BWRC host with your own key (ask the owner to add it; never put a shared password in a file):

```sshconfig
Host kodiak-rig
    HostName <rig host>
    User <rig login>
    ProxyJump <your BWRC jump host>
```

Everything lives in `~/Desktop/kodiak` on the rig.

## 1. Is anyone using it?

Run all of these. If any shows activity, or you are unsure, **stop and ask in Slack**.

```sh
ssh kodiak-rig 'who; \
  ps -eo user,pid,etime,cmd | grep -E "udp_tsi_host|execute_program|arduino|pyuartsi" | grep -v grep; \
  tmux ls 2>/dev/null; screen -ls 2>/dev/null; \
  find ~/Desktop/kodiak -newermt "-10 minutes" -type f | head'
```

Never run an FPGA reset (`udp_tsi_host.py fpga-reset`) or a chip reset (`scripts/bash/arduino_all.sh t`)
unless the rig is confirmed idle. Say in Slack when you start and when you are done.

## 2. Health check, before your own images

```sh
cd ~/Desktop/kodiak
python scripts/py/udp_tsi_host.py ping                       # FPGA harness alive (not yet the chip)
python scripts/py/udp_tsi_host.py run sw/benchmarks/hello.riscv
```

After the program loads you must see:

```
0x1000 readback 0x80000000 — confirmed chip BootAddrReg
```

`readback None`, followed by a hang after `Proxy FESVR started.`, means the FPGA and/or the chip need a
reset. Only do that when the rig is idle (see §1). Run `hello.riscv` again before anything else.

## 3. Run Merlin images

Copy a delivery into its own directory on the rig, then run the images in this order: `vlen_probe.elf`
(expect `vlenb 64`, nonzero `mstatus_vs`), then each `_debug` image, then its plain twin.

```sh
scp -r out/artifacts/delivery/<package>/ kodiak-rig:Desktop/kodiak/sw/models/<package>/
ssh kodiak-rig 'cd ~/Desktop/kodiak && \
  python scripts/py/udp_tsi_host.py --credited 8 --auto run sw/models/<package>/vlen_probe.elf'
# a single image, keeping the console:
ssh kodiak-rig 'cd ~/Desktop/kodiak && \
  python scripts/py/udp_tsi_host.py --credited 8 --auto run sw/models/<package>/<image>.elf' \
  | tee <image>.console.txt
# or every ELF in a directory, through the rig's wrapper (retries, timeout):
ssh kodiak-rig 'cd ~/Desktop/kodiak && scripts/bash/execute_program.sh sw/models/<package>/'
```

Without `--auto` the loader stops at "Press Enter to kick hart 0 via MSIP", which looks like a hang.
Keep it attached until the image prints `DONE`: the console is HTIF, so a detached loader makes a working
image look hung.

## 4. Grade

Each delivery carries `grade.py` (numpy only):

```sh
python <package>/grade.py <image>.console.txt --model <model>
```

On a log that stopped early it reports the last `STAGE`, any `FAIL` line and failed memory probes.
Keep returned consoles: copy them into Merlin's `out/runs/` with the delivery's `manifest.json`.
