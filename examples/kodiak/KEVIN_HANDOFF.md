# Kodiak OPU handoff

Use Merlin branch `feat/OPU-VME`. The Kodiak and Iris package updates are on their respective
`handwritten-implementation` branches. Merlin's setup fetches tested package commits, so the
package branch heads may contain newer documentation than the image-building pins.

## Compile and check an image

```sh
git clone --branch feat/OPU-VME https://github.com/ucb-bar/merlin.git
cd merlin/examples/kodiak
./run.sh setup --llvm <llvm23-install> --chipyard <chipyard> --zephyr-sdk <zephyr-sdk-0.17.0>
./run.sh preflight
./run.sh op gemm 64 64 64
./run.sh kernel-test opu_tile_loop --quick
```

`setup` prints any missing dependency or pin drift. `preflight` must pass before using the output
as a hardware image. `op gemm` compares RVV, a scalar matrix twin, and the shipped OPU image on
its pinned Spike extension. To test a custom C kernel, register it under the Kodiak package's
`kernels/` and pass `--matrix-kernel <name>`; the kernel ABI and test flow are in that package's
README.

`./run.sh firesim --models spectformer --gate independent-rvv` builds both one-core images, audits
OPU instructions, checks the exact images on Spike, and prints an immutable FireSim plan. Add
`--submit` to place the pair in the shared FPGA queue. The control leg is plain RVV on the **same
OPU bitstream**. `./run.sh firesim-collect <plan-dir> --job device=N --job control=M` grades the
returned full-output hashes, W8A8 numerical gate, bitstream identity, and target compute cycles.
For a verified whole-model OPU pair with an exact routing sidecar, it saves
`firesim/utilization.json` with routed MAC share, static tile fill, whole-inference effective
peak, and an explicit unmeasured array occupancy field.
The queue can take hours. See [the full example guide](README.md) for capture, build, package,
and board options.

## Verified scope

| Workload | Exact one-core OPU FireSim result | Matched RVV control | Output check |
| --- | ---: | ---: | --- |
| Spectformer int8 | 3,531,733,777 target compute cycles | 6,120,729,118 cycles | W8A8 pass; full hash/prefix match its Spike image |
| TinyLlama two-layer, 64 tokens | 15,380,012,801 cycles | 27,955,258,588 cycles | Full 2,048,000-value hash/prefix match Spike; 4,096-prefix hardware W8A8 pass |
| TinyLlama two-layer, 128 tokens | 16,727,634,195 cycles | 41,920,445,165 cycles | Both full 4,096,000-value hashes/prefixes match Spike; 4,096-prefix hardware W8A8 pass |

An aligned-weight prepack s64 OPU image completed as FireSim job 3866 in **1,101,993,218**
target cycles with the same full 2,048,000-value hash as the verified baseline. Its matched RVV
control is running as job 3867, so the new pair's speedup and grade are pending. This device
result performs 9,865,003,008 valid routed MACs, or **0.874215% of the 1,024-MAC/cycle
theoretical peak across the inference**; the earlier OPU image measured 0.0626384%. The s128
aligned-prepack OPU and RVV images have both passed exact Spike gates, including the baseline's
full 4,096,000-value hash, but have not yet been submitted to FireSim.

The full TinyLlama model's weights alone are about 1.2 GiB and do not fit this 512 MiB board.
The two-layer sections above are the hardware targets. The 64-token and 128-token exact OPU
Spike images also passed complete W8A8 comparisons and all token argmaxes against their host
reference. The paired FireSim OPU and RVV images are built with one vector hart, matching the
one-core bitstream.

For utilization, use [the formula and evidence](UTILIZATION.md) and
`python examples/lib/opu_utilization.py <verified-plan-dir>`. The current bitstream has 1,024
physical int8 MAC cells but no active-cycle counter. The baseline ~0.05–0.12% figures and the
prepacked s64 device's 0.874215% are
**whole-inference effective throughput as a fraction of peak**, not array occupancy. Exact
Spectformer and 64-token TinyLlama Spike images executed 105 and 143 OPU shim calls respectively,
with zero scalar fallbacks. The graded 128-token pair gives **2.5061×** control-over-device
speedup and **0.115576%** whole-inference effective peak for the OPU image; the bitstream still
has no array-active counter.

The Kodiak OPU target uses legacy BME instructions. It cannot execute Iris's Zvt/VME encodings;
those need a separate VME-enabled Kodiak RTL configuration and bitstream. The current OPU FireSim
bitstream is usable for the BME flow above. Its archived artifact identity was checked during the
hardware jobs, while reproducing that archive from a clean source tree remains unverified because
the live Chipyard/Saturn checkout has pin drift. Physical Kodiak rig access and scheduling are
separate from FireSim; follow the Kodiak package's `RUNNING.md` and coordinate with the rig owner
before any chip load or reset.

As of 2026-11-06, `ssh -o BatchMode=yes kodiak-rig hostname` returns `bwrckodiak` without a
laptop relay. In an owner-confirmed exclusive window, a SHA-verified JTAG program restored FPGA
Ethernet and UDP ping. The exact OPU-direct benchmark that passed in August then failed at the
chip's `0x1000` BootAddrReg readback under the historical uncredited `--no-reset` loader path.
No Merlin image was loaded; existing tmux sessions were preserved. The hash-verified comparison
is in `out/artifacts/probes/chipyard_kodiak/physical_probe_20261106/receipt_addendum.json`.
Chip power/reset/clock/wiring and the Vivado version difference still need investigation before
direct Kodiak testing. The Spike-gated small GEMM OPU/RVV images are ready for retry after the
known positive control passes.

A sparse Spectformer OPU profile image passed its exact Spike gate with all 90 routed calls,
matching full output hash, and **10.6057% effective peak during OPU call windows on Spike**.
Its FireSim job 3868 is queued behind the prepack control. This call-window rate includes packing
and stalls; it is not physical MAC array occupancy. A separate draft Chipyard counter patch is
available as [PR 2395](https://github.com/ucb-bar/chipyard/pull/2395), but no instrumented
bitstream or counter measurement exists yet.
