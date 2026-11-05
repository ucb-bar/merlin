# Kodiak + OPU on FireSim (1 core)

Two patches and one Verilog file. The Kodiak part of this is a port of **your own**
`KodiakFireSimConfig` (`raghav-g13/chipyard@kodiak-multi-chip-firesim`, `ad438e3c`,
`generators/chipyard/src/main/scala/config/KodiakConfigs.scala:303`) — active mixins carried over
verbatim and in order, commented-out ones dropped. The CTC variant is deliberately not ported
(`testchipip.ctc` isn't in this checkout). So the only new thing here is the OPU.

## The one thing that will bite you

Your `kodiak-multi-chip-firesim` and `-opu-kernels` branches both pin `generators/saturn` at
`a898bdc`, which **does not contain `OuterProductUnit.scala` at all**, and all three Kodiak configs
there pass `VectorParams.genParams` whose `useOpu` defaults to false. `-opu-kernels` only bumps
firesim/firemarshal to add an `opu-m2-gemm` *workload* — it changes no RTL. That's why the OPU had
to be added rather than obtained.

So: repin saturn.

    generators/saturn -> miles-rusch-berkeley/saturn-vectors, branch opu-int8
                         ea373800169ce7bbb9ca27ececf74101a7caf9d5   (has OuterProductUnit.scala)

## Apply

    # 1. saturn @ ea37380 (branch opu-int8, already on the remote)
    cd generators/saturn && git apply /path/to/01-saturn-OPUConfigs.patch

    # 2. chipyard @ 6a9b4cd95081100dc932e422cc52777bc9a98b2d
    cd ../.. && git apply /path/to/02-chipyard-firechip-TargetConfigs.patch

Configs added: `KodiakOPUConfig` (2 tiles, OPU on both), `KodiakOPU1CoreConfig`,
`Kodiak1CoreConfig`, plus `FireSimKodiakOPU1CoreConfig` on the firechip side.
Geometry is `WithShuttleVectorUnit(512, 256, VectorParams.opuParams)` — vLen 512, dLen 256, so a
32x32 int8 MACC array = 1024 MAC/cycle, tile edge 64.

**Build the 1-core one.** The 2-core `KodiakOPUConfig` elaborates but I couldn't get it to fit/route
on the U250 — that's the "1 core only" caveat, not a functional limit.

## The FPGA clock-gate trap (`eicg_passthrough.v`)

Build `FireSimKodiakOPU1CoreConfig` with the stock clock-gate model and you will lose a full
synthesis run to unclosable hold violations. rocket-chip's `EICG_wrapper` is a latch-and-AND, i.e.
combinational logic in the clock path, so the tools add a *second* global clock net behind it for the
OPU cluster array's 131,585 clock loads — and that net can't be delay-matched to its ungated parent.
Measured on `FireSimOPUV256D128ShuttleConfig`: gated domain 3.79 ns late, **4.207 ns clock skew,
WHS -4.068 ns, 258,620 hold-failing endpoints.** Changing frequency does not help; hold and skew are
both period-independent.

`eicg_passthrough.v` replaces it with a passthrough, selected via `ClockGateModelFile` — no Scala
class and no target RTL changes. `WithEICGPassthrough` is first in the config chain so it overrides
the `WithClockGateModel` that `chipyard.config.AbstractConfig` already applies.

**Edit one line.** In the patched `OPUConfigs.scala`, `EICGPassthrough.vsrc` has a placeholder
`/ABSOLUTE/PATH/TO/eicg_passthrough.v` — point it at wherever you put the file. It prefers
`$MERLIN_EICG_PASSTHROUGH_V` if set, but don't rely on that alone: FireSim elaborates over ssh to the
build host and ssh strips arbitrary env vars, so an export in the launcher never arrives. Hence the
hardcoded fallback. Elaboration `require`s the file exists and fails loudly if not — deliberately,
because a missing `ClockGateModelFile` silently falls back and you get a *gated* build that looks
like the ungated one.

Ungating is a power/timing tradeoff, not free correctness: `OuterProductCluster.pipe` is assigned
unconditionally, so it does rely on the gate to hold state. The argument that this is safe (the
sequencer holds `clock_enable` high for every in-flight cycle, and `pipe` is re-primed per readout)
is written out in the header of the `.v`. We validate it by re-running the certified OPU microkernel
corpus against this model and requiring bit-identical digests — treat it as validated-by-test, not
proven.

## What we've run on it

Whole SpectFormer int8, bit-exact against golden (cos 1.0, max_rel 0.0), OPU vs RVV-only on the same
image: **1.584x** on `alveo_u250_firesim_kodiak_opu_1core`. Happy to share the workload side too if
useful.
