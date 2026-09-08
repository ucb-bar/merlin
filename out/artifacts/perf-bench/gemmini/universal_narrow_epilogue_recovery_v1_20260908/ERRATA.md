# Erratum: LOOP_CONV bias does not use execute-path D

The original v1 analysis incorrectly inferred that
`hardcode_d_to_garbage_address: true` prevents all accumulator bias preload.
That flag applies to the ordinary execute/preload path. The target has a
separate convolution bias loader:

- `LoopConvLdBias` emits `LOAD3_CMD` (`LoopConv.scala:84-190`).
- Its destination is encoded as an accumulator address with
  `accumulate=false` and `read_full=false`.
- `LOOP_CONV_WS_CONFIG_6` supplies the bias DRAM address
  (`LoopConv.scala:1340-1347`).
- The public `gemmini_loop_conv_ws` macro passes `bias` through CONFIG_6
  (`gemmini.h:384-400`).

Pinned evidence inspected on 2026-09-08:

| Evidence | SHA-256 |
|---|---|
| `/scratch/jack/chipyard/generators/gemmini/src/main/scala/gemmini/LoopConv.scala` | `8609d9f27ff037ffdeaba53f70e8c0a9b3a994c3871e9eacbee20e58d3dc6c6d` |
| `/scratch/jack/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/gemmini.h` | `2f97540b6ba4378c22572e7d4a41d5d0c95931ac55dcf1fa5246b3923e4ad3ff` |
| Jack's generated `target_config.json` | `8a6c8e777c5765efdabd6bb84312f22d5e4cd2ab9fb04e5313441472129a51e4` |

The corrected current admission remains 0/53, but for a different reason:
all 53 captured convolutions require per-channel output scaling while a single
LOOP_CONV store configuration supplies one scale. The required compiler work is
therefore native LOOP_CONV plus exact channel-partitioned stores, followed by
separate handling of residual/global tails.
