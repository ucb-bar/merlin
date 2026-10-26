# Clone the handoff branches and copy the ELF inputs

Source branches:

- Merlin: `ucb-bar/merlin`, **`comparisons/tvm-workload-tuning`**.
- Companion TVM: `grammy117/tvm`, **`gemmini/firesim-chunks`**, pinned at
  **`3423bd6818ab752ddd9cdb5d5773743e48c65f1d`**.

These branches already contain the changes. Do not apply the older handoff
source patches on top of them. No PR or merge is required to run the handoff.

## Copy two archives separately

Generated binaries and workload payloads stay outside Git. A branch checkout
alone does **not** contain the ELFs or prepared tuning inputs. Transfer these
archives from the sender, verify their hashes, and unpack both into one directory:

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| `elfs.tar.gz` | 57762342 | `01fe62d14e69c04189b7aa9bb66efc8a1448c2cd2b51d98bced635f44b289cf2` |
| `tuning-inputs.tar.gz` | 3049348 | `042bf82d63f5c7f3a5a6b9c38ab69c111d8d46f2401c0057419cece3bbfa9c87` |

`elfs.tar.gz` contains ResNet baseline/current-tuned/profiled ELFs and the three
small TinyLlama chunk ELFs. `tuning-inputs.tar.gz` contains the semantic chunks,
input/reference fixtures, replayable database hints, job order and runner
integration templates. Its retained provenance describes the original numerical
release; old absolute paths are historical metadata, not paths the launcher
must use. New actual-export chunks receive fresh independent references.

```bash
git clone --branch comparisons/tvm-workload-tuning https://github.com/ucb-bar/merlin.git merlin
git clone --branch gemmini/firesim-chunks https://github.com/grammy117/tvm.git tvm
git -C tvm checkout 3423bd6818ab752ddd9cdb5d5773743e48c65f1d
mkdir -p delivery
tar -xzf elfs.tar.gz -C delivery
tar -xzf tuning-inputs.tar.gz -C delivery
export MERLIN="$PWD/merlin" TVM="$PWD/tvm" DELIVERY="$PWD/delivery"
```

The FireSim launcher only needs the requested ELF. Its supplied bitstream already
sets the expected DDR3 memory model. Source/native-library/Python setup is for
building new candidates, not for executing the already prepared ELFs. For DSE,
use the matching native TVM library identified in
[FIRESIM_CHUNK_TUNING.md](FIRESIM_CHUNK_TUNING.md), or build that exact source.
The observed environment and original source pins accompany the tuning inputs.

## Execute

1. TinyLlama baseline plus **ResNet baseline and our current tuned ResNet**.
2. TinyLlama chunk **EvolutionarySearch + XGBoost** DSE.
3. TinyLlama tuned E2E with the same inputs/protocol as its baseline.

Parallelize independent jobs across actually available slots. Stop a DSE
candidate when its active measured cycles reach the current best; exclude
startup/loading/initialization/warmup and tighten in-flight limits. Respect the
shared four-hour maximum DSE deadline. No F2 specification or runtime is inferred.

Full TinyLlama baseline/tuned E2E ELFs must be supplied or built from the actual
remote calibrated exports; the local chunk ELFs are not full-model replacements.
Follow [FIRESIM_CHUNK_TUNING.md](FIRESIM_CHUNK_TUNING.md) for the details.
