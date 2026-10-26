# Published FireSim DSE handoff

**Execution scope: do not implement the proposed optimization knobs.** They
were ideas to investigate. Use the previously agreed existing TVM/Gemmini
contraction DSE, graph fusion fixed at level 2, meaningful chunks, parallel jobs,
incumbent active-cycle limits and connected/E2E confirmation. Do not add host
loop scheduling, producer placement, panelized packing or new search axes.

Use only capabilities supported by the pinned hardware. Do not invent hardware
support or silently move an eligible Gemmini contraction to the CPU. Existing
permitted host work must remain explicit. A frontend import failure is distinct
from unsupported hardware; an earlier E2E result qualifies its actual ELF and
source path, not every fresh export.

Everything needed to restore the prepared inputs is committed on
`comparisons/tvm-workload-tuning`. The source changes are already in the
branch. The frozen 170,024,497-byte input archive is stored in five
checksum-verified parts; no external payload download or old workspace path
is required. A new baseline ELF is also committed directly for immediate use.

## Fresh machine

```bash
git clone --depth 1 --single-branch --branch comparisons/tvm-workload-tuning \
  https://github.com/ucb-bar/merlin.git merlin
export MERLIN="$PWD/merlin"
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-firesim-dse-publication/v1/latest"
cd "$DELIVERY"
sha256sum -c SHA256SUMS
# Python 3.12+; no Torch, TVM or simulator is needed to restore the inputs.
python3 restore.py --output-dir "$DELIVERY/unpacked"
export INPUTS="$DELIVERY/unpacked"
```

Read `$INPUTS/README.md` for setup, job rosters, numerical policy, shared
budget, observer qualification and baseline/tuned confirmation. That README is
the frozen preparation record and describes an old patch-based source transfer.
**For this branch checkout, skip its source-patch/overlay application block.**
Use the committed source files directly. The archived patch remains available
only for reproducing its original `ee4031d` base.

The restore helper verifies all five parts, the assembled archive and all 182
payload file hashes. Then use `$INPUTS/relocate_proofs.py` to write new reference
proof copies as instructed there. All 19 fixture proofs were checked from a
fresh extraction. Templates are intentionally unqualified; use actual installed
manager, compiler/functional-engine receipts and physical slot assignments.

## ELF-only baseline

`$DELIVERY/ELFS/resnet-baseline.elf` is the complete new-source ResNet baseline.
It can be submitted directly to the existing ELF launcher without restoring
inputs or installing a compiler. Its SHA256 is
`4ec5e14fa82724d6aff3526a6beeb273dfa18637d4e316b386f5cbe487d44acf`.
The positive and wrong-oracle Spike controls passed, checking all 1,000 logits.
The negative-control ELF and functional receipts are inside the restored bundle.

The supplied bitstream already sets memory; **DDR3 is expected**. No memory
configuration, hardware rebuild, F2 specification or runtime estimate is added.
Use the guest active measured window, including host work and completion.
FireSim execution and application accuracy remain NOT_RUN for this new source.

## Tuning scope

- ResNet: 18 full-shape chunks, 21 distinct contraction workloads, 10 tuning
  owners; all 18 host checks have zero difference. The source adds all 54 biases
  in INT32 with explicit equivalence/bounds proofs and frozen weights/scales.
- Initial ResNet campaigns reserve 48 proposals; continuation reserves 20.
  TinyLlama reserves 24 once actual inputs exist. One shared proposal ledger,
  a cap of 96 and one absolute deadline of at most four hours cover the study.
- Use stock TVM EvolutionarySearch/XGB through the published Gemmini fork and
  supported adapter APIs. Independent campaigns share physical slot locks,
  batch submissions and bounded host builds. There is no new TVM source patch.
- TinyLlama still lacks current-policy calibrated exports. Q/O, gate/up, full
  down projection and QK/PV/cache/context coverage are recorded as pending.
  The older dynamic-row probes do not replace these inputs.
- Before paid DSE, demonstrate startup/warmup exclusion, an actual measured
  cycle cutoff, in-flight tightening and real remote cancellation. Stop at
  `active_cycles >= incumbent` for the same workload. The previous adapter's
  total-cycle cap and completed-result filtering did not implement this.

Host scheduling and bounded packing remain proposed work; this first roster
tunes contraction schedules and confirms connected host-inclusive costs.
ResNet fixed-set accuracy and TinyLlama matched WikiText-2 test PPL gates remain
separate from compiler correctness. The frozen instructions record those gaps.

TVM source: `https://github.com/grammy117/tvm`, branch
`gemmini/firesim-chunks`, commit `3423bd6818ab752ddd9cdb5d5773743e48c65f1d`.
The archive's source-pins file records the original local Git remote; use the
published repository above and verify this exact commit.

See `transfer-manifest.json`, `publication-checks.json` and `PROVENANCE.md`
for identities and validation. No paid hardware jobs were launched locally.
