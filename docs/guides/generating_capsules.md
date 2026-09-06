---
title: Generating capsules for a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-09-05
related: [adding_a_target, gemmini_experiment, capsule_bench]
code_refs: [merlin/contract/capsules/generate_corpus.py, merlin/python/merlin/targetgen/corpus_synth.py, merlin/python/merlin/targetgen/conformance.py]
---

# Generating capsules for a target

One command regenerates a target's whole capsule corpus:

```bash
PYTHONUNBUFFERED=1 \
MERLIN_MLC_DIR=<mlc modeling dir> \
MERLIN_M2M_PYTHON=<model2MLIR venv python> \
TMPDIR=/scratch/<you>/tmp \
PYTHONPATH=$PWD/merlin/python .venv/bin/python \
  merlin/contract/capsules/generate_corpus.py --target <target>
```

`--target` is the only scoping flag; omitting it regenerates every target with a profile.
`--comparison-manifest` additionally emits the cross-target op-comparison manifest.

## Set PYTHONUNBUFFERED=1

The generator prints per-family progress, but Python block-buffers stdout when it is redirected, so
without this the log stays **empty for the whole run** and a slow generation is indistinguishable from a
hung one. Measured 2026-09-05: an atlas run was killed after 70 minutes of an apparently silent log that
was in fact buffered.

## What the environment controls

| variable | effect if missing |
|---|---|
| `MERLIN_MLC_DIR` | RTL facts unavailable → perf families report "gate unestablished" |
| `MERLIN_M2M_PYTHON` | the model/pytorch builders skip loudly and `return None` |
| `SPECIR_ROOT` | the SPEC builder skips the same way |
| `TMPDIR` | large builds need scratch; an empty `/tmp` can fail the build |

A skipped builder **leaves the existing directory untouched** — regeneration is stale-making, not
destructive. The provenance manifest MERGES rather than replaces, so a skipped capsule also keeps its
existing classification. The real hazard is narrower: a skipped capsule silently goes stale while every
sibling regenerates against a new template, and nothing reports it.

## Reading the output

Three outcomes, and they mean different things:

- **`[skip] … gate unestablished`** — a fact class needed to decide the family could not be derived.
  Usually an EXTRACTION gap, not a property of the machine. Recoverable by re-extracting RTL facts.
- **`[skip] <capsule>: <reason>`** — an honest refusal. Example: radiance declares no `op.matmul` (it is
  SIMT, with warp-level ops), and the program emitter does not read the op token, so an unchecked
  reference would emit a *different* op's program under that name.
- **`[FAIL] N capsule(s) could not be written`** — a real defect. The process exits non-zero, and *the
  rest of the corpus is still written*, so re-running after a fix is cheap.

⚠️ A failed capsule can leave a **half-written directory**: `capsule.yaml` and `capsule.interface.mlir`
present with **no `golden.yaml`**. A capsule with no answer key is worse than an absent one. After any
`[FAIL]`, check:

```bash
for d in merlin/contract/capsules/<target>/*/*/; do
  [ -f "$d/capsule.yaml" ] && [ ! -f "$d/golden.yaml" ] && echo "NO GOLDEN: $d"
done
```

## Goldens are untracked, by design

`golden.yaml`, the hidden capsule stores and `capsule.weights.safetensors` are gitignored — they are the
bench's answer keys. Consequences:

- a fresh clone or `git worktree` has **zero goldens**, and every capsule test then fails on a missing
  answer key, which reads as broken code rather than missing data;
- the wheel bundle needs its own exclusion, because `setup.py` copies `contract/` from disk;
- to move a working corpus to another machine, transfer the goldens out of band or regenerate them there.

## Commit step: sync the wheel bundle

`merlin/python/merlin/_data/` is a gitignored build artifact that must match the source trees, and a
pre-commit hook blocks the commit when it drifts:

```bash
.venv/bin/python build_tools/scripts/check_bundled_data.py --sync
```

Run it after regenerating, then commit. Verify no answer keys entered the bundle — it should print 0:

```bash
find merlin/python/merlin/_data -name golden.yaml | wc -l
```

## Per-target state (measured 2026-09-05)

| target | capsules | notes |
|---|---:|---|
| gemmini | 181 | reference corpus; conv, layers and app members all present |
| radiance | 182 | regenerated this date; SIMT, so 9 perf families skip on absent timing facts |
| atlas | 119 | `conv_geometry` 0/10 — no conv capsules minted yet |
| mx_gemmini | 49 | |
| saturn_opu | 47 | |
| saturn_opu_rvv | 46 | `TargetContractMissing` — its tool-generated package is absent |

Per-target inputs live in `merlin/contract/capsules/profiles/<target>{,.synth,.hidden}.yaml`; the derived
coverage requirement lives in `merlin/contract/capsules/conformance/<target>.yaml`.

## Checking what a target SHOULD produce

`corpus_synth.synthesize` is pure and instant, so you can ask what the spec would emit without running a
generation — useful for telling "not minted yet" apart from "cannot be expressed":

```python
import yaml
from merlin.targetgen.corpus_synth import synthesize
spec = yaml.safe_load(open("merlin/contract/capsules/conformance/<target>.yaml"))
out = synthesize(spec)
print(len(out["capsules"]), out["provenance"].get("unexpressable"))
```

An empty `unexpressable` with members missing from disk means the corpus is simply stale. A populated
`unexpressable` names what the target cannot express and why.

## Verifying coverage afterwards

```bash
MERLIN_MLC_DIR=<...> .venv/bin/python build_tools/scripts/check_conformance_coverage.py \
  --target <target> --json
```

Takes ~5 minutes per target (it re-derives from the capture set). ⚠️ Read the **totals**, not the
per-axis ratios: a target that *requires* fewer cells on an axis scores 100% on it while covering less.
Measured on this date, atlas `epilogue` read 2/2 against gemmini's 4/5 only because atlas requires two
stages where gemmini requires five, and atlas requires **zero** application shapes against gemmini's
twelve — so that axis cannot register a miss at all.
