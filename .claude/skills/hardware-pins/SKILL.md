---
name: hardware-pins
description: >-
  Which external hardware revision a result is about — pinning, verifying and recording RTL/toolchain
  versions in merlin. Use whenever you run RTL or a simulator, derive facts from an external checkout
  (Scala/Verilog/headers), build a binary for a board or sim, or write any artifact that claims a
  numerical or functional verdict. One registry: merlin/contract/hardware_pins.yaml.
---

# Hardware provenance (MANDATORY for anything that claims a verdict)

## The failure this prevents

A session certified a microkernel 31/31 against `saturn` and reported it as a hardware result. The
revision it actually built and ran against was `opu-int8 @ ea373800` — the only revision that contains the
outer-product unit at all. The revision named for the tapeout, `249340a6`, **does not contain that unit**:
no `OuterProductUnit.scala`, no `OuterProductSequencer.scala`, no `opuInsns`. Both are "saturn-vectors",
both had been checked out in the same tree, and the artifact recorded neither. The mistake surfaced only
because a person happened to name a commit in conversation.

**A result attributed to the wrong device is worse than no result.** It gets cited.

## What to do

Before producing anything that claims a verdict:

```python
from merlin.common import provenance as PROV

ver  = PROV.verify("saturn_opu_int8")          # declared vs actual; never mutates the checkout
prov = PROV.record(pins={"saturn_opu_int8": ver},
                   sources=[...],               # the files a derivation actually READ
                   artifacts={"sim": sim_path}) # binaries whose identity matters
```

Then put `prov` in the artifact (`provenance=` on the report / manifest / run record). `PROV.require(...)`
is the raising form — use it where proceeding under drift would be meaningless.

## The four rules

1. **Declare the revision in `merlin/contract/hardware_pins.yaml`.** Tracked and reviewed, so changing
   what a result is measured against is a diff someone sees. Full 40-char shas, quoted (an all-digit sha
   is valid hex and YAML reads it as a number, silently dropping leading zeros — the loader rejects this).

2. **Verify by content, not by name.** A pin lists `requires_paths` whose presence the work needs, and
   `forbids_paths` whose *absence* is the point. That is what catches the failure above: a checkout can be
   the right repo, on a plausible branch, and still be missing the unit. Branch names move; forks share
   them; the sha is the truth and the file list is the cross-check.

3. **Record what was READ, not just what was checked out.** `source_digest` hashes the actual bytes of the
   sources a derivation consumed. A dirty tree changes what gets emitted while leaving the commit looking
   correct, so the commit alone does not identify the result.

4. **Stamp binaries so they carry their own provenance.** A built image should print its stamp
   (`kernels.opu_cert.provenance_stamp`) and the report should compare the console's stamp against the
   expected one. A path and a timestamp cannot tell a stale ELF from a fresh one; a stamp can.

## Never

- **Never mutate someone's checkout to satisfy a pin.** Other people and sessions work in those trees on a
  shared host; silently moving HEAD is a worse failure than the drift. Verify and record.
- **Never treat `UNKNOWN` as "unchanged".** A fact that could not be read is drift, not agreement.
- **Never add to `build_tools/scripts/provenance_ratchet.txt`.** That list may only shrink; a new entry is
  a new unattributable result. Regenerate the artifact with provenance instead.
- **Never claim a verdict for hardware you did not verify you were on.** If the pin drifts materially
  (wrong commit, missing required path, forbidden path present), the report's `gaps` must say so and it
  must not read as certified.

## Enforcement

- `build_tools/scripts/check_provenance.py` — pre-commit (`--staged`), session Stop hook (`--stop-hook`),
  and `--verify-pins` for live checkout verification. It scans **untracked** reports under
  `out/artifacts/` too, because that is where the reports actually live; a version that only looked at
  tracked files passed while an unattributed certification sat on disk. **`--staged` runs that scan as
  well** — it used to skip it, so the pre-commit hook (the only caller that passes `--staged`) reported
  "0 verdict-claiming report(s) checked" against 86 for the bare invocation. The whole scan costs ~1.4 s.
  Any unreadable work list is a REFUSAL, never an empty one: a `git` that cannot run means nothing was
  examined, which is not the same as clean.
- Reports are recognised as verdict-claiming when they carry `certified` / `correct` / `passed` == true.

## Related

`artifact-layout` (where output goes), `merlin/common/provenance.py`,
`docs/design/incremental_target_evolution_opu.md` §8c.2 (the certification this came out of).
