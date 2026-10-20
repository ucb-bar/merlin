---
title: Recover missing public archive count observations
kind: guide
status: current
owner: timing
last_verified: 2026-10-20
related: [spike_timing, phase2_archive_screen]
code_refs:
  - packages/merlin-timing/src/merlin_timing/archive.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/annotated_spike_evidence.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/_counter_symbol_aliases.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/archive_spike_runtime.py
---

# Recover original archive counts

A missing count report does not necessarily require another hardware-cycle
measurement. An original public archive may already retain a passing paid label,
its exact consumed L3 ELF and embedded-input caller, but lack an independent
counter-window review and native count observation. Recover those missing items
prospectively; preserve the archive, paid label, original reports and source-group
splits.

First reopen the archive with `archive.inspect_archive`. Select the indexed L3
ELF and caller, rather than a later file at the same generated path or the L2
ELF. Verify every copied byte against the indexed artifact. The original selected
counter instruction, clock, entry/completion functions and whole-call boundary
remain required.

Use the original disassembler and counter-window reviewer. If disassembly names
an alias instead of the selected entry, the existing
`_counter_symbol_aliases.window` API can issue its separately versioned V2 review
only after proving both names have the same positive executable function extent
in the actual ELF. It then reruns the original counter-order checks. A name
suffix, nearby address or saved success flag supplies no such proof. Preserve
the original V1 refusal and retain the fresh review's disassembly, tool, source
and caller identities.

Freeze a fresh `ArchiveSpikeRuntime` from the original functional command and
selected observer, ordered libraries, dependencies, model and counter selection.
Run plain and annotated configurations on the **same original L3 ELF** using
the original evidence runner. Both executions must complete successfully and
their consoles must agree. Require complete observed features and exact
ELF/input/scope/model joins. Enforce the operator's resource envelope, shared CPU
leases and instruction prohibitions before execution. This offline archive
operation does not replace the boxed count observer used by queued candidates.

Write a new companion binding. Change only the previously missing per-record
counter review, and derive a new plan with `archive.prepare`. Keep all original
case and label fields unchanged. Rebind the companion attachment to that plan,
retain every original report envelope byte-for-byte, and add the newly produced
report. `archive.attach` independently checks its exact scope and report identity.
Neither the new review nor the attachment issues a timing qualification owner.

The existing installed metadata workflow can replay the result:

```sh
spike-timing archive-prepare \
  --archive "$ARCHIVE" --archive-sha256 "$ARCHIVE_SHA256" \
  --binding "$RECOVERED_BINDING" --output "$WORK/recovered-plan.json"
spike-timing archive-attach \
  --plan "$WORK/recovered-plan.json" --reports "$RECOVERED_OBSERVATIONS" \
  --output "$WORK/recovered-dataset.json"
spike-timing validate \
  --dataset "$WORK/recovered-dataset.json" --prediction-kind event_timing \
  --output "$WORK/recovered-validation.json"
```

Use fresh output paths under the executing task. These commands perform metadata
intake and validation; they launch no simulator. A subsequent normal selection
must still reconstruct its original live Count and native-semantics owners and
pass the complete raw-activity and source-domain gates. Select companion paths
before issuance; do not mutate an already issued owner.

The 2026-10-20 recovery followed this route for all ten absent original public
reports: 473 original envelopes remained unchanged, and the companion contains
483 reports for the same 483 original archive rows. With the separate 159-report
supplement, the union has 642 reports, not 642 distinct capsules. The fixed
`dc413` primary archive passed its original statistical and raw held gates.
Those ten recovered programs all contain one flush and at most one activated
store; they do not close the separate zero-flush or held activated-store-33 gap.
No coefficients, paid labels, held splits, numerical policies or timing gates
changed.

## Export retained public labels before requesting new measurements

A completed qualification may retain usable capsule-result and invocation files
without having written a timing archive or timing-status file. Use the original
`phase1.capsule_timing_archive.audit`, `training_index` and `training_archive`
APIs on that run before concluding that paid measurements are absent. Export to
an ordinary fresh destination in the executing task; never call `finalize_run`
on a read-only Phase 1 run. The training exporter excludes hidden records and
retains executed repeats, original artifact roles, label windows and producer
identities. Bound the resulting file and total output size before export.

Count observations can be reused for a new original label only after reopening
the full execution correspondence: exact consumed ELF, embedded-input caller,
source, hardware, counter scope, runtime and complete original report. Preserve
the original count receipt and identify the reuse explicitly. Equal compiler
names, geometries or candidate hashes alone are insufficient. Observe only
executions that lack such evidence; the label must remain the original paid
measurement rather than a native estimate.

The final V26 qualification supplied another 146 eligible public labels. Two
count observations were made freshly (one control and one previously unseen
ELF); 144 report envelopes were reused after exact checks. All 146 execute zero
flushes. Of their 146 paid labels, 145 exactly repeat already retained execution
identities and cycle values. The complete ledger therefore contains 788 records
across the primary 483, supplement 159 and final 146, for 454 unique execution
identities. Repeated labels do not create independent schedule comparisons.

The unchanged `dc413` model still passed the primary archive's complete gates.
The final 146 alone failed: held p95 error was 0.3516642547 against 0.30, and
there were zero eligible held variant pairs against the required 20 across
three mechanism families. A held per-column readout also exceeded its TRAIN
host-store/write/cache activity ranges. These records provide no new eligible
TRAIN activated-store-33 witness. Retain these refusals; exporting more repeat
labels does not qualify the zero-flush domain or justify fitting against held
errors.
