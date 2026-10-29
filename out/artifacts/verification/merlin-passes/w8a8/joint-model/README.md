# Explicit W8A8 numerical context

Qualified named instance, October22,2026. This packet discharges the old numerical
premises for a named arithmetic family; it does not claim complete W8A8 or native execution.
Continue the [concrete-context assignment](../../progress/planner/assignments/w8a8-concrete-context/README.md)
and [eight-day goal](../../progress/planner/eight-day-paper/README.md).
`qualification-v1.json` indexes final evidence. No source has been staged, committed,
integrated or pushed. Continue attention1533 after this milestone.

## Proved construction

`W8A8IEEEPrimitives.model fallback` explicitly implements the constrained default-key
operations with the pinned executable unpacked floating model: binary32 multiplication,
addition, subtraction, division and reciprocal/rounded-sqrt rsqrt; bitwise absolute value;
half-even rounding with signed zero; signed integer conversion; binary32-to-binary64
widening, source binary64 division and binary64-to-binary32 narrowing. Other operations
come from the supplied fallback. The standalone witness uses NaN/poison sentinels only
outside the constrained slice; this is not a full native primitive implementation.

The following compile together without additional numerical-law hypotheses:

- `W8A8IEEELaws.finite_laws`: all seven original source-law fields.
- `W8A8IEEELaws.dynamic_laws`: all three additional dynamic fields, including source laws.
- `W8A8IEEEPrimitives.agreement`: all nine original arithmetic Agreement fields.
- `PolicyW8A8IEEEContext.inhabited`: an actual context satisfying the original `ContextOk`.
- `PolicyW8A8IEEEQualified.check_sound`: complete/suffix checker soundness at that family.
- `PolicyW8A8IEEEQualified.check_eq_numerical`: equality of its Boolean checker and the
  existing conditional checker; no replacement checker formula is merely asserted equal.
- `PolicyW8A8IEEEQualified.domain_inhabited`: a joint model/input witness for checked bounds.
- `PolicyW8A8IEEEPrefix.entry_bound`: actual prefix and requested bounds in the named family.

Proofs include precision-parametric mantissa/rounding bounds, signed32/64 conversion,
full declared source narrowing interval, exact widening of normal scales, finite dynamic
division and range of the source binary64 reciprocal. Quant-ext's direct-f32 versus
f64-then-f32 reciprocal identity is a separate owned theorem; it is not duplicated here.
The frontend worker owns the constructor adapter and its two source numerical laws.

`Driver.W8A8IEEE` and `W8A8IEEEMain.lean` retain the original generic reports as nested evidence
and expose fresh specialization roots. `ieee_{evidence,policy}.py` bind actual exports,
contracts, source/runtime identities and fresh theorem audits; replay regenerates checks.
The legacy generic reports still label their premises assumed. The outer report separately
states that seven-plus-three laws and nine Agreement fields are proved for this family.

All broader legacy flags remain false. Logical context existence is now proved, while
native fidelity, reviewed policy/source-interface admission, selected-capture binding,
successful attention execution and whole-pass enforcement remain separate.

## Development evidence and corrected diagnoses

New private profile: `out/build/lean-w8a8-joint-model/repo`. All earlier qualified profiles
and the complete68-file delivery remain preserved. Construction/specialization/driver logs
are retained here. Initial full guard: **10,168 theorem audits, zero violations**.
Kernel fixtures include the joint context/input witness, scale endpoints, tiny reciprocal
overflow exclusion, signed32/64 extrema, half-even ties, signed zero and overflow controls.
Only the standard `propext`, `Classical.choice`, `Quot.sound` axioms occur in printed roots.

| Run | Result | Interpretation |
| --- | --- | --- |
| development-v1 |20passed,4failed|Two alleged positive fixtures actually used unsupported source metadata; a receipt hit integrity refusal; two kernel assertions incorrectly expected no axioms|
| development-v2 |6passed,2failed|Kernel assertion corrections, replay and diagnostic tests pass; new literal candidates still used unsupported metadata|
| calibration-v3 |4passed,2failed|Reports establish that supported string metadata accepts both runtime and literal scales. Two assertions incorrectly expected runtime-scale refusal; corrected afterward|
| dynamic-v4 |4passed|Actual complete dynamic boundary accepts; compiler-valid rounding, constant and returned-value mutations all refuse|

The original captured fixture uses `input_dtype = i8` as a TypeAttr. The declared source
adapter accepts absent metadata or the strings `i8`/`int8`, so the original is retained as
an explicit refusal. With string metadata, runtime scales legitimately qualify in the
reused-byte contraction: this boundary does not require quantizing its dequantized floats
again. Literal-scale cases also demand and check a real dequantized-value bound. No checker,
source contract, production selection or numerical premise was weakened for these tests.

The original integrity refusal remains unresolved: later rechecks of852 records in each
saved report match, which does not validate the failed attempt. See
`integrity-development-v1.json`. Additive `ieee_runtime.py` now retains the first mismatching
expected/actual reference and stat observations, then rejects immediately. It does not
retry-to-accept; two tests cover actual changed bytes and a transient first observation.
Historical `interpreted.py` is unchanged. The new diagnostic wrapper applies after the
existing initial capture checks, which retain their original rejection behavior.

Native development probes match all tested non-NaN bits and retain NaN sign/payload
differences. No universal native-fidelity claim follows. Package-doc and artifact-layout
gates pass; the structure gate retains the unrelated `zephyr_model.py`3557>3537 size debt.

## Current qualification and continuation

The final serialized guard passed: **10,168 theorem audits, zero violations**, log
`guard-final-v1.log`. `freeze-v1.json` seals1,372records,21newownedfiles and the852-file
runtime `f307d3ad4a582f2d725ba5ee28f8817cd2ddfab5c1d0d207f70a6b7550f1db38`.
Eight fresh proof-root axiom probes and elaborated statements are retained. All older68
sources match. The65-case final affected suite **passed with zero failures/errors/skips
in745.61seconds**, including retained RMSnorm/rotary counterexamples (`qualified-v2.log/.xml`,
collection in `collection-final-v1.log`). Twenty honest changed-boundary receipts are
catalogued separately from deliberately forged test evidence: five complete positives and
15complete refusals, with12positive requested-prefix checks.

Eight curated boundaries replay in separate fresh Python processes, including all five
complete positives, unsupported metadata and valid return mutation refusals, and a prefix
case. See `curated-replays-v1/SUMMARY.json`. Actual retained TinyLlama independently replays
under this same frozen identity: source/reference1060/1087operations, all163bounds including
158finite source dequantizers, joint logical-context/input witness, and the same next
refusal1533. Full numerical execution and archive-domain admission remain false.

`identity-post-qualification-v1.json` checks all1,372current freeze records and all four
prior profiles (5,783unique paths), zero mismatches. Five retained native comparison reports
have zero tested non-NaN bit differences and **17NaN sign/payload bit differences**; see
`native-comparisons-v1/SUMMARY.json`. These are observations, not universal fidelity proofs.
All qualification/replay sessions are terminal. This is a qualified logical instance and
specialization; whole-pass/native/review/capture/composition obligations are still open.

The first frozen attempt (`qualified-v1`) hit Errno122 while writing test files to `/tmp`:
31failures/34errors in9.86seconds, not a qualification or a model counterexample. All1,372
freeze records were rechecked unchanged. Its raw logs/XML and `quota-attempt-v1.json` remain.
Four completed UNFROZEN development temp trees were copied and every byte verified under
`development-temporaries-v1/`; their old names remain symlinks. The relocation record covers
1,170entries. No frozen or foreign file was relocated. Current `TMPDIR` and pytest basetemp
are under `out/build/lean-w8a8-joint-model/`, on the project filesystem.

`implementation-v1.patch` contains21newownedfiles. `combined-w8a8-owned-v3.patch` contains
the complete89-file additive delivery, preserving the previous68files. Both pass `git apply
--check` in empty private directories; neither was applied or staged. Exact records/snapshots
are `owned-files-v1.json`, `combined-owned-files-v3.json`, `source-v1/` and
`source-closure-v3/`. `identity-pre-receipts-v1.json` rechecks5,783unique files across all five
historical/current profiles with zero mismatches; the postqualification check above repeats
and passes that inventory.

After sealing this named instance and additive source delivery, continue the assigned
ordered floating accumulation/attention1533 work. Preserve negative-infinity masks,
attained maxima, exp underflow and positive-denominator obligations explicitly. Current
fallback operations do not establish sin/cos/exp laws. Independent policy review and capture
selection do not prevent continuing those owned proofs, but their absence cannot be hidden
in a final correctness claim.
