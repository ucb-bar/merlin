# Correction: the seven-case EL3 verdict gap is confounded

The previous interpretation of the 135-versus-128 L2 count as broader compiler
capability is withdrawn. The recorded historical counts are accurate, but
their acceptance rules differ by the authored ABI declaration. All source
snapshots, historical grades and the immutable CSV remain unchanged.

For all six residual-convolution, residual-linear and sibling-feed chain
cases, the retained named Spike output payloads are identical across the
three best source-coherent candidates. The golden names only final `Y0`.
Readback also contains `Yc`, or `G`, `U` and `Hm`. The ordinary pointer-ABI
path requires matching output-name sets, rejecting those intermediates. EL3's
`whole_program` declaration selects semantic-golden comparison that checks
expected output names and accepts the extra observed values. This accounts
for the fail/fail/pass pattern without an arithmetic-output difference.

For these `bounded_int` policies, the rejecting name-set guard is in the
frozen `capsule_runner.py:1224`; the equivalent exact-integer guard is at1209.
The route selection is at4889–4897, direct comparison at5805 and ordinary
comparison at5829–5832. The generic missing non-identity `CONFIG_ST` scale
hint was not a valid explanation: edge cases explicitly use identity scale,
and model-size output bytes match as well.

The seventh distinct-scale readout case was `ADMISSION_DEFERRED` in EL1 and
EL2 (`snapshot owner stopped new admission`), with no L2 or numerical verdict.
It is not an observed unsupported feature or failed implementation.

This audit does not automatically promote the baseline results to L2 passes.
Completion-order checks are downstream of numerical success and were skipped
for the baseline failures. EL3's completion-order record is also deferred to
certification. Complete common qualification remains necessary. Five harness
receipt hashes match across arms; the sixth uses procedural versus materialized
harnesses, and custody limitations are recorded. Input interfaces and all
named observed output bytes match for all six cases.

Detailed proof and independent review are under the result directory:

- `reviews/chain-verdict-audit.md` and `.json`: case, output, interface, harness
  and source identities, hashes and frozen code references.
- `reviews/chain-comparator-impact.md`: downstream obligations, differential
  historical feedback and the common-evaluator remedy.

The main figure is now a two-panel descriptive draft at
`.tmp/paper-results-20261028/paper-figure-el123/el123_fixed_budget.pdf` with a
prominent comparison-confound notice. The failed-self-check panel is removed:
test mixes and exposure differ, and the numerical acceptance defect can also
contaminate that metric. Counts remain in the CSV and derived JSON. The former
three-panel figure is archived alongside it. The draft cannot support the
former positive correctness claim until the frozen candidates are evaluated
under consistent output and downstream qualification rules.

Token accounting remains valid. Full four-hour totals are111.123M /120.624M /
105.396M for EL1/EL2/EL3. Fresh input plus output is3.006M /2.440M /2.849M;
EL3 is5.2% lower thanEL1 on both usage definitions, but16.8% higher thanEL2
on fresh input plus output. Roughly98% of input tokens are cached context.
There is no duplicate-owned-response accounting defect in the current CSV.
The older cumulative stream counters were excluded. The modest full-window
differences concern continued fixed-budget work with large repeated contexts,
not stop-on-target effort or demonstrated billing savings.

A fair next comparison preserves all authored sources and uses one contract
for visible outputs, common admission, and the entire remaining validation
ladder. Preserve historical verdicts and timestamps alongside requalification
records; do not backdate newly acquired evidence. Since historical feedback
affected subsequent search, regrading alone cannot reconstruct a clean causal
authoring ablation. No compiler, model, simulator, framework fix or new
experiment was launched for this correction.
