# MACC producer legality review packet (unapplied)

The curated patches and compact JSON evidence in this directory are tracked.
The reproduction scripts below remain in local task scratch; they are not
distributed with this packet. Existing source fixtures and gate regressions are
in `merlin/tests/ir/lean_bf16_f32acc_cases.py`, `test_compiler_macc.py` and
`test_macc_ssa_joins.py`. The source file's hash in `manifest.json`, rather than
the branch name, identifies the patch base.

`producer-legality.patch` and `producer-legality-reject.patch` are alternative
proposed changes to `src/merlin/llvmlower/passes_xdsl.py`. Neither has been applied. `manifest.json`
binds the reviewed source and candidate SHA256 values. The numerical contract
remains a draft; this packet creates no independent approval or admission.

The proposed producer checks the actual multiply/add/yield dataflow, matching
operand/result half formats, positive-zero initialization, projected maps,
ordinary source flags and the supported direct single-block function scope.
It preserves named matmul maps. The unchanged-source alternative retains the
original IR outside this checked profile, establishing **zero MACC policy
coverage for that module**. The rejection alternative explicitly rejects the
requested unsupported policy before any mutation. The check is atomic across all half-contraction candidates in
a module so an unsupported site cannot be silently omitted from a partially
changed boundary's conservative denominator.

Human review must decide whether to install these producer legality changes
and choose unchanged-source fallback or explicit rejection. The change can reduce optimization coverage: mixed-format inputs,
nonzero initialization, unsupported attributes/maps/nested sites and mixtures
of supported/unsupported half contractions keep the source route. An independent
review of the numerical reference and its fusion/format assumptions is a
separate decision; approval of this producer patch would not approve them.

Evidence on the final candidate:

- `git apply --check` passes for both alternative patch files.
- Sixteen cases in `regressions.json` exercise actual production output and the
  unapplied candidate function. Standard named/generic bf16/f16 forms conform;
  transposed named forms become conforming. Nonzero/negative-zero initialization,
  wrong generic multiply/add operands, f32 inputs and a mixed supported/unsupported
  module retain byte-identical normalized source under the fallback proposal
  (zero policy coverage). The rejection alternative refuses each before mutation.
- Six small JIT cases in `native-regressions.json` compare actual source,
  production and candidate output for both formats: wrong multiply, nonzero
  initialization and transposed maps. Production differs from source; the
  proposal matches each source example. Values are finite exactly representable
  integers, so these examples establish concrete bugs without relying on a new
  accumulation-accuracy or rounding permission.
- Native and structural evidence bind the same candidate/source hashes.

Reproduce from the repository root with the existing explicitly selected
Lean/MLIR environment and one bounded CPU lane:

```sh
source out/build/pass-verif-integration-20261028/environment.sh
export TMPDIR="$PWD/out/build/macc-engineering-20261029/tmp"
export LEAN_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 30 nice -n 19 out/build/pass-verif-integration-20261028/config-runtime/bin/python \
  out/build/macc-engineering-20261029/producer-review/regressions.py
taskset -c 30 nice -n 19 out/build/pass-verif-integration-20261028/config-runtime/bin/python \
  out/build/macc-engineering-20261029/producer-review/native_regressions.py
```

`prepare.py` derives the review candidate and unified patch from the exact
installed producer function; it never writes the production source. The scripts
execute a scratch candidate for review only. They are not a runtime plugin,
fallback permission or acceptance authority.

Related checker engineering is separate: the live implementation now preserves
one shared SSA tensor as one wrapper input, with proved uniqueness, sharing and
frozen-environment binding. Both formats, internal callers and dependent sites
have actual checker tests. An additional initialization-aware wrapper change
now binds captured target operands back to their actual source identities and
retains copied initialization producers; 24 direct checker tests cover both
formats, calls, dependent sites and initialization used as left/right/both
contraction operands, including broken-join controls. Final live checks and
fresh replay are recorded separately from this unapplied producer packet.
Both final initialization-aware live checks and independently fresh replay now
passed for bf16/f16 (two cases, 97.70s). The initialization coverage gap is closed
for this checked source profile; it is not a remaining human-review blocker.
