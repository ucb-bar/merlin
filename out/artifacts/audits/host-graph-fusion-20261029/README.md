# Actual host graph fusion qualification

Work is on `feat/pass-verif-integration`, based on
`c95fb101271c93dabcf7ace72c509bf46398977c`. This adds an explicit compiler gate for
the existing host dispatch fusion transformation. Existing numerical approvals,
frozen semantics, locks and the five checks selected by `all` remain unchanged.
Implementation is committed in `1fe7c728a`; the two test repairs are in
`6796c8608`. The [qualification manifest](qualification.json) binds the exact
tested source bytes and compiler receipts. These changes are local and unpushed.

The earlier graph theorem checked a proposed logical region. The new gate reads
the actual emitted helper bodies, binds their actual call arguments and results,
and checks the complete public entry. The producer report cannot supply a proof.
Both xDSL and textual compiler routes share the boundary before preprocessing.
The compiler returns products only after its canonical worker accepts; failure
quarantines products. The standalone API makes no compiler-enforcement claim.

## Paper wording

Merlin validates supported host dispatch fusion against the actual input and
output IR. The validator proves preservation of every public result under a
typed pure tensor model, including values used outside the fused region.

Use **Host dispatch fusion** as the table label. Its compact description is
**Actual helper bodies and complete public outputs**. This result concerns
dispatch grouping. It does not claim scalar-loop fusion, eliminated physical
intermediates, improved performance or verified machine code.

## Theorem and coverage mapping

| Obligation | Established result |
| --- | --- |
| `Merlin.HostGraphFusion.check_sound` | Complete ordered public results agree for every well-typed initial state and common typed total pure-action interpretation |
| `Merlin.GraphRegions.checkFusion_preserves_observers` | Existing region theorem retains all live outputs in prefix/suffix context |
| `ActualHostFusion.accepted` | Kernel reduction checks the independently projected source, target, actual call body and interface |
| `ActualHostFusion.complete` | Closed instance of the complete-entry theorem for that actual boundary |
| Canonical compiler worker | Fresh proof and axiom audit, actual byte binding, gate join and rejected-product quarantine |
| Native comparisons | Finite evidence against unfused execution and independent wrapping-integer references, separate from the theorem |

The profile has one public single-block entry, positive static unencoded
byte-addressable integer tensors, identity-indexed parallel generics, initialized
public inputs and unread destination operands. The closed scalar subset admits
integer constants, add/subtract/multiply, bitwise operations, signed min/max,
truncation and extension without poison modifiers. A single target call contains
2 through 128 stages. Other entry-level calls contain at most one stage and the
source has no existing multi-stage entry-level call. Every ordered public result and the
complete expanded primitive graph must match. Live-in port ordering may differ
only with checked permutation and actual argument/body bindings.

Syntax guards are 1 MiB ASCII IR per module, 512 projected nodes, 4096 expanded
operations and call depth below 32. Shape elements are not enumerated. Private
definitions are syntax checked, but equivalence of unused private symbols is not
claimed. The capture begins immediately before fusion and does not certify
earlier layout rewrites or subsequent compiler passes.

Trusted components are xDSL parsing, typed SSA projection, canonical scalar-body
hashing and collision resistance, normal private-call semantics, common total
deterministic typed tensor actions, initialized inputs, and unobservable
successful logical allocation. Action premises include read locality, frame and
type preservation. Lean and its standard imported environment remain trusted.
The transitive audit permits only `propext`, `Quot.sound` and `Classical.choice`.
It admits neither native reflection nor `sorry`.

Concrete MLIR arithmetic correspondence, allocation failure, physical buffer
behavior, external effects, floating-point, reductions, dynamic or encoded
layouts, recursive/external reachable calls and broader fusion patterns remain
outside this result. It does not prove success/error equivalence for a failing
allocator. The [ledger](../../../../merlin/lean/LEDGER.md) and
[compiler guide](../../../../docs/guides/lean_compiler_stack.md#logical-host-graph-fusion)
record the durable contract.

## Qualification

The retained final host fusion suite passes all **20 cases without skips** in
`qualification-v5.log/xml`. Each accepted proof audits 117 theorems with no
violations and binds 474 source references. The existing graph suites pass all
71 cases after two stale test mocks were refreshed. These counts overlap with
the broader regression run and must not be added together. All 33 shared-worker
and audit-freshness regressions pass in `worker-regressions.log/xml`.
Both retained ordinary compiler receipts also pass separate fresh-process
replay in `compiler-replay-0/` and `compiler-replay-1/`. Their 474 source references
and all proof/tool/import artifacts were rehashed against the tested checkout.
All 235 existing pinned statement hashes and frozen file hashes still match.

The broader v4 run was interrupted after 125 passes and the two stale mock
failures. Thirty non-corpus compiler-gate tests completed, along with six of the
40 representative DCE/CSE corpus cases. The remaining corpus cases are **not
qualified by this run**. That sweep is outside the selected graph work; its
partial results and interrupt remain in the log/XML. The repaired graph tests
and shared-worker checks have separate completed runs.

The final suite covers genuine emitted graphs at i8/i16/i32/i64, a shape exceeding
a trillion elements without enumeration, prefix/suffix context and public
producer fanout. It includes fresh Python-process replay, five valid target
corruptions rejected by Lean, domain refusals, candidate/report/input mutation,
copied-token and missing-tool controls, canonical compiler rejection and native
execution on both compiler routes. Both native arms compare every public output
against unfused execution and an independent integer reference, including wrap
and high-bit inputs. The quarantine control deliberately injects a wrong target
at the real submission boundary while retaining the canonical worker.

## Reproduce

Use the existing Lean 4.35.0 SDK, project `.venv`, and installed torch-mlir compiler
environment. No dependency is fetched by the gate. Keep the venv executable
spellings rather than resolving their symlinks. This SDK's Clang headers need
the explicit include path below. The gate freshly builds its four-module proof
closure; `CheckerTools` also requires the existing main checker build.

```sh
export MERLIN_LEAN_REQUIRED=1
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/toolchain/lean-4.35.0-linux/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_CLANG="$MERLIN_LEAN_SDK_BIN/clang"
export C_INCLUDE_PATH="$MERLIN_LEAN_SDK_BIN/../include/clang"
export PATH="$MERLIN_LEAN_SDK_BIN:$PATH"

(cd merlin/lean && lake build MerlinVerify merlin-lean merlin-lean-audit)

.venv/bin/python -m pytest -q -ra --tb=short \
  -o tmp_path_retention_policy=all \
  merlin/tests/ir/test_compiler_host_graph_fusion.py \
  --basetemp=out/build/host-graph-fusion-replay/qualification

.venv/bin/python -m pytest -q -ra --tb=short \
  -k 'not representative_kernels_lower_identically_under_the_gate' \
  merlin/tests/ir/test_host_graph_optimizations.py \
  merlin/tests/ir/test_host_graph_policy.py \
  merlin/tests/ir/test_lean_pass_gate.py \
  merlin/tests/ir/test_speculative_gate.py \
  merlin/tests/ir/test_pass_gate_audit_freshness.py \
  --basetemp=out/build/host-graph-fusion-replay/regressions
```

Use a new destination for every run. Pytest normally deletes passed test trees,
so explicit retention is necessary for proof and negative-control evidence.
The retained final tree is `out/build/host-graph-fusion-20261029/pytest/v5/`.

The contextual native example computes `a = x + y`, then `b = a * y`,
`c = b + x`, `d = b - a`, and `e = c + d`, returning `(e, b, a)`.
The actual producer groups the middle three stages into a private helper that
returns `(b, c, d)`. The suffix and public return retain their exact SSA bindings.
Each compiler verdict records `accepted`, `fused_stages=3`,
`compiler_enforcement=true` and `native_fidelity=false`. Arithmetic, helper
argument/result, public return and suffix mutations each fail the Lean check.

For a retained compiler `proof/REPORT.json`, independent replay is available
through `compiler_host_graph_fusion.replay(report_path, new_destination,
lean=selected_executable)`. Replay validates the saved input bytes and performs
a new projection, proof build and audit using current sources. Saved status
does not grant authority. Reports retain source, ledger, input, query, import
and tool hashes. Changing bound files requires a fresh run.

## Preserved failures and next work

`qualification-v1.log` retains a valid contextual fusion refusal caused by the
old exact live-in ordering requirement and two incorrect test keyword failures.
The additive theorem now checks live-in permutation while the projector retains
actual argument bindings. Existing `GraphRegions.checkFusion` is unchanged.
The test now uses the real `textual` compiler option. Initial negative contextual
cases do not count as standalone evidence until the valid baseline succeeds.

`qualification-v2.log` retains two native runtime compilation failures caused by
missing Clang headers. Selecting the shipped headers resolves them;
`native-v3.log` records both native routes passing. `qualification-v4.log`
retains the broader run before two old graph mocks were refreshed. One stubbed
LLVM lowering without emitting newly required copy-removal reports. That
hook-only test now disables unrelated automatic copies. The other mock failed
on the current `connected` keyword before reaching its intended conformance
failure; it now accepts the actual call signature. Real native comparisons keep
automatic copies enabled.

Repository layout/access tests report 169 passes, one baseline dependency failure
and three explicit skips. The failure is the two existing optional-package imports
in `src/merlin/targetgen/package_runtime.py`. Skips are unavailable bubblewrap
user namespaces, absent release wheels, and missing `setuptools`. Actual release
wheel/install qualification was not performed. Whole-repository checks retain
16 structure findings, 44 regex findings, seven target-name findings and 48
artifact-layout findings. Their cited files are unchanged from the base commit;
no guard or allowlist was relaxed. Docs, doc paths, changed Python lint and
whitespace checks pass. The curated staged artifact check is separate from the
failing whole-repository scan.

The historical `test_lean_graph_regions.py` and
`test_lean_graph_regions_source.py` capture suites were not run. Their fixed
`/home/jeremy/Projects/merlin-graph-proof-intake` producer checkout and
`/home/jeremy/Projects/model2MLIR/.venv/bin/python` oracle do not exist here.
This qualification instead exercises the current branch's real host graph
producer and installed compiler on both routes. It does not refresh those older
pinned capture receipts. The native MLIR execution-engine helper library and
standalone `mlir-opt` are also not part of this qualification; native comparisons
use the installed torch-mlir route and selected Clang.

The next engineering extensions are multiple fused regions with exact adjacent
joins, composition with layout preprocessing, and broader indexing and reduction
profiles. A concrete arithmetic source bridge would reduce the current primitive
correspondence premise; reuse the existing graph source-bridge libraries when
extending them to this actual-call profile. Each extension needs actual compiler boundaries,
complete-output proofs and mutation tests. W8A8 policy, Q/DQ/MACC approval,
DCE/CSE expansion and upstream MLIR correctness are separate.

Only this README and the compact qualification manifest are tracked. Raw logs,
queries, object trees, failed evidence and native products remain under this
audit directory and `out/build/host-graph-fusion-20261029/`. Copy those trees to
transfer the original local evidence, or regenerate from the recorded sources.
