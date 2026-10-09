# Opt-in native clocked termination patch

This patch corrects dependency preparation in the public Arc state lowering
pass. Its first visit must enqueue the clock and condition before the second
visit consumes their lowered values. The original implementation returns before
doing so; even a constant condition can fail with `value has not been lowered`.
The condition must also use the old state at the rising edge, before register
updates. Reading the new state can reject a legal response that empties a queue
on that edge, and miss an illegal response when the queue becomes valid on the
same edge. The patch prepares and samples the condition in `Phase::Old`, as
the public lowering already does for clocked trigger inputs. The clock remains
in `Phase::New` for edge detection. RTL, predicates and requested termination
status remain unchanged.

## Exact public selection

- Repository: [llvm/circt](https://github.com/llvm/circt).
- Revision: `0d63c41c9121106b01372aca2b60eb44ed4a6e87`
  (the independently resolved `firtool-1.161.0` tag).
- Source: `lib/Dialect/Arc/Transforms/LowerState.cpp`.
- Original source SHA256:
  `9d1f41f470479e414e361b0da55968a12bb039956c9572c463d1603b1483d10f`.
- Selected public SDK asset: `circt-full-shared-linux-x64.tar.gz`,
  release `firtool-1.161.0`, SHA256
  `b9ae9472d8cd6c67f6508807a6d03bb9e1917bee3227ed4ee0732008dd12afb0`.

The independently downloaded SDK's `ArcConstants.h` and `ArcOps.td` byte-match
that public revision. Their SHA256 values are respectively
`1ac5774ad81409d6f6acbe2b22d02b00359882967d21e72311667158a84e787a` and
`c3f521523c69e49b059d79637e4a78bbd546248d518748213bafef70bdbeefd5`.
The SDK's `SimOps.td` also byte-matches that public revision, SHA256
`574b817428c17e5eb2e24d312dad0473d2df1b7394d77c58c32dcb8245f54cd1`;
it defines condition sampling on the clock's rising edge.
The dependency-only diagnostic plugin was rebuilt with its actual compilation
header roster and produced identical bytes. The combined sampling fix was
compiled with those selected SDK headers pinned; applying this patch reproduces
the compiled source after removing its explicit registration and naming wrapper.
These checks establish the recorded source/header/build relationship;
they do not establish a reproduced origin for every SDK binary or its full
dynamic dependency closure.

## Replay

1. Select a separate public checkout at the exact revision. Verify its source
   hash and closed build inputs before modification. Keep it outside compiler
   author inputs and preserve the unmodified native failure.
2. Run `git apply --check --unidiff-zero` with
   `circt-state-dependencies.patch` in that exact source checkout, then
   explicitly apply it with `git apply --unidiff-zero` and rebuild the selected
   native tool. The zero-context hunk requires the original source hash check;
   it does not authorize fuzzy application to another revision. Alternatively,
   an independently recorded side-by-side pass build can select the changed pass
   explicitly; the plugin registration wrapper is not part of this patch.
   The patch includes a native lowering regression at
   `test/Dialect/Arc/clocked-terminate-old-state.mlir`; its `FileCheck` checks
   the condition read and requested stop before the queue-state write.
3. Run the original source through native parsing and lowering. Retain actual
   commands, tools, inputs, generated products and full output membership.
   Recheck the same original memory latency, read enable, write mask and clock
   controls, and both true and false termination conditions at clock edges.
   Include same-edge state transitions in both directions: consuming an already
   valid queue entry must preserve the old validity for the assertion, while
   a newly valid entry must not hide a previously invalid response. Check the
   complete declared outputs, raw requested stop and absence of a rising edge.
4. Observe the selected runtime's termination request through its actual public
   ABI in addition to original numeric outputs and process outcome. Ordinary
   completion, a successful requested stop and a failed requested stop are
   distinct. Raw request status is an observation, not a success verdict.

No discovery, patch application, build or runtime selection happens when Merlin
is imported. This recipe grants no tool, plugin, runtime or author authority.

## Retained limits

Actual private diagnostics preserved the stock failure and demonstrated native
lowering after the dependency fix, including complete selected-source LLVM/state
preparation. Native memory and separate-clock controls retained their original
full numeric outputs. A true-termination negative initially returned normally
with matching numeric rows; observing the public context field exposed the
failure request. This is why numeric agreement and exit zero are insufficient.

Separate clocked-state controls exposed both the false rejection and missed
failure under new-state sampling. With old-state sampling, the sampling and
stop outcomes match for all four controls:

| Original control | Required raw request | Dependency-only pass | Old-state pass |
| --- | ---: | ---: | ---: |
| Consume an already valid entry | 0 | 2, incorrect early stop | 0 |
| Respond while the entry first becomes valid | 2 | 0, missed failure | 2 |
| Respond with an empty queue | 2 | 2 | 2 |
| True condition without a rising edge | 0 | 0 | 0 |

The legal-consume and no-rising-edge runs preserve their complete original
output rows. Both required failure stops truncate the non-stopping output roster;
those failed processes and partial rows remain retained. Separate numeric-only
observations preserve the complete original rows, without qualifying the stopped
runs numerically. No original full-output requirement is removed. These small
controls demonstrate the tested sampling behavior only; they do not establish
whole-platform execution or assertion support in other lowering routes.

Ordinary assertions still block runtime admission: the selected default SV
route discards assertion bodies, while native `--lower-to-core` refuses
`verif.clocked_assert`. Those failed controls remain part of the original
qualification denominator. No unsupported assertion, asynchronous reset,
blackbox behavior, program loading, whole-RTL equivalence, physical runtime or
timer role is qualified by this patch. Built plugins, selected target RTL,
private control answers and execution receipts are intentionally absent here.
