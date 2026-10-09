# Opt-in native termination dependency patch

This patch corrects dependency preparation in the public Arc state lowering
pass. Its first visit must enqueue the clock and condition before the second
visit consumes their lowered values. The original implementation returns before
doing so; even a constant condition can fail with `value has not been lowered`.
The patch changes only that preparation. It does not alter RTL, predicates,
clock edges, memory behavior or requested termination status.

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
An explicit diagnostic plugin built from the affected public source against the
SDK was rebuilt with its actual compilation header roster and produced identical
bytes. These checks establish the recorded source/header/build relationship;
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
3. Run the original source through native parsing and lowering. Retain actual
   commands, tools, inputs, generated products and full output membership.
   Recheck the same original memory latency, read enable, write mask and clock
   controls, and both true and false termination conditions at clock edges.
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

Ordinary assertions still block runtime admission: the selected default SV
route discards assertion bodies, while native `--lower-to-core` refuses
`verif.clocked_assert`. Those failed controls remain part of the original
qualification denominator. No unsupported assertion, asynchronous reset,
blackbox behavior, program loading, whole-RTL equivalence, physical runtime or
timer role is qualified by this patch. Built plugins, selected target RTL,
private control answers and execution receipts are intentionally absent here.
