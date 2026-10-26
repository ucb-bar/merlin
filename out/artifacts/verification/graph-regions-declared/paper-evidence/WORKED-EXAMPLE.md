# Actual retained example: producer, two consumers and outside observer

This reuses the sealed **qualification-v1** original i8 case. It runs no tool and changes no historical evidence. It illustrates the scoped conditional graph/call theorem, not native numerical fidelity or all graph transformations. `QUALIFIED-EXAMPLE-BINDINGS.json` carries the original stored references; absolute Linux receipt paths remain historical, not laptop qualification.

Pinned implementation: `4499eebb959c0f92cefd92cc6d80646ba66474ed`; actual `plan_connected_region` and `compile_source_multioutput`. Domain: static 3×47 i8 pointwise stages, synchronous constant-argument calls, one private producer and two public outputs. The complete source has nine returns. Its following generic reads BOTH outputs.

Readable source sketch (explanatory notation; full unabridged authoritative MLIR is retained in `qualification-v1-originals/verified-i8/capture/source.mlir`):

```mlir
// Each generic has identity maps, parallel iterators and an unread empty output.
%p  = linalg.generic ins(%input, %w0, %state) { %product = arith.muli %x, %w
                                             %r = arith.addi %product, %s }
%c0 = linalg.generic ins(%p, %w1) { %r = arith.addi %v, %w }
%c1 = linalg.generic ins(%p, %w1) { %r = arith.muli %v, %w }
%next = linalg.generic ins(%c0, %c1, %position) { /* unchanged outside observer */ }
func.return %c0, %c1, %next, %nextpos, %input, %w0, %w1, %state, %position
```

The actual changed representation is the schedule **Kernel**, while the complete source MLIR remains identical. The production realization allocates the virtual private intermediate and updates both reads:

| Actual field | Original virtual Kernel | Actual allocated Kernel |
| --- | ---: | ---: |
| `produce.out` | 1024 | 0 |
| `add.ao` | 1024 | 0 |
| `multiply.ao` | 1024 | 0 |
| call order | produce, add, multiply | produce, add, multiply |
| element count / element bytes | 141 / 1 | 141 / 1 |
| two public output selectors | 6, 7 | 6, 7 |

There are exactly three changed scalar argument values. All other fields of the full raw calls, Kernel name/arguments, ordered scalar operands, source context and return observations are retained. The original virtual private offset is not an executable physical-before program. The theorem derives agreement between the logical source region and the actual allocated calls under instruction/memory/ABI contracts; it does not prove physical execution equivalence to an out-of-capacity virtual program.

The authoritative source exports and original planner use distinct numbering systems. Their mapping is checked explicitly, without equating integers by fiat:

| Source site | Planner walk ordinal | Function-body ordinal in Lean | Exported SSA result |
| --- | ---: | ---: | ---: |
| `%p` | 6 | 4 | 9 |
| `%c0` | 10 | 5 | 16 |
| `%c1` | 13 | 6 | 21 |

The actual plan values `[2,0,3,9,1,16,21]` bind input/weight/state and the producer/output identities. Plan outputs `[5,6]` refer to values 16 and 21. This producer is not directly returned; an escaped producer has a separate physical refusal. Logical planner observer retention is a distinct supported result.

Retained original checker output, from `qualification-v1-originals/verified-i8/process.json` (return code 0, empty stderr):

```text
'actual_kernel_decoded' depends on axioms: [propext]
'actual_calls_valid' depends on axioms: [propext, Quot.sound]
'actual_generic_binding' depends on axioms: [propext, Classical.choice, Quot.sound]
'actual_complete_observers' depends on axioms: [propext, Classical.choice, Quot.sound]
'actual_nonzero_witness' depends on axioms: [propext, Quot.sound]
```

The exact integrated root is `actual_complete_observers` in the retained `Check.lean`. Its imported proof chain is ordered scalar execution → tensor stage meanings → per-instruction physical refinement → sequence refinement → derived local composition → existing `checkRealization_native` observer preservation. The conclusion includes source binding and selected scalar executions, alongside complete returned observations. Nonzero witnesses are mathematical common-interpretation witnesses, not native executions.

The fresh batch is retained at `qualification-v1/fresh-replay/0/`; its checker output repeats these exact roots with return code 0. The batch command is `qualification-v1/fresh-process.json`. The fixed 46-case roster, exact-root audits, post-checks and immutable seal are retained in qualification-v1. No command was rerun to write this example.

Negative example: the separately retained scalar declaration changes the producer's final add from operands `[3,2]` to `[3,0]`, preserving types. The authoritative compiler accepted both buffers (`torch-mlir 20261001`, Python 3.12.14); the retained Lean query proves `checkScalar ... = false` and exits 0. Paths: `qualification-v1-originals/scalar-wrong_operand/{EVIDENCE.json,Check.lean,process.json}`. Thus a same-typed wrong source value is refused by the actual Lean checker, without asserting a numerical law.

The old cut still assumes initialized packing, actual instruction fidelity, synchronous successful commit, disjoint complete physical effects, native Action frame/locality and ABI publication. Its original ABI/full post-pass declaration associations were stronger premises than the later wrapper checks. That five-source wrapper's frozen qualification FAILED and has no fresh batch. New captured-source/loader modules are not covered by this example. Frozen Core.Sem identification, runtime/alias/completion/allocator truth, independent review, ordinary compiler gating, full model coverage and the separate gSIM figure remain outside the claim.
