# Paper appendix: registered-kind coverage

The literal producer registry and actual pipeline AST have the same eight keys. These are eight registered transformation kinds, not eight proved passes. Scope tags below are contracts, not aliases or additional transformations. This table joins the frozen claim ledger; it does not extend its domains or approve a specification.

| Registered kind | Actual scope tags | Selected claim/evidence families | Relation/domain limit | New complete gate | Graham variant partitions |
| --- | --- | --- | --- | --- | --- |
| shape_view_normalization | identical_memref_cast, static_dimensions | shape-standalone, new-shape-actual-after | Historical possible successful descriptor/world outcomes with trusted target matching; new actual-after bridge unqualified. | Unsupported by new route (source-only determination) | 2 |
| exact_integer_simplification | scalar_signless_wrapping_integers | integer-prefix-chain | Canonical word entry outcomes, including modeled error/world; local canonical-output contracts; no numerical/poison/native fidelity. | Unsupported by new route (source-only determination) | 8 |
| scalar_redundancy_elimination | total_scalar_integer_operations, preserved_unknown_effects, ssa_use_def_and_dominance | scalar-cse-dce-licm, scalar-unroll-segment, integer-prefix-chain, new-live-compiler-gate | Typed total-pure CSE/DCE with retained effects; common finite abstract control. Static no-tail restrictions apply to the longer chains. | Two-kind source route; held/unqualified | 3 |
| scalar_loop_invariant_motion | total_scalar_integer_operations, total_speculation, loop_external_operands, unchanged_effect_order | scalar-cse-dce-licm, scalar-unroll-segment, integer-prefix-chain, new-live-compiler-gate | Total speculation and scoped captures; common finite abstract control/effect contracts. The composed SCF domain is narrower. | Two-kind source route; held/unqualified | 5 |
| ordered_loop_unrolling | static_nonnegative_index_bounds, positive_step, exact_iteration_order, forwarded_loop_carried_state, bounded_operation_growth | scalar-unroll-segment, integer-prefix-chain | Selected chain: static nonnegative positive-stride no-tail bulk. Inline/tail/general nested variants do not inherit that composition. | Unsupported by new route (source-only determination) | 5 |
| ordered_loop_segmentation | static_nonnegative_index_bounds, positive_step, exact_iteration_order, forwarded_loop_carried_state | scalar-unroll-segment, integer-prefix-chain, reuse-segment | Static signed-control/carry bounds in scalar chain; memory chain uses possible public successes under a different relation. | Unsupported by new route (source-only determination) | 3 |
| buffer_reuse | static_contiguous_initialized_integer_buffers, sequential_nonoverlapping_lifetimes | reuse-segment | Initialized private epochs and possible public successes; total/private/unobservable allocation. Finite-capacity failure preservation excluded. | Unsupported by new route (source-only determination) | 5 |
| sequential_dependency_schedule | ssa_dependencies, sequential_effect_order | **Unmapped in selected ledger** | No family in the selected claim ledger. Development/local routes do not establish general full-entry schedule correctness. | Unsupported by new route (source-only determination) | 3 |

All exact roots, domain/observation relations, evidence selectors and statuses are in [REGISTERED_COVERAGE.json](REGISTERED_COVERAGE.json). Historical engineering qualification belongs to its exact original family; corrected admission/gate/actual-after source versions inherit none. Global native permissions are held. Parsing/projection, primitive/effect/control/descriptor/allocation/frame correspondence and human review remain explicit.

Shared family counts are shown once below. Reusing a family under several registered kinds does not multiply its tests, receipts or unique inputs. Null metrics remain unknown.

| Family census | Original tests | Accepted CHECKs | Issued original receipts | Distinct raw tuples | Positive replay records |
| --- | --- | --- | --- | --- | --- |
| scalar-cse-dce-licm | ? | ? | ? | ? | ? |
| scalar-unroll-segment | 64 | ? | 8 | 8 | 8 |
| integer-prefix-chain | 75 | ? | 17 | 16 | 17 |
| shape-standalone | 78 | 14 | ? | 8 | 14 |
| reuse-segment | 71 | 17 | ? | 8 | 17 |
| declared-call-v1 | 46 | ? | 2 | ? | 2 |
| new-live-compiler-gate | ? | ? | ? | ? | ? |
| new-shape-actual-after | ? | ? | ? | ? | ? |

Shape’s14 accepted CHECK records cover8 pairs; memory’s17 cover8 triples in each historical suite. Their78/71 test totals are not receipt counts. Integer’s17 receipts cover16 tuples. Declared-call v1 is a separate supplemental route, not a registry key:46 is its original fixed test roster and2 records form its positive replay association, not a fresh46-test suite. New candidate and scalar-family unknown counts are not zeros.

Graham’s pinned inventory partitions34 registered variants and57 broader mechanisms. Its broader categories are:

| Broader category | Inventory rows |
| --- | --- |
| selector | 1 |
| broader_transformation | 41 |
| helper_plan | 8 |
| candidate_transformation | 2 |
| extension_point | 2 |
| analysis_admission | 1 |
| target_implementation_boundary | 1 |
| emission_boundary | 1 |

The combined91 inventory rows include helpers, candidate transforms, extension/target/emission boundaries and a selector/admission mechanism. They are neither91 passes nor a proved-pass denominator. The57 broader rows remain explicitly outside this fixed registry join; their full row IDs/status boundaries are retained in the JSON and original CSV. No coverage percentage is defined.

Required next evidence/reviews and owners are in [OBLIGATIONS.md](OBLIGATIONS.md). The selected service-profile harness passed12 fixed SDK-free cases; see [service evidence](service/EVIDENCE_TABLE.md). It exercises actual APIs with mocked custody/Context and grants no native qualification or production activation. This table alone does not establish ordinary service enforcement.
