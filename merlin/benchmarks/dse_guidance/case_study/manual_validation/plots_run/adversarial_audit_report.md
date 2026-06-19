# Adversarial audit of the headline conclusions (all)

> Validity ledger: each conclusion → the artifact/metric behind it → whether it survives leave-one-out, macro-vs-micro, and pad-waste threshold perturbation. `robust` = survives all; `winner robust, magnitude fragile` = the ranking holds but the number moves a lot; `blocked-by-capture` = the conclusion cannot be confirmed without a richer capture.

| conclusion | metric class | survives LOO | survives macro/micro | survives threshold | driven by | verdict |
|---|---|---|---|---|---|---|
| primitive-set frontier: a 2-primitive set covers the corpus where one fails | derived | no (flips: ['smolvla']) | yes (macro+micro+worst all reported) | no (winner varies: {'5pct': 'gemv_lane_64+tile_4x16', '10pct': 'gemv_lane_128+tile_4x16', '20pct': 'gemv_lane_128+tile_8x16'}) | specific pair shifts with finer tiles/threshold | **claim robust (a 2-set suffices); specific pair is threshold/LOO-sensitive** |
| dense-GEMM is corpus-narrow (skinny/GEMV dominates the MAC mass) | derived | magnitude NOT stable (max micro swing 0.4679) | winner yes; magnitude no (macro/micro differ widely) | n/a | pi05 | **winner robust, magnitude fragile** |
| resident_weight_object is necessary (K-loop weight residency is a search axis) | derived (weight bytes) + assumed (K) | n/a (per-workload predicate) | n/a | n/a | configured/reference K | **blocked-by-capture (needs a loop-preserving capture to confirm K)** |
| skinny_gemm_or_gemv_engine is necessary/useful across the corpus | derived | see predicate_audit per-workload | yes | n/a | skinny-GEMM projections (not true GEMV) | **robust (name now honest: skinny OR gemv)** |
| low-bit abstractions are blocked (cannot be evaluated) | unavailable | n/a | n/a | n/a | f32 fake-quant capture | **blocked-by-capture** |
| attention/KV abstractions are blocked (attention lowered) | unavailable | n/a | n/a | n/a | flat capture | **blocked-by-capture** |
| bounded_loop_command / loop_carried_state_handle are useful | assumed | n/a | n/a | n/a | configured/reference K | **blocked-by-capture (loop erased; useful, not provable)** |
| capture fidelity is the limiting factor (flat captures erase loop/KV/low-bit) | derived | uniform across corpus | n/a | n/a | capture level (uniform) | **robust** |
| HW/SW boundary placement is itself a DSE search axis | derived | n/a | n/a | n/a | structural enumeration | **descriptive (enumeration, not a single decision)** |

Full supporting artifact + decision-vs-fact per row: `conclusion_validity_table.csv`.

