# ISA / capsule coverage report (capsule_bench_v0)

Total capsules: **25**  ·  by kind: {'isa': 12, 'layer': 6, 'model_slice': 7}  ·  by label: {'public': 20, 'hidden': 5}

## Oracle tiers reached (passing)

| tier | capsules passing |
|---|---|
| L0 | 25 |
| L1 | 25 |
| L2 | 25 |
| L3 | 25 |
| L4 | 0 |
| L5 | 0 |

## Instruction-class coverage (explicit not-covered rows)

| class | capsules exercising |
|---|---|
| CONFIG_EX | 23 |
| CONFIG_LD | 25 |
| CONFIG_ST | 25 |
| MVIN | 25 |
| MVOUT | 25 |
| PRELOAD | 23 |
| COMPUTE_PRELOADED | 23 |
| COMPUTE_ACCUMULATE | 0  _(not covered)_ |
| FLUSH | 25 |
| FENCE | 25 |
| LOOP_WS | 0  _(not covered)_ |
| LOOP_CONV | 0  _(not covered)_ |

## Mode coverage

| mode | passing capsules |
|---|---|
| i8 | 4 |
| relu | 5 |
| acc_scale | 4 |
| k_accumulate | 5 |
| resident_reuse | 1 |
| conv2d | 3 |
| movement | 2 |
| padded_edge | 1 |

## Heavy-oracle availability (honest)

- VCS (L4) recorded unavailable on **0** capsules
- FireSim (L5) recorded unavailable on **0** capsules

_Not-run is not pass: a mandatory tier recorded unavailable yields capsule status=incomplete, never pass._
