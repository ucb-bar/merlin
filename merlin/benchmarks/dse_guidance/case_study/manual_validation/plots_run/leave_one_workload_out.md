# Macro/micro + leave-one-workload-out robustness (all)

> Anti-overfitting: each cross-workload finding recomputed dropping one workload. A finding that flips is corpus-specific, not general.

## best_2_primitive_set

- all: ['gemv_lane_64', 'tile_8x16']
- all_worst: 0.9977
- all_macro: 0.9998
- loo_changes_winner: ['smolvla']
- robust: False

## dense_gemm_mac_dominance

- macro: 0.1395
- micro: 0.0397
- micro_loo: {'bitvla': 0.0397, 'groot_n1d7': 0.04, 'molmoact': 0.0398, 'openvla': 0.0397, 'pi05': 0.5076, 'rdt': 0.0253, 'rdt2': 0.0397, 'smolvla': 0.0155, 'tiny_llama': 0.0397, 'xr0': 0.0397}
- collapses_if_removed: []
- note: consistent across views

## residency_pressure_rank

- all: ['molmoact', 'pi05', 'groot_n1d7', 'tiny_llama', 'smolvla', 'rdt', 'rdt2', 'xr0', 'bitvla', 'openvla']
- top: molmoact
- note: absolute bytes are small/random-init; ranking is the robust signal

