# DSE-guidance — commit manifest

The workload-contract analysis workstream (`merlin.dse_guidance`), branch
`feature/kernel-policy-mining`. Companion change in `model2MLIR` (branch
`fix/linear-bias-overrank`, commit `bf0bae0` — `import_fx` emits `prov.fqn`).

Commits (newest first):

```
(this phase)  contract-completeness package: state lifetime, compiler-proof matrix, abstraction
              pressure ranking, workload families, DSE search-space template, measurement priority,
              per-region numerical dtype + honesty fields + TorchAO integration plan
2a85004  presentation-grade package + measured accuracy gate
520d610  reframe to workload-contract analysis (envelope + abstractions)
478cf70  P1-a per-component calibration + P1-b per-dispatch host cost
eb36f7a  measured dispatch coupling (first measured runtime leg)
df0c874  numerical-contract audit + polished cross-workload case study
06ffb3c  cross-workload prov.fqn case study (rdt, openvla, llama x2)
2c79e7c  role recovery on a REAL VLA capture (rdt, prov.fqn)
2c53216  FQN role auto-recovery + multi-point cost calibration
b922e93  Level-1 region attribution from real IR
a449cc5  docs (framing, pipeline order, honest status)
a92fb19  tests + fixtures
b294b9f  VLA DSE-guidance framework (package + schemas + CLI)
```

## Reproduce
```
bash merlin/benchmarks/dse_guidance/reproduce_case_study.sh
```

## Test status
- `merlin/python/tests/test_dse_guidance.py` — all guidance tests pass.
- Bounded repo suite passes except the pre-existing, unrelated
  `test_precision.py::test_f32_accumulation_is_far_more_accurate` NaN (in the `llvmlower`
  f32-accumulation path; fails identically with all dse-guidance changes reverted).

## Discipline
No file claims a speedup, cycle, or energy number for unbuilt hardware. Calibration against an
existing target is demoted to a sanity anchor. Every reported number carries an evidence label.
